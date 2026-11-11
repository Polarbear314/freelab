#!/usr/bin/env python3
"""
FreeLAB harness -- a local Streamlit app on top of the FreeLAB pipeline.

Same three modes as the pipeline, all running against your local Ollama:

  1. Chat        -- ask the cascade general questions (no document).
  2. Document QA -- upload a PDF / TXT / MD / DOCX; it chunks + embeds the file
                    ONCE at upload, caches the vectors in session state, then
                    answers questions fast (only the question is re-embedded).
  3. Batch       -- upload a file of many questions (+ an optional document),
                    answer them all, download the results as CSV.

All the real logic lives in `freelab_core.py`; this file is just the UI, so it
stays in sync with the CLI and the Gradio app.

RUN
  pip install streamlit pypdf python-docx
  ollama pull qwen3:0.6b && ollama pull qwen3:8b && ollama pull nomic-embed-text
  streamlit run freelab_streamlit.py
"""

import csv
import io
import json
import os
import pickle
import re
import tempfile
import urllib.error

import streamlit as st

import freelab_core as fl

EMBED_BATCH = 64        # embed chunks in batches so large docs show progress
SOURCE_PREVIEW = 400    # chars of each source chunk to show
GRAPH_SIM_THRESHOLD = 0.7   # cosine above which two chats get a similarity link
                            # (higher = only clearly-related topics connect)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freelab_data")
STATE_PATH = os.path.join(DATA_DIR, "state.pkl")


# --- persistence: the memory layer (survives restarts) -------------------------
def save_state():
    """Best-effort pickle of all conversations (messages, docs, sources)."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(STATE_PATH, "wb") as f:
            pickle.dump({"conversations": st.session_state.conversations,
                         "current": st.session_state.get("current")}, f)
    except Exception:  # noqa: BLE001 -- persistence must never crash the app
        pass


def load_state():
    """Load saved conversations. Returns True if state was restored."""
    if not os.path.exists(STATE_PATH):
        return False
    try:
        with open(STATE_PATH, "rb") as f:
            data = pickle.load(f)
        convs = data.get("conversations") or []
        if not convs:
            return False
        st.session_state.conversations = convs
        st.session_state.current = data.get("current") or convs[-1]["id"]
        return True
    except Exception:  # noqa: BLE001
        return False

USER_AVATAR = "🧑"
BOT_AVATAR = "⚡"

# ChatGPT/Claude-style polish: narrow centered column, soft message bubbles,
# quiet chrome. Structural only -- base colors come from .streamlit/config.toml.
CSS = """
<style>
/* hide streamlit chrome for a cleaner, app-like frame */
#MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
footer { visibility: hidden; height: 0; }

/* centered canvas; wide enough for chat on the left + sources docked right */
.block-container {
  max-width: 1300px;
  margin-left: auto; margin-right: auto;
  padding-top: 3.5rem; padding-bottom: 7rem;
}

/* thin vertical rule between the chat and the sources panel */
.vrule { border-left: 1px solid #E5E3DB; height: 74vh; width: 1px; margin: 0 auto; }

/* system font stack */
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}

/* branded header */
.fl-header { text-align: center; margin: .25rem 0 1.25rem; }
.fl-title  { font-size: 1.9rem; font-weight: 700; letter-spacing: -.02em; color: #2B2B28; }
.fl-sub    { font-size: .92rem; color: #8A8A82; margin-top: .15rem; }

/* chat bubbles */
[data-testid="stChatMessage"] { background: transparent; padding: .15rem 0; }
[data-testid="stChatMessageContent"] {
  background: #F4F3EE;
  border: 1px solid #EAE8E0;
  border-radius: 16px;
  padding: .85rem 1.1rem;
  line-height: 1.55;
}

/* rounded chat input */
[data-testid="stChatInput"] { border-radius: 16px; }
[data-testid="stChatInput"] textarea { font-size: 1rem; }

/* subtle tab + sidebar refinement */
button[data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

/* keep the settings sidebar pinned open and visible (Streamlit was
   auto-collapsing it, and the expand arrow lives in the hidden toolbar) */
section[data-testid="stSidebar"] {
  border-right: 1px solid #EEEDE7;
  transform: none !important;
  visibility: visible !important;
  min-width: 300px !important;
  width: 300px !important;
  margin-left: 0 !important;
}
</style>
"""

HEADER = """
<div class="fl-header">
  <div class="fl-title">FreeLAB</div>
  <div class="fl-sub">Small model drafts · big model verifies · runs locally</div>
</div>
"""


# --- helpers -------------------------------------------------------------------
def _tmp_path(uploaded):
    """Persist a Streamlit UploadedFile to a temp path (pypdf/docx need a path)."""
    suffix = os.path.splitext(uploaded.name)[1]
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def ingest_uploaded(uploaded):
    """Chunk + embed a document ONCE. Shows a progress bar over the batches.

    Scanned PDFs are OCR'd first (with their own progress bar) via the core.
    """
    path = _tmp_path(uploaded)
    ocr = {"bar": None}

    def ocr_progress(done, total):
        if ocr["bar"] is None:
            ocr["bar"] = st.progress(0.0, text="OCR-ing scanned pages...")
        ocr["bar"].progress(done / total, text=f"OCR-ing scanned pages... {done}/{total}")

    try:
        text = fl.extract_text(path, ocr_progress=ocr_progress)
    finally:
        os.unlink(path)
        if ocr["bar"] is not None:
            ocr["bar"].empty()
    if not text.strip():
        raise ValueError("No text found in this document, even after OCR.")
    chunks = fl.chunk_text(text)
    vecs = []
    bar = st.progress(0.0, text="Embedding chunks...")
    n = len(chunks)
    for start in range(0, n, EMBED_BATCH):
        vecs.extend(fl.embed(chunks[start:start + EMBED_BATCH]))
        bar.progress(min((start + EMBED_BATCH) / n, 1.0), text="Embedding chunks...")
    bar.empty()
    return {"name": uploaded.name, "chunks": chunks, "vecs": vecs,
            "orig_tok": fl.approx_tokens(text)}


def stats_line(index, kept, path, score, tokens):
    d, j, g = tokens
    kept_tok = fl.approx_tokens(fl._context(kept))
    orig_tok = index["orig_tok"]
    pct = 100 - 100 * kept_tok // max(orig_tok, 1)
    verdict = f" (judge score {score})" if score is not None else ""
    return (f"compression: {len(index['chunks'])} → {len(kept)} chunks, "
            f"~{orig_tok} → ~{kept_tok} tokens ({pct}% smaller) · "
            f"path: **{path}**{verdict} · tokens: draft={d} judge={j} biggen={g}")


def _highlight(text, query):
    """Bold the query's meaningful words in a source passage (markdown)."""
    terms = sorted({w for w in re.findall(r"[A-Za-z0-9']+", query.lower())
                    if len(w) > 3}, key=len, reverse=True)
    if not terms:
        return text
    pat = re.compile(r"(?i)\b(" + "|".join(re.escape(t) for t in terms) + r")\b")
    return pat.sub(r"**\1**", text)


@st.cache_data(show_spinner=False)
def render_page(pdf_bytes, page, dpi=150):
    """Render one PDF page to PNG bytes (cached by page). Needs PyMuPDF."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = doc.load_page(page - 1).get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def render_sources_panel(conv):
    """Sources panel with two views: the passages the AI cited, and the full
    document to browse — with the cited pages clearly flagged."""
    st.markdown("#### Sources")
    srcs = conv.get("last_sources") or []
    index = conv.get("index")
    if not srcs and not index:
        st.caption("Ask about an attached document and its source passages — "
                   "with page numbers and the actual page image — show up here. "
                   "You'll also be able to browse the whole document.")
        return
    tab_cited, tab_full = st.tabs(["⭐ Cited", "📄 Full document"])
    with tab_cited:
        _render_cited(conv, srcs)
    with tab_full:
        _render_full_document(conv, index, srcs)


def _render_cited(conv, srcs):
    """The exact passages the AI used to answer (page image + highlighted text)."""
    if not srcs:
        st.caption("Ask a question and the passages the AI used appear here.")
        return
    query = conv.get("last_query", "")
    pdf_bytes = conv.get("pdf_bytes")
    st.caption(f"{len(srcs)} passage(s) the AI used to answer:")
    with st.container(height=500):                # own scroll area
        for n, s in enumerate(srcs):
            loc = f"chunk {s['chunk']}"
            if s.get("page"):
                loc = f"page {s['page']} · " + loc
            with st.expander(f"{loc}  ·  score {s['score']:.2f}",
                             expanded=(n == 0)):
                if s.get("page") and pdf_bytes:
                    try:
                        st.image(render_page(pdf_bytes, s["page"]),
                                 use_container_width=True)
                    except Exception:  # noqa: BLE001 -- render is best-effort
                        pass
                st.markdown(_highlight(s["text"], query))


def _render_full_document(conv, index, srcs):
    """Browse the entire document; pages/passages the AI cited are flagged."""
    if not index:
        st.caption("Attach a document to browse it here.")
        return
    cited_pages = sorted({s["page"] for s in srcs if s.get("page")})
    pdf_bytes = conv.get("pdf_bytes")
    n_pages = index.get("pages", 1)

    if pdf_bytes and n_pages:                     # single PDF -> scroll all pages
        if cited_pages:
            st.caption("✅ Pages the AI drew from: "
                       + ", ".join(str(p) for p in cited_pages)
                       + " — scroll to see the whole document.")
        else:
            st.caption("Scroll through the whole document below.")
        with st.container(height=560):            # one scrollable stack of pages
            for p in range(1, n_pages + 1):
                used = p in cited_pages
                st.markdown(f"**Page {p}**"
                            + ("  ·  ✅ used to answer" if used else ""))
                try:
                    st.image(render_page(pdf_bytes, p), use_container_width=True)
                except Exception:  # noqa: BLE001
                    st.caption("(couldn't render this page)")
                st.divider()
    else:                                         # non-PDF -> full text by chunk
        cited_idx = {s["chunk"] for s in srcs}
        if cited_idx:
            st.caption("⭐ marks the chunks the AI used to answer.")
        with st.container(height=500):
            for i, ch in enumerate(index["chunks"], start=1):
                mark = "⭐ " if i in cited_idx else ""
                st.markdown(f"{mark}**chunk {i}**")
                st.markdown(ch)


def ingest_files(files):
    """Ingest one or more attached files into a single combined index (embed once).

    Scanned PDFs are OCR'd per file with their own progress bar.
    """
    names, all_pages = [], []
    for uploaded in files:
        path = _tmp_path(uploaded)
        ocr = {"bar": None}

        def ocr_progress(done, total, _o=ocr, _n=uploaded.name):
            if _o["bar"] is None:
                _o["bar"] = st.progress(0.0, text=f"OCR {_n}…")
            _o["bar"].progress(done / total, text=f"OCR {_n} {done}/{total}")

        try:
            pages = fl.extract_pages(path, ocr_progress=ocr_progress)
        finally:
            os.unlink(path)
            if ocr["bar"] is not None:
                ocr["bar"].empty()
        if any(p.strip() for p in pages):
            all_pages.extend(pages)
            names.append(uploaded.name)
    if not all_pages:
        raise ValueError("No text found in the attached file(s), even after OCR.")

    combined = "\n".join(all_pages)
    chunks = fl.chunk_text(combined)
    pageno = fl.chunk_pages(all_pages)          # 1-based source page per chunk
    vecs = []
    bar = st.progress(0.0, text="Embedding chunks…")
    n = len(chunks)
    for start in range(0, n, EMBED_BATCH):
        vecs.extend(fl.embed(chunks[start:start + EMBED_BATCH]))
        bar.progress(min((start + EMBED_BATCH) / n, 1.0), text="Embedding chunks…")
    bar.empty()
    name = names[0] if len(names) == 1 else f"{len(names)} files"
    # keep the raw PDF bytes for the page-image source view -- only for a single
    # PDF, where chunk page numbers map 1:1 onto that file's pages
    pdf_bytes = None
    if len(files) == 1 and files[0].name.lower().endswith(".pdf"):
        pdf_bytes = bytes(files[0].getvalue())
    return {"name": name, "chunks": chunks, "vecs": vecs, "pageno": pageno,
            "pages": len(all_pages), "orig_tok": fl.approx_tokens(combined),
            "pdf_bytes": pdf_bytes}


OLLAMA_DOWN = "⚠️ Could not reach Ollama at localhost:11434. Is `ollama serve` running?"


def answer_query(text, conv, keep, threshold, use_cascade, web_lookup):
    """Route one question: document-grounded > web lookup > plain local chat.

    Source passages are stored on the CONVERSATION (conv["last_sources"] etc.),
    so each chat keeps its own sources and nothing carries over between chats.
    """
    index = conv.get("index")
    if index:                                   # a file is attached -> answer it
        kept, scores = fl.rank_chunks(text, index["chunks"], index["vecs"], keep)
        pageno = index.get("pageno") or []
        kept_pages = [pageno[i] if i < len(pageno) else None for sc, i in scores]
        multipage = index.get("pages", 1) > 1 and any(kept_pages)
        if multipage:                           # page-labeled context + inline cites
            final, path, score, tokens = fl.answer_from_chunks_doc(
                text, kept, kept_pages, threshold=threshold, use_cascade=use_cascade)
        else:
            final, path, score, tokens = fl.answer_from_chunks(
                text, kept, threshold=threshold, use_cascade=use_cascade)
        conv["last_sources"] = [
            {"chunk": i + 1, "score": sc,
             "page": pageno[i] if i < len(pageno) else None,
             "text": index["chunks"][i]}
            for sc, i in scores]
        conv["last_query"] = text
        conv["pdf_bytes"] = index.get("pdf_bytes")
        final += "\n\n" + stats_line(index, kept, path, score, tokens)
        cited_pages = sorted({p for p in kept_pages if p})
        if multipage and cited_pages:
            final += (f"\n\n*📄 Sourced from page(s): "
                      f"{', '.join(str(p) for p in cited_pages)} — "
                      f"see the Sources panel.*")
        else:
            final += (f"\n\n*{len(scores)} source passage(s) → see the "
                      f"**Sources** panel.*")
        return final

    conv["last_sources"] = []                   # non-doc answer -> clear panel
    conv["pdf_bytes"] = None
    if web_lookup:
        final, path, score, _, sources = fl.web_answer(
            text, keep=keep, threshold=threshold, use_cascade=use_cascade)
        if sources:
            links = "\n".join(f"- [{s['title'] or s['url']}]({s['url']})"
                              for s in sources if s.get("url"))
            final += f"\n\n**Sources**\n{links}"
        return final
    final, path, score, _ = fl.chat_answer(
        text, threshold=threshold, use_cascade=use_cascade)
    return final


# --- conversation history (just stored text; no effect on model compute) ------
def _init_conversations():
    ss = st.session_state
    if "conversations" not in ss:               # first run this session -> load disk
        load_state()
    if not ss.get("conversations"):
        ss.conversations = [{"id": "c1", "title": "New chat", "history": []}]
        ss.current = "c1"
    ss.setdefault("current", ss.conversations[-1]["id"])


def _conv():
    """The currently selected conversation dict."""
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.current:
            return c
    return st.session_state.conversations[-1]


def _new_conversation():
    import time
    cid = f"c{int(time.time() * 1000)}"
    st.session_state.conversations.append(
        {"id": cid, "title": "New chat", "history": []})
    st.session_state.current = cid
    save_state()


def _sidebar_history():
    st.markdown("### Chat history")
    if st.button("➕ New chat", use_container_width=True):
        _new_conversation()
        st.rerun()
    # scrollable box so a long list of chats doesn't grow the sidebar forever
    with st.container(height=240):
        for c in reversed(st.session_state.conversations):
            label = c["title"] or "New chat"
            label = label if len(label) <= 32 else label[:32] + "…"
            active = c["id"] == st.session_state.current
            if st.button(("🟢 " if active else "💬 ") + label,
                         key=f"conv_{c['id']}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.current = c["id"]
                save_state()
                st.rerun()


# --- settings (stored in session_state so they persist across views) ----------
def _init_settings():
    ss = st.session_state
    ss.setdefault("keep", fl.DEFAULT_KEEP)
    ss.setdefault("threshold", fl.DEFAULT_THRESHOLD)
    ss.setdefault("use_cascade", True)
    ss.setdefault("web_lookup", False)
    ss.setdefault("show_sources", True)
    ss.setdefault("m_draft", fl.DRAFTER)
    ss.setdefault("m_verify", fl.VERIFIER)
    ss.setdefault("m_embed", fl.EMBED_MODEL)
    # push model names into the core each run so they persist across views
    fl.DRAFTER, fl.VERIFIER, fl.EMBED_MODEL = ss.m_draft, ss.m_verify, ss.m_embed


def _settings_view(conv):
    """The Settings screen — only shown when the user picks the Settings tab.

    Uses the value=/assign pattern (not widget `key`s) so the settings live in
    plain session_state and survive switching away from this view — binding
    keys directly would let Streamlit clear them when the widgets leave the page.
    """
    ss = st.session_state
    st.subheader("⚙️ Settings")
    ss.keep = st.slider("keep (chunks retained)", 1, 20, ss.keep)
    ss.threshold = st.slider("threshold (judge score to keep draft)",
                             0, 100, ss.threshold, 5)
    ss.use_cascade = st.checkbox("use cascade (draft → judge)",
                                 value=ss.use_cascade)
    ss.web_lookup = st.checkbox("🌐 Web lookup (uses internet)",
                                value=ss.web_lookup,
                                help="Off = fully local, nothing leaves your "
                                     "machine. On = chat searches the web and "
                                     "cites sources.")
    ss.show_sources = st.checkbox("📑 Show sources panel", value=ss.show_sources,
                                  help="Hide to give the chat the full width.")
    with st.expander("Models"):
        ss.m_draft = st.text_input("drafter", ss.m_draft)
        ss.m_verify = st.text_input("verifier / judge", ss.m_verify)
        ss.m_embed = st.text_input("embedding model", ss.m_embed)
    if st.button("Warm models"):
        with st.spinner("warming models…"):
            try:
                fl.warmup(log=None)
                st.success("🟢 Models warm")
            except urllib.error.URLError:
                st.error(OLLAMA_DOWN)
            except Exception as e:  # noqa: BLE001
                st.error(f"Warmup failed: {e}")

    if conv.get("index"):
        st.divider()
        st.markdown(f"📎 **Active document:** {conv['index']['name']} — "
                    f"{len(conv['index']['chunks'])} chunks")
        if st.button("Clear document"):
            for k in ("index", "last_sources", "last_query", "pdf_bytes"):
                conv.pop(k, None)
            save_state()
            st.rerun()

    st.divider()
    _batch_sidebar(ss.keep, ss.threshold, ss.use_cascade)


def _chat_view(conv):
    """The Chat screen — transcript, bottom input with file attach, sources."""
    ss = st.session_state
    keep, threshold = ss.keep, ss.threshold
    use_cascade, web_lookup, show_sources = (ss.use_cascade, ss.web_lookup,
                                             ss.show_sources)
    history = conv["history"]

    if show_sources:
        col_chat, col_div, col_src = st.columns([3, 0.15, 1.25], gap="small")
        with col_div:
            st.markdown("<div class='vrule'></div>", unsafe_allow_html=True)
    else:
        col_chat = st.container()
        col_src = None

    with col_chat:
        for m in history:
            avatar = USER_AVATAR if m["role"] == "user" else BOT_AVATAR
            with st.chat_message(m["role"], avatar=avatar):
                st.markdown(m["content"])

    # chat input stays at top level so it pins to the bottom of the screen
    submitted = st.chat_input("Message FreeLAB…  (attach a file with the 📎)",
                              accept_file="multiple",
                              file_type=["pdf", "txt", "md", "docx"])
    if submitted:
        text = getattr(submitted, "text", "") or ""
        files = list(getattr(submitted, "files", []) or [])

        shown = text
        if files:
            shown = (shown + "\n\n" if shown else "") + \
                    "📎 " + ", ".join(f.name for f in files)
        history.append({"role": "user", "content": shown})
        if conv["title"] == "New chat" and shown.strip():   # name the conversation
            conv["title"] = (text.strip() or files[0].name)[:40]

        with col_chat:
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(shown)
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                final = ""
                if files:                       # index attachments first (embed once)
                    with st.spinner("indexing attachment…"):
                        try:
                            conv["index"] = ingest_files(files)
                        except urllib.error.URLError:
                            final = OLLAMA_DOWN
                        except ValueError as e:
                            final = f"⚠️ {e}"
                index = conv.get("index")

                if not final:
                    if not text.strip() and files and index:
                        final = (f"Indexed **{index['name']}** — "
                                 f"{len(index['chunks'])} chunks across "
                                 f"{index.get('pages', 1)} page(s), embedded once. "
                                 f"Ask me anything about it.")
                    elif not text.strip():
                        final = "Type a message, or attach a file with the 📎."
                    else:
                        spin = ("reading your document…" if index
                                else "searching the web…" if web_lookup
                                else "thinking…")
                        with st.spinner(spin):
                            try:
                                final = answer_query(text, conv, keep, threshold,
                                                     use_cascade, web_lookup)
                            except fl.WebUnavailable as e:
                                final = f"⚠️ {e}"
                            except urllib.error.URLError:
                                final = OLLAMA_DOWN
                st.markdown(final)
        history.append({"role": "assistant", "content": final})
        save_state()                            # persist the memory layer

    if col_src is not None:
        with col_src:
            render_sources_panel(conv)


# --- memory graph --------------------------------------------------------------
def _conv_text(c):
    """Representative text for a conversation (title + the user's messages)."""
    parts = [c.get("title", "")]
    parts += [m["content"] for m in c.get("history", []) if m["role"] == "user"]
    return "\n".join(p for p in parts if p).strip()


def _conv_embedding(c):
    """Embed a conversation once (cached on the conv, refreshed if it grew).
    Returns None if there's nothing to embed or Ollama is unreachable."""
    text = _conv_text(c)
    if not text:
        return None
    n = len(c.get("history", []))
    if c.get("_embed") is not None and c.get("_embed_n") == n:
        return c["_embed"]
    try:
        v = fl.embed([text])[0]
    except Exception:  # noqa: BLE001 -- graph still works without semantic links
        return None
    c["_embed"], c["_embed_n"] = v, n
    return v


_CLUSTER_PALETTE = ["#D97757", "#6C8EBF", "#7FB069", "#B07BAC", "#E0A458",
                    "#5FA8A0", "#C46D5E", "#8A8FBD", "#9C8B6E"]


def _build_graph():
    """Neutral node/edge model for the memory graph.

    nodes: {id, kind: 'chat'|'doc', label}
    edges: {u, v, weight, kind: 'doc'|'sim'} — weight drives the layout, so
    more-similar chats sit closer and unrelated ones drift apart.
    """
    convs = st.session_state.conversations
    nodes, edges, docs = [], [], {}
    for c in convs:
        if not c.get("history"):
            continue
        nodes.append({"id": f"chat:{c['id']}", "kind": "chat",
                      "label": (c.get("title") or "New chat")[:26]})
        idx = c.get("index")
        if idx:
            docs[idx["name"]] = True
            edges.append({"u": f"chat:{c['id']}", "v": f"doc:{idx['name']}",
                          "weight": 0.6, "kind": "doc"})
    for name in docs:
        nodes.append({"id": f"doc:{name}", "kind": "doc", "label": name[:26]})

    embs = {c["id"]: _conv_embedding(c) for c in convs if c.get("history")}
    embs = {k: v for k, v in embs.items() if v}
    ids = list(embs)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = fl.cosine(embs[ids[i]], embs[ids[j]])
            if sim >= GRAPH_SIM_THRESHOLD:      # only genuinely related chats link
                edges.append({"u": f"chat:{ids[i]}", "v": f"chat:{ids[j]}",
                              "weight": float(sim), "kind": "sim"})
    return nodes, edges


def _graph_view(conv):
    """Interactive memory graph (Plotly + networkx): chats clustered by topic
    and shared documents; tap a chat node to open it."""
    st.subheader("🕸 Memory graph")
    convs = st.session_state.conversations
    if not any(c.get("history") for c in convs):
        st.caption("Start chatting and your conversations show up here — "
                   "clustered by topic and by the documents they share. "
                   "Tap a chat node to open it.")
        return
    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError:
        st.info("The memory graph needs two libraries:  "
                "`pip install plotly networkx`")
        return

    nodes, edges = _build_graph()
    graph = nx.Graph()
    for n in nodes:
        graph.add_node(n["id"])
    for e in edges:
        graph.add_edge(e["u"], e["v"], weight=e["weight"])
    pos = nx.spring_layout(graph, weight="weight", seed=42,
                           k=1.2 / max(len(nodes), 1) ** 0.5)

    # colour chats by cluster (connected component); documents stay neutral
    comp_color = {}
    for i, comp in enumerate(nx.connected_components(graph)):
        col = _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)]
        for nid in comp:
            comp_color[nid] = col

    edge_x, edge_y = [], []
    for e in edges:
        x0, y0 = pos[e["u"]]
        x1, y1 = pos[e["v"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none",
                            line=dict(color="#DAD7CD", width=1))

    xs, ys, colors, symbols, sizes, texts, ids = [], [], [], [], [], [], []
    for n in nodes:
        x, y = pos[n["id"]]
        xs.append(x)
        ys.append(y)
        texts.append(n["label"])
        ids.append(n["id"])
        if n["kind"] == "doc":
            colors.append("#9AA7B8")
            symbols.append("square")
            sizes.append(15)
        else:
            colors.append(comp_color.get(n["id"], "#D97757"))
            symbols.append("circle")
            sizes.append(20)
    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers+text", text=texts, textposition="top center",
        textfont=dict(size=11, color="#2B2B28"), customdata=ids,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=sizes, color=colors, symbol=symbols,
                    line=dict(width=1.5, color="white")))

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(showlegend=False, height=520, dragmode="pan",
                      clickmode="event+select", plot_bgcolor="white",
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.caption("🟠 chats (coloured by topic cluster) · ⬜ documents · "
               "closer = more related.  Tap a chat node to open it.")
    # rotating key: after opening a chat we bump the nonce so the next graph
    # render is a fresh chart with no stale selection to re-fire
    nonce = st.session_state.setdefault("_graph_nonce", 0)
    event = st.plotly_chart(fig, width="stretch", on_select="rerun",
                            key=f"memgraph{nonce}")

    # tap-to-open: a selected chat node opens that conversation
    try:
        pts = (event or {}).get("selection", {}).get("points", [])
    except Exception:  # noqa: BLE001
        pts = []
    for p in pts:
        cd = p.get("customdata")
        nid = cd[0] if isinstance(cd, (list, tuple)) else cd
        if isinstance(nid, str) and nid.startswith("chat:"):
            st.session_state.current = nid.split("chat:", 1)[1]
            st.session_state._pending_view = "💬 Chat"
            st.session_state._graph_nonce = nonce + 1
            st.rerun()

    labels = {(c.get("title") or "New chat"): c["id"]
              for c in convs if c.get("history")}
    with st.expander("…or open one from a list"):
        pick = st.selectbox("open", list(labels), label_visibility="collapsed")
        if st.button("Open in Chat", type="primary"):
            st.session_state.current = labels[pick]
            st.session_state._pending_view = "💬 Chat"
            st.rerun()


# --- app -----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="FreeLAB", page_icon="⚡", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    _init_conversations()
    _init_settings()
    conv = _conv()                  # active conversation (owns its doc/sources)

    # sidebar is now just the conversation list
    with st.sidebar:
        _sidebar_history()

    # a view switch requested elsewhere is applied BEFORE the radio exists
    # (Streamlit forbids writing a widget's key after the widget is created)
    if "_pending_view" in st.session_state:
        st.session_state.view = st.session_state.pop("_pending_view")

    # top nav (very top of the page): Chat · Graph · Settings
    view = st.radio("view", ["💬 Chat", "🕸 Graph", "⚙️ Settings"], horizontal=True,
                    label_visibility="collapsed", key="view")
    st.markdown(HEADER, unsafe_allow_html=True)
    if view.endswith("Settings"):
        _settings_view(conv)
    elif "Graph" in view:
        _graph_view(conv)
    else:
        _chat_view(conv)


def _batch_sidebar(keep, threshold, use_cascade):
    """Batch mode, tucked into the sidebar so the main screen stays pure chat."""
    with st.expander("📄 Batch mode"):
        qf = st.file_uploader("Questions (.txt / .csv)",
                              type=["txt", "csv"], key="qf")
        bdoc = st.file_uploader("Optional document",
                                type=["pdf", "txt", "md", "docx"], key="bdoc")
        if st.button("Run batch", type="primary"):
            if not qf:
                st.warning("Upload a questions file first.")
                return
            qpath = _tmp_path(qf)
            try:
                questions = fl.read_questions(qpath)
            finally:
                os.unlink(qpath)
            if not questions:
                st.warning("No questions found.")
                return
            index = None
            if bdoc is not None:
                try:
                    index = ingest_files([bdoc])
                except urllib.error.URLError:
                    st.error(OLLAMA_DOWN)
                    return
            rows = []
            bar = st.progress(0.0, text="Answering…")
            for i, qq in enumerate(questions):
                try:
                    kept = (fl.rank_chunks(qq, index["chunks"], index["vecs"], keep)[0]
                            if index else [qq])
                    final, path, score, _ = fl.answer_from_chunks(
                        qq, kept, threshold=threshold, use_cascade=use_cascade)
                except urllib.error.URLError:
                    st.error(OLLAMA_DOWN)
                    return
                rows.append({"question": qq, "answer": final, "path": path,
                             "judge_score": score if score is not None else "",
                             "chunks_used": len(kept)})
                bar.progress((i + 1) / len(questions), text="Answering…")
            bar.empty()
            st.success(f"Answered {len(rows)} question(s).")
            st.dataframe(rows, use_container_width=True)
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=["question", "answer", "path",
                                                "judge_score", "chunks_used"])
            w.writeheader()
            w.writerows(rows)
            st.download_button("Download CSV", buf.getvalue(),
                               file_name="freelab_batch_results.csv",
                               mime="text/csv")


if __name__ == "__main__":
    main()

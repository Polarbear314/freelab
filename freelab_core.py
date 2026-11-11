#!/usr/bin/env python3
"""
FreeLAB core -- the shared pipeline, one source of truth.

Everything that actually does the work lives here: Ollama plumbing, embeddings,
query-aware compression, and the draft -> judge -> answer cascade. The CLI
(`freelab.py`), the eval harness, and the Gradio app (`freelab_app.py`) all
import from this module so there is no duplicated logic.

Pipeline (see README for the full rationale):

    [1] COMPRESS  embedding model keeps only query-relevant chunks
    [2] DRAFT     small LLM answers from the kept chunks
    [3] JUDGE     big LLM verifies the draft cheaply (a few tokens)
    [4] ANSWER    big LLM writes the full answer only when the draft is escalated
"""

import csv
import json
import math
import os
import re
import time
import urllib.request
from datetime import datetime

OLLAMA = "http://localhost:11434"
DRAFTER = "qwen3:0.6b"
VERIFIER = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_KEEP = 5
DEFAULT_THRESHOLD = 70      # judge score >= this keeps the draft
JUDGE_MAX_TOKENS = 128      # enough to verify the key step, far less than full reasoning
DRAFT_MAX_TOKENS = 350


# --- Ollama plumbing -----------------------------------------------------------
def _post(path, payload, timeout=600):
    req = urllib.request.Request(
        OLLAMA + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(inputs, model=None):
    """Embed a list of strings. Returns a list of vectors (one per input).

    `model` defaults to the current EMBED_MODEL global (read at call time, so
    the app can change it at runtime via the settings sidebar).
    """
    return _post("/api/embed",
                 {"model": model or EMBED_MODEL, "input": inputs})["embeddings"]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CTRL_TOKEN_RE = re.compile(r"/(?:no_)?think\b")


def _clean(text):
    """Strip qwen3 control tokens / thinking blocks that sometimes leak out."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _CTRL_TOKEN_RE.sub("", text)
    return text.strip()


def generate(prompt, model, think=False, num_predict=None):
    opts = {"temperature": 0.2}
    if num_predict:
        opts["num_predict"] = num_predict
    body = _post("/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        "think": think, "keep_alive": "10m", "options": opts,
    })
    return _clean(body.get("response") or ""), body.get("eval_count") or 0


def current_date_str():
    """Today's date, read from the OS clock (no network needed)."""
    return datetime.now().strftime("%A, %B %-d, %Y")


# --- file -> text (shared by the front-ends) -----------------------------------
class OcrUnavailable(ValueError):
    """A scanned PDF needs OCR but the OCR tooling isn't installed.

    Subclasses ValueError so the UIs' existing `except ValueError` handlers
    surface the (helpful) message without extra wiring.
    """


def _pdf_text_layer(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _ocr_pdf_pages(path, dpi=200, ocr_progress=None):
    """OCR a scanned PDF: render each page to an image and read it with Tesseract.

    Returns a list of page texts. Requires `pip install pymupdf pytesseract
    pillow` plus the Tesseract engine (`brew install tesseract` on macOS).
    Raises OcrUnavailable with guidance if any piece is missing.
    `ocr_progress(done, total)` is called per page.
    """
    try:
        import io
        import fitz                       # PyMuPDF (pip install pymupdf)
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise OcrUnavailable(
            "This looks like a scanned PDF, which needs OCR. Install the OCR "
            "extras:  pip install pymupdf pytesseract pillow  (and the engine: "
            "brew install tesseract).") from e

    doc = fitz.open(path)
    total = doc.page_count
    pages = []
    try:
        for i in range(total):
            pix = doc.load_page(i).get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            try:
                pages.append(pytesseract.image_to_string(img))
            except pytesseract.TesseractNotFoundError as e:
                raise OcrUnavailable(
                    "Scanned PDF: the Tesseract OCR engine isn't installed. "
                    "On macOS run:  brew install tesseract") from e
            if ocr_progress:
                ocr_progress(i + 1, total)
    finally:
        doc.close()
    return pages


def extract_pages(path, ocr_progress=None):
    """Return a document as a list of per-page texts.

    Page granularity is what lets the UI cite which page a passage came from.
    PDFs use their text layer when present, else OCR; DOCX/TXT/MD have no real
    pages, so they come back as a single-element list.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if len(_pdf_text_layer(path).strip()) >= 20:   # real text layer
            from pypdf import PdfReader
            return [(p.extract_text() or "") for p in PdfReader(path).pages]
        return _ocr_pdf_pages(path, ocr_progress=ocr_progress)
    if ext == ".docx":
        import docx
        d = docx.Document(path)
        return ["\n".join(p.text for p in d.paragraphs)]
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [f.read()]


def extract_text(path, ocr_progress=None):
    """Plain text of a PDF / DOCX / TXT / MD file (pages joined).

    Thin wrapper over extract_pages so the CLI, batch, and Gradio paths keep a
    simple string interface while the Streamlit app can use pages for citations.
    """
    return "\n".join(extract_pages(path, ocr_progress=ocr_progress))


def read_questions(path):
    """One question per line (.txt) or first column (.csv)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        questions = []
        with open(path, encoding="utf-8", errors="ignore", newline="") as f:
            for row in csv.reader(f):
                if row and row[0].strip():
                    questions.append(row[0].strip())
        return questions
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]


# --- [1] compression -----------------------------------------------------------
CHUNK_SIZE = 120
CHUNK_OVERLAP = 20


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks


def chunk_pages(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Map each chunk of the combined page text to its 1-based source page.

    Chunks are word-windows over `"\\n".join(pages)`; newlines add no words, so
    per-page word counts line up with chunk_text's indexing. Returns a list the
    same length as chunk_text(combined), giving the page each chunk starts on.
    """
    import bisect
    starts, c = [], 0
    for p in pages:
        starts.append(c)
        c += len(p.split())
    total, pageno, i = c, [], 0
    while i < total:
        pageno.append(bisect.bisect_right(starts, i))   # 1-based page number
        i += size - overlap
    return pageno


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rank_chunks(query, chunks, chunk_vecs, keep):
    """Rank pre-embedded chunks against a query. Embeds ONLY the query.

    This is the embed-once path: the document's chunk vectors are computed once
    at ingest time and reused here for every question. Returns (kept_chunks,
    top) where top is a list of (score, index) for the retained chunks.
    """
    qv = embed([query])[0]
    scored = sorted(((cosine(qv, cv), i) for i, cv in enumerate(chunk_vecs)),
                    reverse=True)
    top = scored[:keep]
    keep_idx = sorted(i for _, i in top)
    kept = [chunks[i] for i in keep_idx]
    kept_scores = sorted(top, key=lambda t: t[1])
    return kept, kept_scores


def compress(query, chunks, keep):
    """One-shot compression: embed the query AND all chunks, then rank.

    Used by the CLI where there is no persistent cache. The app uses
    rank_chunks() against cached vectors instead.
    """
    vecs = embed([query] + chunks)
    qv, cvs = vecs[0], vecs[1:]
    scored = sorted(((cosine(qv, cv), i) for i, cv in enumerate(cvs)), reverse=True)
    top = scored[:keep]
    keep_idx = sorted(i for _, i in top)
    return [chunks[i] for i in keep_idx], top


# --- [2][3][4] cascade ---------------------------------------------------------
def _context(chunks):
    return "\n\n".join(f"[chunk {n+1}]\n{c}" for n, c in enumerate(chunks))


def draft_answer(query, chunks):
    prompt = (f"Use ONLY the context to answer concisely.\n\n"
              f"Context:\n{_context(chunks)}\n\nQuestion: {query}\nAnswer:")
    return generate(prompt, DRAFTER, num_predict=DRAFT_MAX_TOKENS)


SCORE_RE = re.compile(r"SCORE:\s*(\d{1,3})", re.IGNORECASE)


def judge(query, draft):
    """One judge: verify the single most important step, then score.

    Not a snap verdict (too shallow -- misses reasoning errors) and not open-ended
    reasoning (too slow). A bounded check that targets the accuracy/speed sweet
    spot. For fact-extraction the check is trivial and fast; for reasoning it does
    just enough work to catch a bad answer.
    """
    prompt = ("Check whether the draft answer is correct. Briefly verify the "
              "single most important fact or step -- do not solve the whole "
              "problem from scratch. Then on the FINAL line output exactly "
              "'SCORE: <integer 0-100>' for how correct the draft is.\n\n"
              f"Question: {query}\n\nDraft answer: {draft}\n")
    text, ntok = generate(prompt, VERIFIER, num_predict=JUDGE_MAX_TOKENS)
    m = SCORE_RE.search(text)
    return (int(m.group(1)) if m else 0), ntok


def big_answer(query, chunks):
    prompt = (f"Use ONLY the context to answer.\n\n"
              f"Context:\n{_context(chunks)}\n\nQuestion: {query}\nAnswer:")
    return generate(prompt, VERIFIER)


def approx_tokens(s):
    return max(1, len(s) // 4)


def warmup(use_compress=True, use_cascade=True, log=print):
    """Load the models into RAM once so cold-start isn't counted in the timing."""
    if log:
        log("warming up models...")
    if use_compress:
        embed(["hi"])
    if use_cascade:
        generate("hi", DRAFTER, num_predict=1)
    generate("hi", VERIFIER, num_predict=1)
    if log:
        log("models warm.")


# --- answering -----------------------------------------------------------------
def _run_cascade(query, draft_fn, big_fn, threshold, use_cascade, on_stage=None):
    """Shared draft -> judge -> answer control flow.

    draft_fn / big_fn are zero-arg callables that return (text, tokens), so the
    same routing logic serves both document Q&A (context prompts) and open chat
    (conversational prompts) without duplicating the escalation logic.

    `on_stage(name, info)` is an optional callback fired as each REAL stage of
    the pipeline completes (drafting, judging, keep/escalate). It only reports
    work that is already happening, so it adds no model compute -- it's the hook
    the UI uses to show a live "thinking" ticker.

    Returns (final_answer, path, score, (draft_tok, judge_tok, gen_tok)).
    """
    def _emit(name, info=None):
        if on_stage:
            try:
                on_stage(name, info or {})
            except Exception:  # noqa: BLE001 -- reporting must never break answering
                pass

    draft_tok = judge_tok = gen_tok = 0
    score = None
    if use_cascade:
        _emit("drafting")
        draft, draft_tok = draft_fn()
        _emit("drafted", {"tokens": draft_tok})
        _emit("judging")
        score, judge_tok = judge(query, draft)
        _emit("judged", {"score": score, "threshold": threshold})
        if score >= threshold:
            path, final = "DRAFT-KEPT", draft
            _emit("kept", {"score": score})
        else:
            _emit("escalating", {"score": score})
            final, gen_tok = big_fn()
            path = "ESCALATED"
            _emit("escalated", {"tokens": gen_tok})
    else:
        _emit("generating")
        final, gen_tok = big_fn()
        path = "BIG-DIRECT"
    return final, path, score, (draft_tok, judge_tok, gen_tok)


def answer_from_chunks(query, kept, threshold=DEFAULT_THRESHOLD, use_cascade=True,
                       on_stage=None):
    """Document Q&A: run the cascade over an already-selected set of chunks.

    Split out from pipeline() so the app can reuse it after ranking against
    cached vectors, without re-embedding the document.
    """
    return _run_cascade(query,
                        lambda: draft_answer(query, kept),
                        lambda: big_answer(query, kept),
                        threshold, use_cascade, on_stage)


# --- page-cited document answering ---------------------------------------------
def _context_labeled(chunks, pages):
    """Context where each chunk is tagged with its source page, so the model can
    cite pages inline."""
    return "\n\n".join(f"[page {pg}]\n{c}" for c, pg in zip(chunks, pages))


_DOC_CITE = ("Answer the question using ONLY the context below. After each fact "
             "or claim, cite the page number it came from in parentheses, e.g. "
             "(p. 12). If a point draws on more than one page, cite them like "
             "(pp. 12, 15). Do not cite pages that aren't in the context.")


def draft_answer_doc(query, chunks, pages):
    prompt = (f"{_DOC_CITE} Be concise.\n\nContext:\n"
              f"{_context_labeled(chunks, pages)}\n\nQuestion: {query}\nAnswer:")
    return generate(prompt, DRAFTER, num_predict=DRAFT_MAX_TOKENS)


def big_answer_doc(query, chunks, pages):
    prompt = (f"{_DOC_CITE}\n\nContext:\n"
              f"{_context_labeled(chunks, pages)}\n\nQuestion: {query}\nAnswer:")
    return generate(prompt, VERIFIER)


def answer_from_chunks_doc(query, kept, pages, threshold=DEFAULT_THRESHOLD,
                           use_cascade=True, on_stage=None):
    """Like answer_from_chunks, but the context is page-labeled and the model is
    asked to cite pages inline. Used for multi-page documents."""
    return _run_cascade(query,
                        lambda: draft_answer_doc(query, kept, pages),
                        lambda: big_answer_doc(query, kept, pages),
                        threshold, use_cascade, on_stage)


def pipeline(query, doc, keep=DEFAULT_KEEP, threshold=DEFAULT_THRESHOLD,
             use_compress=True, use_cascade=True):
    """Full one-shot pipeline over a raw document. Used by the CLI/eval.

    The app does not call this; it ingests once (chunk + embed) then calls
    rank_chunks() + answer_from_chunks() per question to avoid re-embedding.
    """
    chunks = chunk_text(doc)
    t0 = time.time()

    if use_compress:
        kept, top = compress(query, chunks, keep)
        top_scores = [round(s, 3) for s, _ in top]
    else:
        kept, top_scores = chunks, None    # baseline: big model reads everything

    final, path, score, (draft_tok, judge_tok, gen_tok) = answer_from_chunks(
        query, kept, threshold, use_cascade)

    dt = time.time() - t0
    stats = {
        "orig_chunks": len(chunks), "kept_chunks": len(kept),
        "orig_tok": approx_tokens(doc), "kept_tok": approx_tokens(_context(kept)),
        "top_scores": top_scores, "path": path, "score": score,
        "tokens": (draft_tok, judge_tok, gen_tok), "time_s": dt,
    }
    return final, stats


# --- chat ----------------------------------------------------------------------
def _chat_prompt(query, concise):
    """Open-chat prompt. Unlike doc Q&A, there is NO 'use only the context'
    clause -- the model answers from its own knowledge. Today's date is supplied
    from the OS clock so date questions don't get refused.
    """
    style = " Answer directly and concisely." if concise else ""
    return (f"Today's date is {current_date_str()}. You are a helpful "
            f"assistant.{style} If you do not know something, say so.\n\n"
            f"Question: {query}\nAnswer:")


def chat_answer(query, threshold=DEFAULT_THRESHOLD, use_cascade=True, on_stage=None):
    """No-document chat: run the cascade with a conversational prompt.

    Short Q&A is decode-bound, so there is nothing to compress -- the value is
    the cheap judge deciding whether the drafter's answer is good enough. The
    prompt lets the model use its own knowledge (not just the question text),
    which is why plain chat no longer deflects with 'no context provided'.
    """
    return _run_cascade(
        query,
        lambda: generate(_chat_prompt(query, concise=True), DRAFTER,
                         num_predict=DRAFT_MAX_TOKENS),
        lambda: generate(_chat_prompt(query, concise=False), VERIFIER),
        threshold, use_cascade, on_stage)


# --- web lookup (opt-in; the ONLY thing that touches the internet) -------------
class WebUnavailable(RuntimeError):
    """Web lookup was requested but the search library/network isn't available."""


def web_search(query, k=6):
    """DuckDuckGo search (no API key). Returns [{title, url, snippet}].

    This is the sole networked path in FreeLAB and only runs when the UI's
    web-lookup toggle is on -- otherwise nothing ever leaves the machine.
    """
    try:
        from ddgs import DDGS                    # current package name
    except ImportError:
        try:
            from duckduckgo_search import DDGS   # older name, same API
        except ImportError as e:
            raise WebUnavailable(
                "Web lookup needs the search library:  pip install ddgs") from e
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=k))
    except Exception as e:  # noqa: BLE001 -- network/rate-limit/etc.
        raise WebUnavailable(f"Web search failed ({e}). Check your connection.") from e
    out = []
    for h in hits:
        out.append({"title": h.get("title", ""),
                    "url": h.get("href") or h.get("url", ""),
                    "snippet": h.get("body") or h.get("snippet", "")})
    return out


def web_answer(query, keep=DEFAULT_KEEP, threshold=DEFAULT_THRESHOLD,
               use_cascade=True, k=6):
    """RAG over web search results: the 'document' is the search snippets.

    Reuses the existing query-aware compression and the draft->judge->answer
    cascade -- web results are just another source. Returns
    (final, path, score, tokens, sources) where sources are the cited results.
    """
    results = web_search(query, k)
    if not results:
        return "No web results found.", "WEB-EMPTY", None, (0, 0, 0), []
    chunks = [f"{r['title']}\n{r['snippet']}" for r in results]
    kept, top = compress(query, chunks, keep)          # embed query+snippets, rank
    final, path, score, tokens = answer_from_chunks(
        query, kept, threshold=threshold, use_cascade=use_cascade)
    kept_idx = sorted(i for _, i in top)
    sources = [results[i] for i in kept_idx]
    return final, path, score, tokens, sources


DEMO_NEEDLE = ("Note for the record: the internal codename for the 2027 "
               "infrastructure initiative is BLUEHERON, approved by the committee.")


def build_demo():
    filler = ("The quarterly operations review covered staffing, budget variance, "
              "vendor timelines, and routine compliance items, no exceptions noted. ")
    paras = [filler * 3 for _ in range(120)]
    paras.insert(73, DEMO_NEEDLE)
    return "\n\n".join(paras), "What is the codename for the 2027 infrastructure initiative?"

#!/usr/bin/env python3
"""
FreeLAB engine -- UI-agnostic ingest + answer routing.

This is the logic that used to live inside `freelab_streamlit.py` (`ingest_files`,
`answer_query`, `stats_line`), lifted out so it depends on neither Streamlit nor
FastAPI. Both the old Streamlit UI and the new v2 API call the same functions here,
so the routing (document > web > plain chat), the embed-once index, and the stats
line have exactly one source of truth.

All heavy lifting still comes from `freelab_core.py`.
"""

from __future__ import annotations

import os

import freelab_core as fl

EMBED_BATCH = 64        # embed chunks in batches (lets the caller show progress)

# Below this many input tokens (query + retrieved context), skip the draft/judge
# cascade and answer directly with the big model. On short inputs the cascade is
# pure overhead (the drafter usually can't help and the big model is already fast),
# so this gives fast, direct 8B answers on small queries and the full cascade only
# when there is enough context for compression + drafting to pay off. 0 disables.
DIRECT_BELOW_TOKENS = 512

OLLAMA_DOWN = ("⚠️ Could not reach Ollama at localhost:11434. "
               "Is `ollama serve` running?")


# --- ingest --------------------------------------------------------------------
def ingest_files(paths, names=None, ocr_progress=None, embed_progress=None):
    """Embed one or more files into a single combined index (embed ONCE).

    `paths`  : list of file paths on disk.
    `names`  : optional display names (defaults to basenames).
    Callbacks are optional: ocr_progress(done,total,name), embed_progress(frac).
    Returns the index dict, or raises ValueError if nothing was extractable.
    """
    names = names or [os.path.basename(p) for p in paths]
    used_names, all_pages = [], []
    for path, name in zip(paths, names):
        cb = (lambda d, t, _n=name: ocr_progress(d, t, _n)) if ocr_progress else None
        pages = fl.extract_pages(path, ocr_progress=cb)
        if any(p.strip() for p in pages):
            all_pages.extend(pages)
            used_names.append(name)
    if not all_pages:
        raise ValueError("No text found in the attached file(s), even after OCR.")

    combined = "\n".join(all_pages)
    chunks = fl.chunk_text(combined)
    pageno = fl.chunk_pages(all_pages)
    vecs, n = [], len(chunks)
    for start in range(0, n, EMBED_BATCH):
        vecs.extend(fl.embed(chunks[start:start + EMBED_BATCH]))
        if embed_progress:
            embed_progress(min((start + EMBED_BATCH) / max(n, 1), 1.0))
    name = used_names[0] if len(used_names) == 1 else f"{len(used_names)} files"
    pdf_bytes = None
    if len(paths) == 1 and paths[0].lower().endswith(".pdf"):
        with open(paths[0], "rb") as f:
            pdf_bytes = f.read()
    return {"name": name, "chunks": chunks, "vecs": vecs, "pageno": pageno,
            "pages": len(all_pages), "orig_tok": fl.approx_tokens(combined),
            "pdf_bytes": pdf_bytes}


# --- stats ---------------------------------------------------------------------
def stats_line(index, kept, path, score, tokens):
    d, j, g = tokens
    kept_tok = fl.approx_tokens(fl._context(kept))
    orig_tok = index["orig_tok"]
    pct = 100 - 100 * kept_tok // max(orig_tok, 1)
    verdict = f" (judge score {score})" if score is not None else ""
    return (f"compression: {len(index['chunks'])} → {len(kept)} chunks, "
            f"~{orig_tok} → ~{kept_tok} tokens ({pct}% smaller) · "
            f"path: **{path}** {verdict} · tokens: draft={d} judge={j} biggen={g}")


# --- answer routing (document > web > plain chat) ------------------------------
def answer_query(text, conv, keep=fl.DEFAULT_KEEP, threshold=fl.DEFAULT_THRESHOLD,
                 use_cascade=True, web_lookup=False, on_stage=None,
                 direct_below=DIRECT_BELOW_TOKENS):
    """Route one question and return the answer string.

    Mutates `conv` in place with `last_sources`, `last_query`, `pdf_bytes`
    (the vault layer persists these), exactly like the old Streamlit path.

    `on_stage(name, info)` (optional) fires as each real pipeline stage runs, so
    the UI can show a live ticker. It reports work that already happens -- no
    extra model compute.
    """
    def _emit(name, info=None):
        if on_stage:
            try:
                on_stage(name, info or {})
            except Exception:  # noqa: BLE001
                pass

    index = conv.get("index")
    if index:                                   # a file is attached -> answer it
        _emit("retrieving", {"total": len(index["chunks"])})
        kept, scores = fl.rank_chunks(text, index["chunks"], index["vecs"], keep)
        _emit("retrieved", {"total": len(index["chunks"]), "kept": len(kept)})
        pageno = index.get("pageno") or []
        kept_pages = [pageno[i] if i < len(pageno) else None for sc, i in scores]
        multipage = index.get("pages", 1) > 1 and any(kept_pages)
        in_tok = fl.approx_tokens(fl._context(kept)) + fl.approx_tokens(text)
        casc = use_cascade
        if casc and direct_below and in_tok < direct_below:
            casc = False                       # short input -> skip cascade, answer directly
            _emit("direct", {"tokens": in_tok, "threshold": direct_below})
        if multipage:
            final, path, score, tokens = fl.answer_from_chunks_doc(
                text, kept, kept_pages, threshold=threshold,
                use_cascade=casc, on_stage=on_stage)
        else:
            final, path, score, tokens = fl.answer_from_chunks(
                text, kept, threshold=threshold, use_cascade=casc,
                on_stage=on_stage)
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
        _emit("searching")
        final, path, score, _, sources = fl.web_answer(
            text, keep=keep, threshold=threshold, use_cascade=use_cascade)
        if sources:
            conv["last_sources"] = [
                {"chunk": i + 1, "score": None, "page": None,
                 "url": s.get("url"), "title": s.get("title"),
                 "text": s.get("body") or s.get("title") or s.get("url")}
                for i, s in enumerate(sources)]
            links = "\n".join(f"- [{s['title'] or s['url']}]({s['url']})"
                              for s in sources if s.get("url"))
            final += f"\n\n**Sources**\n{links}"
        return final
    casc = use_cascade
    if casc and direct_below and fl.approx_tokens(text) < direct_below:
        casc = False                           # short chat -> direct big model
        _emit("direct", {"tokens": fl.approx_tokens(text), "threshold": direct_below})
    final, path, score, _ = fl.chat_answer(
        text, threshold=threshold, use_cascade=casc, on_stage=on_stage)
    return final

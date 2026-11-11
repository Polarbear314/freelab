#!/usr/bin/env python3
"""
FreeLAB harness -- a local Gradio app on top of the FreeLAB pipeline.

Three tabs, all running against your local Ollama (no cloud, no API keys):

  1. Chat        -- ask the cascade general questions (no document).
  2. Document QA -- upload a PDF / TXT / MD / DOCX; it chunks + embeds the file
                    ONCE at upload, then answers questions fast against the
                    cached vectors (no re-embedding per question).
  3. Batch       -- upload a file of many questions (+ an optional document),
                    answer them all, and download the results as CSV or Markdown.

The one big engineering lever (see the build outline): a document is embedded
exactly once at ingest and its vectors are reused for every later question. That
is what makes repeated Q&A and batch mode fast.

RUN
  pip install gradio pypdf python-docx
  ollama pull qwen3:0.6b && ollama pull qwen3:8b && ollama pull nomic-embed-text
  python3 freelab_app.py
  # then open the printed http://127.0.0.1:7860 URL
"""

import csv
import os
import tempfile
import urllib.error

import gradio as gr

import freelab_core as fl

EMBED_BATCH = 64        # embed chunks in batches so large docs show progress
SOURCE_PREVIEW = 400    # chars of each source chunk to show in the panel


# --- ingest (embed once) -------------------------------------------------------
def ingest(path, progress=gr.Progress()):
    """Turn an uploaded file into a cached index: {name, chunks, vecs}.

    Text (and OCR for scanned PDFs) comes from the shared core. Embeds every
    chunk ONCE here; later questions only embed the question and rank it against
    these cached vectors.
    """
    name = os.path.basename(path)
    progress(0, desc=f"Reading {name}...")

    def ocr_progress(done, total):
        progress(done / total, desc=f"OCR page {done}/{total}")

    try:
        text = fl.extract_text(path, ocr_progress=ocr_progress)
    except fl.OcrUnavailable as e:
        raise gr.Error(str(e))
    if not text.strip():
        raise gr.Error("No text found in this document, even after OCR.")
    chunks = fl.chunk_text(text)
    vecs = []
    for start in progress.tqdm(range(0, len(chunks), EMBED_BATCH),
                               desc="Embedding chunks"):
        batch = chunks[start:start + EMBED_BATCH]
        vecs.extend(fl.embed(batch))
    return {"name": name, "chunks": chunks, "vecs": vecs,
            "orig_tok": fl.approx_tokens(text)}


def _sources_md(kept, scores):
    lines = [f"**{len(kept)} chunk(s) used** (ranked by relevance to the question):\n"]
    for (score, idx), chunk in zip(scores, kept):
        preview = chunk[:SOURCE_PREVIEW].replace("\n", " ")
        ellipsis = "..." if len(chunk) > SOURCE_PREVIEW else ""
        lines.append(f"- **chunk {idx + 1}** (score {score:.3f}): {preview}{ellipsis}")
    return "\n".join(lines)


def _stats_md(index, kept, path, score, tokens):
    d, j, g = tokens
    kept_tok = fl.approx_tokens(fl._context(kept))
    orig_tok = index["orig_tok"]
    pct = 100 - 100 * kept_tok // max(orig_tok, 1)
    verdict = f"  (judge score {score})" if score is not None else ""
    return (f"compression: {len(index['chunks'])} -> {len(kept)} chunks, "
            f"~{orig_tok} -> ~{kept_tok} tokens ({pct}% smaller)  \n"
            f"path: **{path}**{verdict}  \n"
            f"tokens: draft={d} judge={j} biggen={g}")


# --- Chat tab ------------------------------------------------------------------
def chat_respond(message, history, keep, threshold, use_cascade, web_lookup):
    history = history or []
    history.append({"role": "user", "content": message})
    sources = []
    try:
        if web_lookup:
            final, path, score, _, sources = fl.web_answer(
                message, keep=int(keep), threshold=int(threshold),
                use_cascade=use_cascade)
        else:
            final, path, score, _ = fl.chat_answer(
                message, threshold=int(threshold), use_cascade=use_cascade)
    except fl.WebUnavailable as e:
        final = f"⚠️ {e}"
    except urllib.error.URLError:
        final = ("Could not reach Ollama at localhost:11434. Is `ollama serve` "
                 "running and are the models pulled?")
    if sources:
        links = "\n".join(f"- [{s['title'] or s['url']}]({s['url']})"
                          for s in sources if s.get("url"))
        final = f"{final}\n\n**Sources**\n{links}"
    history.append({"role": "assistant", "content": final})
    return history, ""


# --- Document tab --------------------------------------------------------------
def doc_ingest(file, progress=gr.Progress()):
    if not file:
        return None, "Upload a file to index it."
    try:
        index = ingest(file, progress)
    except urllib.error.URLError:
        return None, "Could not reach Ollama at localhost:11434."
    msg = (f"Indexed **{index['name']}** — {len(index['chunks'])} chunks, "
           f"~{index['orig_tok']} tokens embedded once. Ask away.")
    return index, msg


def doc_ask(question, index, keep, threshold, use_cascade):
    if not index:
        return "Index a document first.", ""
    if not question.strip():
        return "Ask a question.", ""
    try:
        kept, scores = fl.rank_chunks(question, index["chunks"], index["vecs"],
                                      int(keep))
        final, path, score, tokens = fl.answer_from_chunks(
            question, kept, threshold=int(threshold), use_cascade=use_cascade)
    except urllib.error.URLError:
        return "Could not reach Ollama at localhost:11434.", ""
    stats = _stats_md(index, kept, path, score, tokens)
    sources = _sources_md(kept, scores)
    return f"{final}\n\n---\n{stats}", sources


# --- Batch tab -----------------------------------------------------------------
def batch_run(questions_file, doc_file, keep, threshold, use_cascade,
              progress=gr.Progress()):
    if not questions_file:
        return [], None, "Upload a questions file (.txt or .csv) first."
    questions = fl.read_questions(questions_file)
    if not questions:
        return [], None, "No questions found in that file."

    index = None
    if doc_file:
        try:
            index = ingest(doc_file, progress)
        except urllib.error.URLError:
            return [], None, "Could not reach Ollama at localhost:11434."

    rows = []
    for q in progress.tqdm(questions, desc="Answering"):
        try:
            if index:
                kept, _ = fl.rank_chunks(q, index["chunks"], index["vecs"],
                                         int(keep))
            else:
                kept = [q]
            final, path, score, tokens = fl.answer_from_chunks(
                q, kept, threshold=int(threshold), use_cascade=use_cascade)
        except urllib.error.URLError:
            return rows, None, "Could not reach Ollama at localhost:11434."
        rows.append([q, final, path, score if score is not None else "",
                     len(kept)])

    out_path = os.path.join(tempfile.gettempdir(), "freelab_batch_results.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer", "path", "judge_score", "chunks_used"])
        w.writerows(rows)
    return rows, out_path, f"Answered {len(rows)} question(s)."


# --- settings ------------------------------------------------------------------
def apply_models(drafter, verifier, embed_model):
    """Push model-name settings into the core module (read at call time)."""
    fl.DRAFTER = drafter.strip() or fl.DRAFTER
    fl.VERIFIER = verifier.strip() or fl.VERIFIER
    fl.EMBED_MODEL = embed_model.strip() or fl.EMBED_MODEL
    return f"Models set — drafter: {fl.DRAFTER}, verifier: {fl.VERIFIER}, embed: {fl.EMBED_MODEL}"


def do_warmup():
    try:
        fl.warmup(log=None)
        return "🟢 Models warm"
    except urllib.error.URLError:
        return "🔴 Ollama not reachable at localhost:11434"
    except Exception as e:  # noqa: BLE001
        return f"🔴 Warmup failed: {e}"


# --- UI ------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="FreeLAB") as demo:
        gr.Markdown("# FreeLAB — fast, free, local AI\n"
                    "Small model drafts, big model verifies. All local via Ollama.")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tab("Chat"):
                    chatbot = gr.Chatbot(height=420, label="Cascade chat")
                    chat_in = gr.Textbox(placeholder="Ask anything...",
                                         show_label=False)

                with gr.Tab("Document Q&A"):
                    index_state = gr.State()
                    doc_file = gr.File(label="Upload PDF / TXT / MD / DOCX",
                                       file_types=[".pdf", ".txt", ".md", ".docx"],
                                       type="filepath")
                    doc_status = gr.Markdown("No document indexed yet.")
                    doc_q = gr.Textbox(label="Question",
                                       placeholder="Ask the document...")
                    doc_btn = gr.Button("Answer", variant="primary")
                    doc_answer = gr.Markdown(label="Answer")
                    with gr.Accordion("Sources (chunks used)", open=False):
                        doc_sources = gr.Markdown()

                with gr.Tab("Batch"):
                    q_file = gr.File(label="Questions (.txt one per line, or .csv)",
                                     file_types=[".txt", ".csv"], type="filepath")
                    batch_doc = gr.File(
                        label="Optional document to answer against",
                        file_types=[".pdf", ".txt", ".md", ".docx"],
                        type="filepath")
                    batch_btn = gr.Button("Run batch", variant="primary")
                    batch_status = gr.Markdown()
                    batch_table = gr.Dataframe(
                        headers=["question", "answer", "path", "judge_score",
                                 "chunks_used"],
                        wrap=True, label="Results")
                    batch_download = gr.File(label="Download results (CSV)")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                warm_status = gr.Markdown("⚪ warming up...")
                keep = gr.Slider(1, 20, value=fl.DEFAULT_KEEP, step=1,
                                 label="keep (chunks retained)")
                threshold = gr.Slider(0, 100, value=fl.DEFAULT_THRESHOLD, step=5,
                                      label="threshold (judge score to keep draft)")
                use_cascade = gr.Checkbox(value=True,
                                          label="use cascade (draft → judge)")
                web_lookup = gr.Checkbox(
                    value=False, label="🌐 Web lookup (uses internet)",
                    info="Off = fully local. On = Chat searches the web + cites sources.")
                with gr.Accordion("Models", open=False):
                    m_draft = gr.Textbox(fl.DRAFTER, label="drafter")
                    m_verify = gr.Textbox(fl.VERIFIER, label="verifier / judge")
                    m_embed = gr.Textbox(fl.EMBED_MODEL, label="embedding model")
                    apply_btn = gr.Button("Apply models")
                    apply_out = gr.Markdown()
                    apply_btn.click(apply_models, [m_draft, m_verify, m_embed],
                                    apply_out)

        # wire tabs to the shared settings controls
        doc_btn.click(doc_ask, [doc_q, index_state, keep, threshold, use_cascade],
                      [doc_answer, doc_sources])
        doc_q.submit(doc_ask, [doc_q, index_state, keep, threshold, use_cascade],
                     [doc_answer, doc_sources])
        doc_file.change(doc_ingest, doc_file, [index_state, doc_status])
        batch_btn.click(batch_run,
                        [q_file, batch_doc, keep, threshold, use_cascade],
                        [batch_table, batch_download, batch_status])

        chat_in.submit(chat_respond,
                       [chat_in, chatbot, keep, threshold, use_cascade, web_lookup],
                       [chatbot, chat_in])

        demo.load(do_warmup, None, warm_status)
    return demo


if __name__ == "__main__":
    build_ui().launch(theme=gr.themes.Soft())

# FreeLAB Harness — Build Outline

A local app on top of the existing FreeLAB pipeline. Three things it should do:

1. **Ask** — chat with the AI (general questions via the cascade).
2. **Document Q&A** — attach a PDF / TXT / MD / DOCX; it chunks + indexes the file once, then answers questions against it fast.
3. **Batch** — attach a file of many questions (and optionally a document); it answers them all and exports the results.

Everything stays free and local: your existing Ollama models, no cloud, no API keys.

---

## 1. Build on what already exists — don't rewrite the pipeline

The core logic (compress → draft → judge → answer) is already written and tested in `freelab.py`. First step of the harness is a small refactor, not new logic:

- Extract the shared functions into a module, `freelab_core.py`: `ollama_generate` / `_post`, `embed`, `chunk_text`, `cosine`, `compress`, `draft_answer`, `judge`, `big_answer`, `pipeline`, `warmup`.
- Then `freelab.py` (CLI), `freelab_eval.py`, and the new app all `import freelab_core` — one source of truth.

## 2. Recommended stack (all free / local)

- **UI: Gradio** (recommended) — has a built-in chatbot component and file upload, runs in the browser locally, ~one Python file. *Alternatives:* Streamlit (similar ease) or FastAPI + HTML (more control, more work).
- **PDF text:** `pypdf` (simple) or `pdfplumber` (better with tables).
- **DOCX text:** `python-docx`.
- **Everything else:** your current Ollama + `nomic-embed-text` + qwen3 models.

## 3. Architecture

```
file → extract text → chunk_text → EMBED ONCE → cache vectors (per document)
                                                     │
question → embed question → rank vs cached vectors → top-k chunks
                                                     │
                                          pipeline: draft → judge → answer
```

- **Ingest** turns any uploaded file into (chunks, vectors) and caches it.
- **Query** embeds only the question, ranks against the cached vectors, runs the pipeline on the top-k chunks.
- **Batch** loops the query handler over a list of questions and exports the answers.

## 4. Key engineering decisions (call these out early)

- **Embedding cache is the #1 speed lever.** The current scripts re-embed the whole document on every question. The harness must embed a document **once** at upload and reuse those vectors for all later questions. This is what makes repeated Q&A and batch mode fast.
- **Cache structure:** `{doc_id: (chunks, vectors)}` in memory; optionally pickle to disk so re-opening a document is instant (stretch).
- **Warm the models on startup** so the first question isn't slow.
- **Show which chunks were used** ("sources" panel) — cheap to add, builds trust, and shows the compression working.
- **Settings:** expose model names, `keep` (chunks retained), and `threshold` (judge escalation) in a sidebar.
- **Large files:** embed chunks in batches with a progress indicator; set a sane max size for v1.
- **Scope out for v1:** OCR for scanned/image PDFs (add later); multi-turn memory (add later).

## 5. Feature spec by mode

- **Chat:** message box + history; each turn runs the cascade (no document). Optional: multi-turn memory later.
- **Document Q&A:** upload → "indexing…" status → once indexed, ask questions; show the answer plus the source chunks and the compression % / path (draft-kept vs escalated).
- **Batch:** upload a `.txt` (one question per line) or `.csv`; optionally attach a document to answer against; run all; show a results table and a **Download** button (CSV or Markdown) with columns: question, answer, path, chunks used.

## 6. UI sketch (Gradio tabs)

- **Tab 1 — Chat:** history + input.
- **Tab 2 — Document:** file upload, indexing status, question box, answer + sources.
- **Tab 3 — Batch:** questions-file upload, optional document, Run button, results table, Download.
- **Sidebar:** model / `keep` / `threshold` settings; a "warm models" indicator.

## 7. Build order (milestones)

1. `freelab_core.py` — extract shared functions from `freelab.py`.
2. Minimal Gradio **Chat** tab wired to the cascade.
3. **Document** ingest + embedding cache + Document Q&A tab.
4. **Batch** tab + export.
5. Polish: sources panel, settings sidebar, startup warmup, progress bars.
6. Stretch: streaming responses, disk-persisted indexes, multi-document library, the two-tier judge, multi-turn chat memory.

## 8. Kickoff prompt for the new chat

Paste this to start the build with full context:

> I'm building a local app ("harness") on top of my existing FreeLAB pipeline, in the folder `Projects/FreeLAB`. FreeLAB is a local AI pipeline (compress → draft → judge → answer) using Ollama with qwen3:0.6b (drafter), qwen3:8b (judge/verifier), and nomic-embed-text (embeddings). The working code is `freelab.py` (the pipeline), plus `freelab_compress.py`, `freelab_cascade.py`, `freelab_eval.py`, and `README.md`.
>
> I want a Gradio app with three tabs: (1) Chat — ask questions via the cascade; (2) Document Q&A — upload a PDF/TXT/MD/DOCX, chunk and embed it ONCE, then answer questions against it fast; (3) Batch — upload a file of many questions (+ optional document), answer all, export results as CSV.
>
> Start by refactoring the shared functions out of `freelab.py` into `freelab_core.py` so the app imports them. Key requirement: embed each document only once at upload and cache the vectors — don't re-embed per question. Then build the tabs in the order above. Everything stays local (Ollama) and free. Please follow the build order in `FreeLAB Harness — Build Outline.md` and explain each piece as we go.

## Related

- `freelab.py`, `freelab_eval.py`, `README.md` (in this folder)
- `[[FreeLAB Build Log]]`, `[[FreeLAB Test Results]]` (in the vault)

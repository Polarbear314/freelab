# FreeLAB

**Fast, free, local AI on a Mac.** A small model does the cheap work; a big model
does the expensive work only when it's actually needed. The goal is big-model
quality at closer-to-small-model speed, using only free, open-weight models
running locally through [Ollama](https://ollama.com) — no API keys, no cloud.

Built and tested on a 16 GB Apple Silicon Mac by Henry Risch (Bowdoin College), Summer 2026.

---

## The idea

Cloud AI feels fast because it runs on huge hardware. Locally, on 16 GB, you're
memory- and bandwidth-bound, so you can't just run a big model on everything.
FreeLAB gets closer to cloud-like responsiveness by **splitting the work by
difficulty** and **shrinking the input before the big model sees it**:

```
big document
    │
    ▼
[1] COMPRESS   a small embedding model keeps only the chunks relevant to the query
    │          → attacks PREFILL, the real cost/OOM driver for large text
    ▼
[2] DRAFT      a small LLM answers from the kept chunks
    │
    ▼
[3] JUDGE      the big LLM grades the draft with a tiny verdict (a few tokens)
    │            score high → keep the draft        (fast path)
    │            score low  → escalate
    ▼
[4] ANSWER     the big LLM writes the full answer   (only on the hard minority)
    │
    ▼
answer
```

The unifying principle, borrowed from speculative decoding and applied at the
**answer** level: *verifying is much cheaper than generating.* The big model
usually only has to judge (≈10 tokens), not write (hundreds).

---

## Why it's built this way (things learned the hard way)

- **Token-level speculative decoding didn't help on 16 GB.** Pairing an 8B
  verifier with a 0.6B drafter in llama.cpp gave ~1.0–1.05× — memory-bandwidth
  contention cancels the gains, and larger draft windows just OOM. So FreeLAB
  works at the *answer* level, not the token level.
- **A small model can't judge its own confidence.** Asked to self-rate, qwen3:0.6b
  returned "100" on answers that were flat wrong. So the *big* model is the judge
  — it's far better calibrated, and judging is cheap.
- **Compression only pays off on large input.** For short Q&A, cost is dominated
  by decode, so there's nothing to compress. For big documents, prefill dominates
  — and on 16 GB, trimming the input is often what makes the doc *fit* at all.
- **Compression must be query-aware.** Blindly summarizing risks dropping the one
  paragraph with the answer. Ranking chunks by relevance to the *question* keeps
  the right material.

---

## Setup

```bash
# one-time: install models
ollama pull qwen3:0.6b        # drafter (small, fast)
ollama pull qwen3:8b          # verifier / judge (big)
ollama pull nomic-embed-text  # embeddings for compression (~275 MB)

# make sure the Ollama server is running (menu-bar app, or: ollama serve)
```

## Run

```bash
# needle-in-a-haystack demo: one fact buried in a ~16k-token document
python3 freelab.py --demo

# ask a real document a question
python3 freelab.py --file sample_doc.md "How many planets are in the solar system?"

# show the baseline: no compression, big model reads the whole document
python3 freelab.py --file big.txt --no-compress "..."

# compression only, no cascade
python3 freelab.py --file big.txt --no-cascade "..."
```

Useful flags: `--keep N` (chunks to retain), `--threshold N` (judge score needed
to keep the draft, 0–100).

---

## Benchmarks

Full write-up with every table and caveat: [`results/BENCHMARKS.md`](results/BENCHMARKS.md). Raw runs are in `results/`.

**Document Q&A (where FreeLAB is built to win).** Three facts buried in a ~28k-token document: a plain `qwen3:8b` reading the whole document got **0/3**; FreeLAB got **3/3**, was **1.6x faster**, and sent **96% less** input to the big model. At ~70k tokens the result held (0/3 vs 3/3), because FreeLAB's input stays around 1k tokens regardless of document size. The plain model fails because Ollama silently truncates input to the context window, so the buried facts never reach it.

**Code generation (HumanEval + MBPP, 40 tasks each, 3 seeds, unit tests as the judge, bootstrap 95% CIs).** FreeLAB **matches the 8B model's pass@1** (0.925 on HumanEval; 0.567 vs 0.542 on MBPP, overlapping CIs), while the 0.6B drafter, about 13x smaller, solves **18-32% of tasks on its own** and the 8B never runs. The gain is cost, not a higher ceiling.

**Where it doesn't help (reported on purpose).** On short-context QA (SQuAD v2, 2x2 ablation) all four configurations tie at ~0.54 F1: there is nothing to compress, and the cascade only adds latency. On short trick-question chat it is about 10% slower. FreeLAB's mechanisms act on context length, so they help on big documents and codebases and do nothing on short inputs.

**A correction I made along the way.** A 5-task pilot showed FreeLAB *beating* the 8B at code (4/5 vs 3/5). At n=120 that edge disappeared into parity, so the honest claim is "same accuracy at lower cost," not "more accurate."

---

## Early spot checks

| Test | Result |
|---|---|
| Needle in ~16k-token doc (`--demo`) | Correct answer, input shrunk **94%** (83→5 chunks); needle scored 0.68 vs 0.43 filler |
| Topical selection (`sample_doc.md`, planets Q) | Picked the right section (0.77 vs 0.64), dropped 3 of 5 chunks, answered "eight" |
| Judge cost | ~**6–7 tokens** per judgment vs 100+ to generate — the cheapness that makes the cascade worth it |
| Full pipeline (`--demo`) | Input shrunk **92%**; answer came from the drafter (18 tok) with the big model only judging (7 tok, `biggen=0`) — correct answer, big model never generated |
| Chunk test (`freelab_eval.py --chunk`) | **3/3** buried facts found in a **~42,000-token** document, compressed **98%** each — chunking holds up at scale |
| Judge — early snap version | Pipeline 5/7, **2 leaked**, fast. A too-shallow verdict misses reasoning errors. |
| Judge — over-deep version | Pipeline 6/7, **0 leaked**, but ~**15 s/question**. Full reasoning catches everything but is slow. |
| **Judge — shipped (bounded verify, `--judge`)** | Pipeline 6/7, **0 leaked**, caught **3/3**, **~5.7 s/question**. One judge, no mode switch: verifies the key step only — deep-judge accuracy at ~1/3 the time. |

The single miss across all versions (counting r's in "strawberry") is a model-capability ceiling, not a routing failure — the 8B itself gets it wrong, so no cascade over it can be right.

---

## The harness (local app)

A [Gradio](https://www.gradio.app) app, `freelab_app.py`, wraps the pipeline in a
browser UI with three tabs — still fully local, no API keys:

- **Chat** — ask the cascade general questions (no document).
- **Document Q&A** — upload a PDF / TXT / MD / DOCX; it chunks and embeds the
  file **once** at upload, caches the vectors, then answers questions fast
  against that cache (only the question is re-embedded per query). Shows the
  answer plus the source chunks and the compression % / path.
- **Batch** — upload a file of many questions (`.txt`, one per line, or `.csv`,
  first column) and an optional document; it answers them all and offers the
  results as a downloadable CSV.

The sidebar exposes `keep`, `threshold`, cascade on/off, and the model names, plus
a "Warm models" button so the first real question isn't slow.

**Embed-once is the key speed lever.** The CLI re-embeds the whole document on
every question; the app embeds each document a single time at ingest and reuses
those vectors for all later questions — this is what makes repeated Q&A and batch
mode fast.

Because all the logic lives in `freelab_core.py`, the UI is swappable. There are
two front-ends — pick whichever you prefer:

```bash
# Streamlit (default)
pip install streamlit pypdf python-docx
streamlit run freelab_streamlit.py

# Gradio (alternative)
pip install gradio pypdf python-docx
python3 freelab_app.py            # opens http://127.0.0.1:7860
```

On a Homebrew Python you may need a virtualenv first: `python3 -m venv .venv &&
source .venv/bin/activate`, then the `pip install` above.

**Scanned PDFs (OCR).** A PDF that is just images of text has no text layer to
extract, so the harness OCRs it automatically at ingest. That needs the Tesseract
engine plus a couple of Python packages (once):

```bash
brew install tesseract                        # the OCR engine (macOS)
pip install pymupdf pytesseract pillow        # render pages + OCR them
```

Without these, a scanned PDF gives a clear message telling you what to install;
everything else keeps working. OCR runs once per document (it's cached like any
other), but it's slow on large scans — expect a wait at ingest.

**Web lookup (opt-in).** A sidebar toggle, **off by default**, lets Chat reach the
internet when you want it to. Off, nothing ever leaves your machine — it's fully
local. On, Chat runs one DuckDuckGo search (no API key), ranks the result snippets
through the same query-aware compression, grounds the answer in them, and cites the
source links. It's the same RAG pipeline as the Document tab, with the web as the
source instead of a file.

```bash
pip install ddgs        # DuckDuckGo search, no key required
```

If the library isn't installed or the search fails, Chat says so and the local
side keeps working.

## Files

- `freelab_core.py` — the shared pipeline (embed, compress, draft/judge/answer); one source of truth
- `freelab.py` — the command-line front-end (imports the core)
- `freelab_streamlit.py` — the Streamlit harness: Chat / Document Q&A / Batch (imports the core)
- `freelab_app.py` — the Gradio harness, same three tabs (imports the core)
- `freelab_eval.py` — judge + chunk eval harness (imports the core)
- `freelab_compress.py` — the compression front-end on its own
- `freelab_cascade.py` — the judge-gated cascade on its own
- `freelab_bench.py`, `freelab_code_bench.py`, `freelab_eval_suite.py`, `freelab_threshold_sweep.py` — benchmarks and the rigorous eval suite
- `freelab_api.py`, `freelab_engine.py`, `web/` — a FastAPI backend and web front-end
- `freelab_vault.py` — stores chats and sources as an Obsidian-style markdown vault
- `sample_doc.md` — a small multi-topic document for testing
- `docs/` — build outline and v2 redesign plan

---

## Honest limitations / next steps

- The big model still runs a judge pass on **every** query. A cheap pre-filter
  (drafter logprobs) could let obviously-easy questions skip the big model
  entirely — a two-tier gate.
- Compression is lossy; a bad chunk ranking can starve the answer. Worth adding a
  fallback that widens `--keep` when the judge score is low.
- The benchmark scripts re-embed the document on every query, which overstates
  FreeLAB's latency; the app embeds once at ingest. Separating ingest time from
  query time would give a fairer speed number.
- Chunking is word-based, not code-aware (by function or class), and the
  retrieve-then-generate loop for code is validated in its two halves, not yet
  end to end.

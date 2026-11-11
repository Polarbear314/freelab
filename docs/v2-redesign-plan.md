# FreeLAB v2 — Obsidian-Style Redesign (Handoff & Plan)

**Purpose.** This is a handoff document for a fresh session. The goal of v2 is to
keep everything FreeLAB already does (local AI chat, document Q&A with page-cited
sources, batch, web lookup, the memory graph) but re-skin and re-organize it so it
**looks and feels like Obsidian** — a calm, keyboard-driven, panel-based knowledge
workspace — with a **calendar**, **to-dos/tasks**, and an enhanced **wiki graph
view** as first-class features.

Read this top to bottom, then start with the "Kickoff prompt" at the end.

---

## 0. Decisions locked (2026-07-26)

The open questions in §7 that "change everything downstream" are now answered:

- **UI architecture → Option B** (FastAPI backend + a web SPA that mimics Obsidian).
  `freelab_core.py` stays untouched and gets wrapped in a thin local HTTP API; the
  UI is rebuilt as a real Obsidian-like workspace (left rail, tabbed panes, ⌘P
  command palette, right context sidebar, real graph). *Why B over A or C:* the
  stated goal is fidelity to Obsidian's sleekness/usability, and §3 warns Streamlit
  (A) structurally can't do panes, command palette, drag/drop, or smooth theming —
  it "still feels like Streamlit." Option C (Obsidian plugin) is the truest feel but
  throws away the tested Python engine for a new TypeScript codebase. B keeps the
  engine, delivers the four requested modules, and is the only path §3 says can get
  "as close as possible to Obsidian." Browser SPA on localhost first; Tauri desktop
  packaging deferred to a polish phase.
- **Data model → own Markdown vault.** Persistence moves from `state.pkl` to a
  `FreeLAB-Vault/` folder (own folder, clean separation from the Second Brain). Built
  and shipped this session: `freelab_vault.py` + a migration. See §4 for the layout.
- **v2.1 scope (must-haves) → all four:** Obsidian-style shell + theme, enhanced
  graph view, calendar + daily notes, tasks/to-dos.

**Done this session (Phase 1 of the roadmap — the "do first, UI-agnostic" work):**
`freelab_vault.py` implements the vault layout below with lossless Markdown
round-trip (invisible HTML-comment sentinels keep the transcript re-parseable while
rendering cleanly in Obsidian), a flat-YAML frontmatter reader/writer (no new deps),
a `- [ ]` task parser (Obsidian Tasks syntax), daily-note helpers, a hidden
`.freelab/` cache for vectors + raw PDF bytes, and a `migrate`/`doctor` CLI. The real
`state.pkl` was migrated (3 chats, message counts + current pointer verified against
the pickle; synthetic round-trip covers tricky Markdown, wikilinks, vectors, and PDF
bytes). Pickle left untouched as a backup.

**Next up (Phase 2+):** wrap `freelab_core.py` in FastAPI (chat, ingest, ask,
web, graph endpoints reading/writing the vault) → build the SPA shell + theme →
graph v2 → calendar → tasks. See §6.

**Done since (Phases 2–3 + polish):** FastAPI backend (`freelab_api.py` +
`freelab_engine.py`) serving the vault; single-file Obsidian-style UI
(`web/index.html`) with ribbon, chat explorer, tabbed workspace, right context
sidebar, ⌘P palette, light/dark theme, and all four modules (chat w/ page-cited
sources + full-document page-image reader, graph, calendar, tasks); model warm-up
restored; a viewport-locked grid so panels scroll independently; and a **live
pipeline ticker** — the real compress → draft → judge → keep/escalate stages
streamed over SSE (`/ask_stream`, core `on_stage` callback), which adds no model
compute since it only reports stages that already run.

---

## 1. The vision in one paragraph

FreeLAB becomes a **local, free, AI-native knowledge workspace** that borrows
Obsidian's interface language (left file/nav rail, tabbed main panes, right
context sidebar, command palette, backlinks, a graph view, daily notes) but where
the "notes" are alive: you chat with a local model, attach documents that get
page-cited, and everything you do is captured as linked notes you can browse,
search, schedule, and turn into tasks. Think **"Obsidian, but the notes talk back,
and it runs entirely on your Mac for free."**

---

## 2. Where FreeLAB is today (inventory for the next session)

Working app, in `Projects/FreeLAB`. Core pipeline is solid and tested; the UI has
grown organically in Streamlit and is the thing being redesigned.

**Code**
- `freelab_core.py` (~500 lines) — the engine, UI-agnostic. Ollama calls, `embed`,
  `chunk_text` / `chunk_pages` (page-aware), `compress` / `rank_chunks` (query-aware
  retrieval), the `draft → judge → answer` cascade, `answer_from_chunks(_doc)`
  (page-cited), `chat_answer`, `web_search` / `web_answer` (DuckDuckGo), `extract_pages`
  + OCR fallback (PyMuPDF + Tesseract), `current_date_str`. **Keep this as the backend.**
- `freelab_streamlit.py` (~840 lines) — the current UI (the part we're redesigning).
- `freelab.py` (CLI), `freelab_eval.py` (evals), `freelab_app.py` (older Gradio UI),
  `freelab_compress.py` / `freelab_cascade.py` (standalone stages).

**Models (local, via Ollama):** `qwen3:0.6b` (drafter), `qwen3:8b` (judge/verifier),
`nomic-embed-text` (embeddings). Runs on a 16 GB Apple Silicon Mac.

**Features already built (must be preserved in v2)**
- Chat with the cascade (small drafts, big verifies).
- Document Q&A: attach PDF/TXT/MD/DOCX, embed once, ask; **scanned PDFs OCR'd**.
- **Page-cited answers**: inline "(p. 12)" + a "Sourced from page(s)…" footer.
- **Sources panel**: cited passages with page images, and a "Full document"
  scrollable reader that flags which pages the AI used.
- **Batch** mode (questions file + optional doc → CSV).
- **Web lookup** toggle (opt-in; otherwise fully local/offline).
- **Conversation history** (multiple chats, persisted to disk).
- **Memory graph** (Plotly + networkx): chats + documents as nodes, clustered by
  topic similarity and shared documents; tap a node to open the chat.
- Persistence: `freelab_data/state.pkl` survives restarts.
- Settings on their own tab (keep, threshold, cascade, models, web, show-sources).

**Known constraints learned this session (save the next session time)**
- Editing `freelab_core.py` requires a **full Streamlit restart** (imported modules
  are cached); editing the UI file only needs a Rerun.
- Streamlit clears a widget's `session_state` value when the widget leaves the page;
  use the `value=`/assign pattern or a "pending" flag, not `key=`-bound reads, for
  cross-view state. Never write a widget's own key after it's instantiated.
- Runs in a venv; extra deps installed: `pymupdf pytesseract pillow ddgs plotly networkx`
  (+ system `tesseract` via Homebrew).

---

## 3. The one big decision to make FIRST — the UI architecture

Getting genuinely close to Obsidian (multi-pane, drag-to-split, command palette,
resizable sidebars, smooth theming, backlinks hover) is **hard in Streamlit**.
Streamlit is a top-to-bottom script renderer; it fights you on persistent panels,
tabs-as-windows, drag/drop, and keyboard-driven UX. Decide the path before building:

**Option A — Push Streamlit further (fastest, lowest fidelity).**
Keep the current stack. Use `st.tabs`, columns, `st.navigation`/pages, custom CSS,
and components for the graph. You can get an "Obsidian-ish" three-column layout and
theme, but not real panes, command palette, or drag/drop. Good for a quick v1.5.
Risk: you hit Streamlit's ceiling and the result still feels like Streamlit.

**Option B — Custom web frontend + Python backend (recommended for true fidelity).**
Wrap `freelab_core.py` in a small **FastAPI** service, and build the UI as a
**React/Svelte** single-page app (or Tauri/Electron for a desktop app) that actually
mimics Obsidian: left rail, tabbed workspace, right panels, command palette (⌘P),
backlinks, a real graph (Cytoscape/D3), calendar, tasks. More work, but this is the
only way to get "as close as possible to Obsidian." The core AI logic doesn't change
— only the presentation layer and a thin HTTP API.

**Option C — Build INTO Obsidian (reuse Obsidian itself).**
Henry already runs Obsidian (the Second Brain vault). Instead of re-creating
Obsidian, make FreeLAB an **Obsidian plugin** (or a local companion server the plugin
calls) so the AI lives inside the real Obsidian: the vault, graph, backlinks,
calendar, and Tasks plugins already exist and are best-in-class. FreeLAB provides the
local model chat, document Q&A, and citations as panes/commands. Highest reuse,
truly Obsidian — but a different codebase (TypeScript plugin) and you inherit
Obsidian's extension model.

**Recommendation.** If the priority is *fidelity to Obsidian*, do **C** (or **B**).
If the priority is *ship something soon on the current code*, do **A** as a stepping
stone. A sensible sequence: **A now** (reorganize + theme + tasks/calendar/graph in
Streamlit) → evaluate → **B/C later** if the ceiling is limiting. The next session
should get Henry to pick, because it changes everything downstream.

---

## 4. Adopt an Obsidian-compatible data model (do this regardless of UI choice)

The most important architectural upgrade: **stop storing state in a pickle and start
storing everything as Markdown files with YAML frontmatter** — i.e., an actual
Obsidian-style vault folder. This single change unlocks calendar, tasks, wiki-links,
graph, search, and interop with Henry's existing Second Brain, and makes the data
portable and human-readable.

Proposed layout (a "FreeLAB vault" folder, optionally *inside* the Obsidian vault):
```
FreeLAB-Vault/
  Chats/         one .md per conversation (frontmatter: created, model, doc, tags)
  Sources/       one .md per attached document (metadata, extracted text, page map)
  Notes/         wiki/concept notes the user or AI writes ([[wikilinks]])
  Daily/         YYYY-MM-DD.md daily notes  ← powers the calendar
  Tasks are just `- [ ]` lines living inside any note (Obsidian Tasks convention)
  .freelab/      embeddings cache, indexes (not shown in the graph)
```
- **Links** come from `[[wikilinks]]` (explicit) + FreeLAB's semantic similarity
  (suggested links via embeddings) — same idea as today's graph, but persisted.
- **Tasks** = parse `- [ ] text 📅 2026-08-01` lines across notes (Obsidian Tasks /
  Dataview syntax). No separate DB needed.
- **Calendar** = Daily notes + any note with a date in frontmatter or a task due date.
- This means FreeLAB's memory *is* a vault Obsidian can already open — bridging to
  the Second Brain instead of duplicating it.

---

## 5. Target UX — the Obsidian feel (feature spec)

**Global layout (Obsidian's anatomy)**
- **Left rail / sidebar:** collapsible. File explorer (Chats, Notes, Daily, Sources),
  plus tabs for Search and Starred. This replaces today's flat "Chat history".
- **Main workspace:** tabbed, ideally splittable panes. A tab can be a chat, a note,
  a source reader, the graph, the calendar, or a task board.
- **Right sidebar:** context panels — **Backlinks**, **Outline**, and the **Sources**
  panel (page images + cited passages) when a chat/doc is open.
- **Command palette (⌘P):** "New chat", "Open note", "Insert wikilink", "Toggle web
  lookup", "Go to today", "Run batch"… keyboard-first.
- **Theme:** Obsidian-like typography and spacing; light + dark; the warm accent is
  fine. Calm, dense-but-readable, minimal chrome.

**Feature modules**
1. **Chat-as-note.** Every conversation is a Markdown note. Answers keep inline
   page citations. Chats can `[[link]]` to notes/sources and show up in the graph.
2. **Wiki notes.** Create/edit Markdown notes with `[[wikilinks]]`, backlinks pane,
   and hover-preview. The AI can draft/update notes on request (ties into the
   existing Second Brain workflow).
3. **Graph view (enhanced).** Chats + notes + sources as nodes; edges from wikilinks,
   shared documents, and semantic similarity (already prototyped). Add: filter by
   type, color by folder/tag, local-graph (neighbors of the current note),
   click-to-open. Consider Cytoscape/D3 if not in Streamlit.
4. **Calendar.** Month/week view of Daily notes + task due dates + chat activity.
   Click a day → open/create that Daily note. (Obsidian Calendar plugin as the model.)
5. **To-dos / tasks.** Parse `- [ ]` items across the vault into a task view: due
   today, upcoming, by note/tag; check off in place; "add task" command. The AI can
   create tasks from a chat ("remind me to…", "turn this into tasks").
6. **Sources / reader.** Keep the current page-image reader and "which pages were
   cited" flagging; make it a proper pane.
7. **Search.** Full-text across the vault + semantic search (embeddings) — "find
   notes like this".
8. **Settings.** Themes, models (keep/threshold/cascade), vault path, web toggle.

---

## 6. Phased roadmap (suggested milestones)

1. **Data model migration.** Move persistence from `state.pkl` → Markdown vault with
   frontmatter. Write/read chats, sources, notes, daily notes. (UI-agnostic; do first.)
2. **Layout & theme.** Rebuild the shell in the chosen stack: left rail, tabbed main,
   right context sidebar, Obsidian-like theme (light/dark).
3. **Notes + wikilinks + backlinks.** Editor, `[[ ]]` autocomplete, backlinks pane.
4. **Graph v2.** Wikilink + shared-doc + semantic edges; filters; local graph;
   click-to-open. (Reuse today's networkx/embedding logic.)
5. **Calendar + Daily notes.**
6. **Tasks.** Parse `- [ ]`, task view, check-off, AI task creation.
7. **Command palette + search** (full-text + semantic).
8. **Polish:** keyboard shortcuts, hover previews, split panes, starred items.

Keep the existing chat / doc-Q&A / citations / web / batch working throughout —
they're the differentiator; the redesign is about *organizing* them like Obsidian.

---

## 7. Open questions for Henry (answer these at the start of the next session)

1. **UI path:** A (extend Streamlit, fast), B (custom React/Tauri frontend, true
   fidelity), or C (Obsidian plugin, reuse Obsidian)? This decides the whole build.
2. **One vault or two:** should FreeLAB write into the **existing Second Brain vault**
   (deep integration, everything in one graph) or its **own FreeLAB vault** (clean
   separation)? A subfolder of the Second Brain is a nice middle ground.
3. **Desktop app or browser:** keep it a localhost web app, or package as a real
   desktop app (Tauri/Electron) for the Obsidian-like window feel?
4. **Light or dark default**, and how closely to clone Obsidian's exact theme.
5. **Scope of v2.1** vs later: which of {graph, calendar, tasks, command palette}
   are must-haves for the first redesigned release?
6. **Editing model:** do you want to *edit* notes in FreeLAB, or is FreeLAB mainly a
   capture+chat surface and you edit prose in Obsidian itself?

---

## 8. Kickoff prompt for the next session (paste this to start)

> I'm redesigning my local AI app "FreeLAB" (in `Projects/FreeLAB`) to look and feel
> like **Obsidian** while keeping its AI features. Please read
> `FreeLAB v2 — Obsidian-Style Redesign Plan.md` in that folder for full context and
> the current state. FreeLAB is a free, local pipeline (compress → draft → judge →
> answer) on Ollama with page-cited document Q&A, OCR, web lookup, conversation
> history, and a memory graph; the engine is `freelab_core.py` (keep it) and the
> current UI is `freelab_streamlit.py` (redesign this).
>
> Before building, help me decide the three things in the "Open questions" section —
> especially the **UI architecture** (extend Streamlit vs. a custom React/Tauri
> frontend vs. an Obsidian plugin) and whether to store data as a **Markdown vault**
> (frontmatter + `[[wikilinks]]`) so it interoperates with my existing Second Brain.
> Then let's build toward: an Obsidian-style shell (left file rail, tabbed main pane,
> right backlinks/sources sidebar, command palette, light/dark theme), plus a
> **calendar** (daily notes), **to-do/tasks** (parsed `- [ ]` items), and an enhanced
> **wiki graph view**. Follow the phased roadmap in the doc, and keep all existing
> chat / document-citation / web / batch functionality working.

---

## 9. Reference — key files & functions to reuse

- **Backend to keep as-is:** `freelab_core.py` → `chat_answer`, `answer_from_chunks`,
  `answer_from_chunks_doc`, `rank_chunks`, `compress`, `extract_pages` (+OCR),
  `web_answer`, `embed`, `cosine`, `chunk_pages`.
- **Logic to lift from `freelab_streamlit.py`:** `ingest_files` (embed-once + OCR
  progress), `render_page` (PDF page → PNG), the sources/reader rendering, the graph
  builder (`_build_graph`, `_conv_embedding`, spring layout + component coloring),
  persistence (`save_state`/`load_state` → replace with vault read/write).
- **Design references:** Obsidian's UI (left rail, panes, ⌘P, graph, backlinks),
  Obsidian **Calendar**, **Tasks**, and **Dataview** community plugins for the
  calendar/task conventions.

## Related (in the vault)
- [[FreeLAB Build Log]] — full running history of what's built and why.
- [[FreeLAB Project Plan]] — the original pipeline design.
- [[OCR (Optical Character Recognition)]], [[Query-Aware Context Compression]],
  [[LLM-as-Judge]], [[RAG Architecture]].

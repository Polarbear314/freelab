#!/usr/bin/env python3
"""
FreeLAB API -- the local HTTP backend for FreeLAB v2 (Obsidian-style redesign).

This is the "Option B" backend: a thin FastAPI service that wraps the untouched
`freelab_core.py` engine and reads/writes the Markdown vault (`freelab_vault.py`).
The v2 web SPA (left rail, tabbed panes, ⌘P palette, graph, calendar, tasks) talks
to these JSON endpoints; no AI logic lives here.

Everything persists to the vault, so state survives restarts and Obsidian can open
the same folder. Runs entirely on localhost against local Ollama -- still free,
still offline unless web lookup is turned on per request.

RUN
    pip install fastapi uvicorn python-multipart
    uvicorn freelab_api:app --port 8765 --reload

ENDPOINTS (all under /api)
    GET    /health                    liveness + whether Ollama is reachable
    GET    /chats                     list chats (id, title, updated, doc, n msgs)
    POST   /chats                     create a new chat  -> chat
    GET    /chats/{id}                one chat (history, sources, doc meta)
    DELETE /chats/{id}                delete a chat (+ its cache)
    PATCH  /chats/{id}                rename ({title})
    POST   /chats/{id}/ask            ask a question ({text, web?}) -> answer, persisted
    POST   /chats/{id}/ingest         multipart upload -> embed once, attach index
    GET    /graph                     nodes + edges (chats/sources, semantic edges)
    GET    /tasks                     parsed `- [ ]` items across the vault
    GET    /daily/{date}              a daily note (creates on demand)
    PUT    /daily/{date}              write a daily note ({body})
    GET    /settings  / PUT /settings keep / threshold / cascade / models / web
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
import urllib.error
import urllib.request

import freelab_core as fl
import freelab_engine as engine
import freelab_vault as vault

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover - import guard for a bare environment
    raise SystemExit("FreeLAB API needs FastAPI: pip install fastapi uvicorn "
                     "python-multipart") from e

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")

app = FastAPI(title="FreeLAB API", version="2.0")
# CORS stays permissive so the UI also works if opened from a separate dev server
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the single-file Obsidian-style UI (web/index.html)."""
    path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>FreeLAB</h1><p>web/index.html not found.</p>",
                            status_code=404)
    return FileResponse(path)

VAULT = os.environ.get("FREELAB_VAULT", vault.DEFAULT_VAULT)

# in-memory runtime settings (persisted lightly to the cache)
SETTINGS = {"keep": fl.DEFAULT_KEEP, "threshold": fl.DEFAULT_THRESHOLD,
            "use_cascade": True, "web": False,
            "direct_below": engine.DIRECT_BELOW_TOKENS,
            "drafter": fl.DRAFTER, "verifier": fl.VERIFIER,
            "embed_model": fl.EMBED_MODEL}


# --- helpers -------------------------------------------------------------------
def _apply_models():
    fl.DRAFTER = SETTINGS["drafter"]
    fl.VERIFIER = SETTINGS["verifier"]
    fl.EMBED_MODEL = SETTINGS["embed_model"]


def _load_conv(cid: str) -> dict:
    for c in vault.list_chats(VAULT):
        if c["id"] == cid:
            return c
    raise HTTPException(404, f"chat {cid} not found")


def _chat_summary(c: dict) -> dict:
    return {"id": c["id"], "title": c.get("title") or "New chat",
            "updated": c.get("updated"), "created": c.get("created"),
            "doc": (c.get("index") or {}).get("name") or "",
            "messages": len(c.get("history") or [])}


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(fl.OLLAMA + "/api/tags", timeout=1.5)
        return True
    except Exception:
        return False


# --- request models ------------------------------------------------------------
class AskBody(BaseModel):
    text: str
    web: bool | None = None


class RenameBody(BaseModel):
    title: str


class DailyBody(BaseModel):
    body: str


class SettingsBody(BaseModel):
    keep: int | None = None
    threshold: int | None = None
    use_cascade: bool | None = None
    web: bool | None = None
    direct_below: int | None = None
    drafter: str | None = None
    verifier: str | None = None
    embed_model: str | None = None


# --- health --------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "ollama": _ollama_up(), "vault": VAULT}


@app.post("/api/warmup")
def warmup():
    """Preload the drafter/verifier/embed models so the first question is fast."""
    _apply_models()
    logs = []
    try:
        fl.warmup(use_compress=True, use_cascade=SETTINGS["use_cascade"],
                  log=logs.append)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"{engine.OLLAMA_DOWN}  ({e})")
    return {"ok": True, "log": logs,
            "models": [SETTINGS["drafter"], SETTINGS["verifier"],
                       SETTINGS["embed_model"]]}


# --- chats ---------------------------------------------------------------------
@app.get("/api/chats")
def list_chats():
    chats = sorted(vault.list_chats(VAULT),
                   key=lambda c: c.get("updated") or "", reverse=True)
    return {"chats": [_chat_summary(c) for c in chats],
            "current": vault.get_current(VAULT)}


@app.post("/api/chats")
def new_chat():
    cid = f"c{int(time.time() * 1000)}"
    conv = {"id": cid, "title": "New chat", "history": [],
            "created": vault._now_iso()}
    vault.write_chat(conv, VAULT)
    vault.set_current(cid, VAULT)
    return _chat_summary(conv)


@app.get("/api/chats/{cid}")
def get_chat(cid: str):
    c = _load_conv(cid)
    vault.set_current(cid, VAULT)
    index = c.get("index") or {}
    return {"id": c["id"], "title": c.get("title"), "history": c.get("history"),
            "doc": index.get("name") or "",
            "pages": index.get("pages") or 0,
            "has_pdf": bool(index.get("pdf_bytes")),
            "last_sources": c.get("last_sources") or [],
            "last_query": c.get("last_query")}


@app.get("/api/chats/{cid}/page/{page}")
def chat_page(cid: str, page: int, dpi: int = 130):
    """Render one page of the chat's attached PDF to PNG (NotebookLM-style reader)."""
    ppath = os.path.join(vault._cache_dir(cid, VAULT), "source.pdf")
    if not os.path.exists(ppath):
        raise HTTPException(404, "no PDF attached to this chat")
    try:
        import fitz
    except Exception:
        raise HTTPException(503, "PyMuPDF not installed (pip install pymupdf)")
    doc = fitz.open(ppath)
    try:
        if page < 1 or page > doc.page_count:
            raise HTTPException(404, "page out of range")
        png = doc.load_page(page - 1).get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


@app.patch("/api/chats/{cid}")
def rename_chat(cid: str, body: RenameBody):
    c = _load_conv(cid)
    c["title"] = body.title.strip() or "New chat"
    vault.write_chat(c, VAULT)
    return _chat_summary(c)


@app.delete("/api/chats/{cid}")
def delete_chat(cid: str):
    import shutil
    c = _load_conv(cid)
    vault._prune_stale(cid, "", VAULT)               # remove the .md
    cdir = vault._cache_dir(cid, VAULT)
    if os.path.isdir(cdir):
        shutil.rmtree(cdir, ignore_errors=True)
    return {"deleted": cid}


@app.post("/api/chats/{cid}/ask")
def ask(cid: str, body: AskBody):
    _apply_models()
    c = _load_conv(cid)
    web = SETTINGS["web"] if body.web is None else body.web
    try:
        answer = engine.answer_query(
            body.text, c, keep=SETTINGS["keep"], threshold=SETTINGS["threshold"],
            use_cascade=SETTINGS["use_cascade"], web_lookup=web,
            direct_below=SETTINGS["direct_below"])
    except urllib.error.URLError:
        raise HTTPException(503, engine.OLLAMA_DOWN)
    c.setdefault("history", []).append({"role": "user", "content": body.text})
    c["history"].append({"role": "assistant", "content": answer})
    if (c.get("title") or "New chat") == "New chat":
        c["title"] = body.text.strip()[:50] or "New chat"
    vault.write_chat(c, VAULT)
    return {"answer": answer, "title": c["title"],
            "last_sources": c.get("last_sources") or []}


@app.post("/api/chats/{cid}/ask_stream")
def ask_stream(cid: str, body: AskBody):
    """Same as /ask, but streams the REAL pipeline stages as Server-Sent Events
    so the UI can show a live 'thinking' ticker, then a final 'done' event with
    the answer. Reports stages that already run -- no extra model compute."""
    _apply_models()
    c = _load_conv(cid)
    web = SETTINGS["web"] if body.web is None else body.web
    events: "queue.Queue" = queue.Queue()

    def on_stage(name, info):
        events.put({"type": "stage", "name": name, "info": info})

    def run():
        try:
            answer = engine.answer_query(
                body.text, c, keep=SETTINGS["keep"], threshold=SETTINGS["threshold"],
                use_cascade=SETTINGS["use_cascade"], web_lookup=web, on_stage=on_stage,
                direct_below=SETTINGS["direct_below"])
            c.setdefault("history", []).append({"role": "user", "content": body.text})
            c["history"].append({"role": "assistant", "content": answer})
            if (c.get("title") or "New chat") == "New chat":
                c["title"] = body.text.strip()[:50] or "New chat"
            vault.write_chat(c, VAULT)
            events.put({"type": "done", "answer": answer, "title": c["title"],
                        "last_sources": c.get("last_sources") or []})
        except urllib.error.URLError:
            events.put({"type": "error", "message": engine.OLLAMA_DOWN})
        except Exception as e:  # noqa: BLE001
            events.put({"type": "error", "message": str(e)})
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def gen():
        while True:
            item = events.get()
            if item is None:
                break
            yield "data: " + json.dumps(item) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/chats/{cid}/ingest")
async def ingest(cid: str, files: list[UploadFile] = File(...)):
    _apply_models()
    c = _load_conv(cid)
    paths, names, tmpdir = [], [], tempfile.mkdtemp(prefix="freelab_up_")
    try:
        for f in files:
            dest = os.path.join(tmpdir, f.filename)
            with open(dest, "wb") as out:
                out.write(await f.read())
            paths.append(dest)
            names.append(f.filename)
        try:
            index = engine.ingest_files(paths, names)
        except ValueError as e:
            raise HTTPException(422, str(e))
        c["index"] = index
        vault.write_chat(c, VAULT)
        # also register the document as a Source note
        vault.write_source({"id": vault.slugify(index["name"]),
                            "name": index["name"], "pages": index["pages"],
                            "orig_tok": index["orig_tok"]}, VAULT)
        return {"doc": index["name"], "pages": index["pages"],
                "chunks": len(index["chunks"]), "tokens": index["orig_tok"]}
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- graph (semantic edges over chats + shared docs) ---------------------------
@app.get("/api/graph")
def graph():
    chats = [c for c in vault.list_chats(VAULT) if c.get("history")]
    nodes, edges, embs = [], [], {}
    for c in chats:
        doc = (c.get("index") or {}).get("name") or ""
        nodes.append({"id": c["id"], "label": c.get("title") or "chat",
                      "type": "chat", "doc": doc})
        try:
            text = " ".join(m.get("content", "") for m in c["history"])[:2000]
            embs[c["id"]] = fl.embed([text])[0] if text.strip() else None
        except Exception:
            embs[c["id"]] = None
    # shared-document edges (through a doc node)
    docs = {}
    for c in chats:
        doc = (c.get("index") or {}).get("name")
        if doc:
            docs.setdefault(doc, []).append(c["id"])
    for doc, ids in docs.items():
        did = "doc:" + vault.slugify(doc)
        nodes.append({"id": did, "label": doc, "type": "source"})
        for cid in ids:
            edges.append({"source": cid, "target": did, "kind": "doc"})
    # semantic edges (cosine >= 0.6)
    ids = [c["id"] for c in chats]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = embs[ids[i]], embs[ids[j]]
            if a and b:
                sim = fl.cosine(a, b)
                if sim >= 0.6:
                    edges.append({"source": ids[i], "target": ids[j],
                                  "kind": "semantic", "weight": round(sim, 3)})
    return {"nodes": nodes, "edges": edges}


# --- tasks / daily / settings --------------------------------------------------
@app.get("/api/tasks")
def tasks():
    return {"tasks": vault.parse_tasks(VAULT), "today": vault.today_str()}


@app.get("/api/daily/{date}")
def get_daily(date: str):
    path = vault.daily_path(date, VAULT)
    if not os.path.exists(path):
        vault.write_daily(date, "", VAULT)
    _, body = vault.parse_frontmatter(vault._read(path))
    return {"date": date, "body": body}


@app.put("/api/daily/{date}")
def put_daily(date: str, body: DailyBody):
    vault.write_daily(date, body.body, VAULT)
    return {"date": date, "ok": True}


@app.get("/api/settings")
def get_settings():
    return SETTINGS


@app.put("/api/settings")
def put_settings(body: SettingsBody):
    for k, v in body.model_dump(exclude_none=True).items():
        SETTINGS[k] = v
    _apply_models()
    return SETTINGS


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("freelab_api:app", host="127.0.0.1", port=8765, reload=True)

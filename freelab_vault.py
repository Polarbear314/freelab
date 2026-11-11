#!/usr/bin/env python3
"""
FreeLAB vault -- the Markdown persistence layer for FreeLAB v2.

This replaces the single `freelab_data/state.pkl` with an actual Obsidian-style
vault: one Markdown file per chat / source / note / daily note, each with YAML
frontmatter and `[[wikilinks]]`. The point is portability and interop -- Obsidian
(or anything) can open the folder directly, and it unlocks the calendar, tasks,
wiki-links, and graph without a bespoke database.

Design rules (match the rest of FreeLAB):
  * No external dependencies -- a tiny flat-YAML reader/writer lives here, so the
    module works on a bare Python 3. (Obsidian only needs simple scalar/list
    frontmatter, which is all FreeLAB writes.)
  * Human-readable + LOSSLESS round-trip. The chat transcript renders as normal
    Markdown in Obsidian; invisible HTML-comment sentinels let the app re-parse
    the exact message list back out.
  * Binary blobs (embedding vectors, raw PDF bytes) never go in Markdown. They
    live in a hidden `.freelab/<chat-id>/` cache the graph ignores.

Vault layout
    FreeLAB-Vault/
      Chats/    <id>--<slug>.md      frontmatter: id, title, created, updated, model, doc, tags
      Sources/  <slug>.md            frontmatter: id, name, pages, tokens, added
      Notes/    <slug>.md            free wiki/concept notes ([[wikilinks]])
      Daily/    YYYY-MM-DD.md         powers the calendar
      .freelab/ <chat-id>/           index.json (chunks/pageno/meta), vecs.json, source.pdf
                _meta.json            {current: <chat-id>}

CLI
    python3 freelab_vault.py migrate [--pickle PATH] [--vault PATH]   # state.pkl -> vault
    python3 freelab_vault.py doctor  [--vault PATH]                   # summarize a vault
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re

# --- where the vault lives -----------------------------------------------------
# Own folder, sibling of the app (the "Markdown vault, own folder" choice).
# Override with the FREELAB_VAULT env var.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VAULT = os.environ.get("FREELAB_VAULT", os.path.join(_HERE, "FreeLAB-Vault"))

SUBDIRS = ("Chats", "Sources", "Notes", "Daily")
CACHE_DIR = ".freelab"

# sentinel comments: invisible in Obsidian preview, exact round-trip for the app
_MSG_OPEN = "<!-- freelab:msg role={role} -->"
_MSG_OPEN_RE = re.compile(r"<!--\s*freelab:msg\s+role=(\w+)\s*-->")
_MSG_CLOSE = "<!-- /freelab:msg -->"

_ROLE_LABEL = {"user": "🧑 **You**", "assistant": "⚡ **FreeLAB**", "system": "⚙️ **System**"}


# =============================================================================
# tiny flat-YAML frontmatter (scalars + one-level string lists only)
# =============================================================================
def _yaml_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    s = str(v)
    # quote if it could confuse a YAML reader
    if s == "" or s[0] in "#&*!|>%@`\"'[]{}," or ": " in s or s.strip() != s:
        return '"' + s.replace('"', '\\"') + '"'
    return s


def dump_frontmatter(meta: dict) -> str:
    """Serialize a flat dict to a `---` fenced YAML block. Lists -> `[a, b]`."""
    lines = ["---"]
    for k, v in meta.items():
        if isinstance(v, (list, tuple)):
            inner = ", ".join(_yaml_scalar(x) for x in v)
            lines.append(f"{k}: [{inner}]")
        else:
            lines.append(f"{k}: {_yaml_scalar(v)}")
    lines.append("---")
    return "\n".join(lines)


def _parse_scalar(s: str):
    s = s.strip()
    if s == "":
        return ""
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1].replace('\\"', '"')
    if s == "true":
        return True
    if s == "false":
        return False
    return s


def parse_frontmatter(text: str):
    """Return (meta_dict, body). Tolerates a missing frontmatter block."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end].strip("\n")
    body = text[end + 4:]
    if body.startswith("\n"):
        body = body[1:]
    meta = {}
    for line in block.splitlines():
        if not line.strip() or ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            meta[k] = [_parse_scalar(x) for x in _split_list(inner)] if inner else []
        else:
            meta[k] = _parse_scalar(v)
    return meta, body


def _split_list(inner: str):
    """Split `a, "b, c", d` honoring quotes."""
    out, buf, q = [], [], False
    for ch in inner:
        if ch == '"':
            q = not q
            buf.append(ch)
        elif ch == "," and not q:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [s.strip() for s in out]


# =============================================================================
# vault plumbing
# =============================================================================
def slugify(text: str, fallback: str = "untitled", maxlen: int = 60) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return (text or fallback)[:maxlen]


def ensure_vault(vault: str = DEFAULT_VAULT) -> str:
    for d in SUBDIRS:
        os.makedirs(os.path.join(vault, d), exist_ok=True)
    os.makedirs(os.path.join(vault, CACHE_DIR), exist_ok=True)
    return vault


def _now_iso() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def today_str() -> str:
    return _dt.date.today().isoformat()


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =============================================================================
# chats  (Chats/<id>--<slug>.md  +  .freelab/<id>/ for vectors & pdf)
# =============================================================================
def chat_path(conv: dict, vault: str = DEFAULT_VAULT) -> str:
    slug = slugify(conv.get("title") or "chat")
    return os.path.join(vault, "Chats", f"{conv['id']}--{slug}.md")


def _render_transcript(history) -> str:
    parts = []
    for m in history or []:
        role = m.get("role", "user")
        label = _ROLE_LABEL.get(role, role)
        parts.append(_MSG_OPEN.format(role=role))
        parts.append(f"### {label}\n")
        parts.append((m.get("content") or "").rstrip())
        parts.append(_MSG_CLOSE)
        parts.append("")  # blank line between messages
    return "\n".join(parts).rstrip() + "\n"


def _parse_transcript(body: str):
    """Recover the exact [{role, content}] list from a rendered transcript."""
    history = []
    pos = 0
    for m in _MSG_OPEN_RE.finditer(body):
        role = m.group(1)
        start = m.end()
        close = body.find(_MSG_CLOSE, start)
        if close == -1:
            close = len(body)
        chunk = body[start:close]
        # drop the "### label" heading line we added for readability
        chunk = re.sub(r"^\s*###[^\n]*\n?", "", chunk, count=1)
        history.append({"role": role, "content": chunk.strip("\n")})
        pos = close
    return history


def write_chat(conv: dict, vault: str = DEFAULT_VAULT) -> str:
    """Persist one conversation as Markdown; cache its index/pdf separately."""
    ensure_vault(vault)
    doc_name = (conv.get("index") or {}).get("name") or ""
    meta = {
        "id": conv["id"],
        "title": conv.get("title") or "New chat",
        "created": conv.get("created") or _now_iso(),
        "updated": _now_iso(),
        "model": conv.get("model") or "",
        "doc": doc_name,
        "type": "chat",
        "tags": conv.get("tags") or ["freelab/chat"],
    }
    body = _render_transcript(conv.get("history"))
    path = chat_path(conv, vault)
    # if the title/slug changed, remove the stale file for this id
    _prune_stale(conv["id"], path, vault)
    _write(path, dump_frontmatter(meta) + "\n\n" + body)
    _write_chat_cache(conv, vault)
    return path


def _prune_stale(cid, keep_path, vault):
    d = os.path.join(vault, "Chats")
    if not os.path.isdir(d):
        return
    for fn in os.listdir(d):
        if fn.startswith(f"{cid}--") and os.path.join(d, fn) != keep_path:
            try:
                os.remove(os.path.join(d, fn))
            except OSError:
                pass


def _cache_dir(cid, vault):
    return os.path.join(vault, CACHE_DIR, cid)


def _write_chat_cache(conv: dict, vault: str):
    """Vectors + chunk map + raw PDF -> hidden cache (never in Markdown)."""
    index = conv.get("index")
    cdir = _cache_dir(conv["id"], vault)
    if index:
        os.makedirs(cdir, exist_ok=True)
        meta = {k: index.get(k) for k in ("name", "pageno", "pages", "orig_tok")}
        meta["chunks"] = index.get("chunks") or []
        _write(os.path.join(cdir, "index.json"), json.dumps(meta))
        vecs = index.get("vecs")
        if vecs is not None:
            _write(os.path.join(cdir, "vecs.json"),
                   json.dumps([list(v) for v in vecs]))
        pdf = index.get("pdf_bytes") or conv.get("pdf_bytes")
        if pdf:
            os.makedirs(cdir, exist_ok=True)
            with open(os.path.join(cdir, "source.pdf"), "wb") as f:
                f.write(pdf if isinstance(pdf, (bytes, bytearray)) else bytes(pdf))
    if conv.get("last_sources"):
        os.makedirs(cdir, exist_ok=True)
        _write(os.path.join(cdir, "last_sources.json"),
               json.dumps({"query": conv.get("last_query"),
                           "sources": conv["last_sources"]}))


def read_chat(path: str) -> dict:
    """Load one chat .md (and its cache) back into a conversation dict."""
    meta, body = parse_frontmatter(_read(path))
    conv = {
        "id": meta.get("id") or slugify(os.path.basename(path)),
        "title": meta.get("title") or "New chat",
        "created": meta.get("created"),
        "updated": meta.get("updated"),
        "model": meta.get("model") or "",
        "tags": meta.get("tags") or [],
        "history": _parse_transcript(body),
    }
    vault = os.path.dirname(os.path.dirname(path))
    _load_chat_cache(conv, vault)
    return conv


def _load_chat_cache(conv: dict, vault: str):
    cdir = _cache_dir(conv["id"], vault)
    ipath = os.path.join(cdir, "index.json")
    if os.path.exists(ipath):
        index = json.loads(_read(ipath))
        vpath = os.path.join(cdir, "vecs.json")
        index["vecs"] = json.loads(_read(vpath)) if os.path.exists(vpath) else []
        ppath = os.path.join(cdir, "source.pdf")
        if os.path.exists(ppath):
            with open(ppath, "rb") as f:
                index["pdf_bytes"] = f.read()
        else:
            index["pdf_bytes"] = None
        conv["index"] = index
        conv["pdf_bytes"] = index.get("pdf_bytes")
    lpath = os.path.join(cdir, "last_sources.json")
    if os.path.exists(lpath):
        ls = json.loads(_read(lpath))
        conv["last_query"] = ls.get("query")
        conv["last_sources"] = ls.get("sources") or []


def list_chats(vault: str = DEFAULT_VAULT):
    d = os.path.join(vault, "Chats")
    if not os.path.isdir(d):
        return []
    return [read_chat(os.path.join(d, fn))
            for fn in sorted(os.listdir(d)) if fn.endswith(".md")]


# =============================================================================
# sources / notes / daily
# =============================================================================
def write_source(source: dict, vault: str = DEFAULT_VAULT) -> str:
    ensure_vault(vault)
    slug = slugify(source.get("name") or source.get("id") or "source")
    meta = {
        "id": source.get("id") or slug,
        "name": source.get("name") or "",
        "pages": source.get("pages") or 0,
        "tokens": source.get("orig_tok") or source.get("tokens") or 0,
        "added": source.get("added") or _now_iso(),
        "type": "source",
        "tags": source.get("tags") or ["freelab/source"],
    }
    body = source.get("body") or f"Attached document **{meta['name']}** " \
                                 f"({meta['pages']} page(s), ~{meta['tokens']} tokens).\n"
    path = os.path.join(vault, "Sources", f"{slug}.md")
    _write(path, dump_frontmatter(meta) + "\n\n" + body)
    return path


def write_note(title: str, body: str, tags=None, vault: str = DEFAULT_VAULT) -> str:
    ensure_vault(vault)
    meta = {"title": title, "created": _now_iso(), "type": "note",
            "tags": tags or ["freelab/note"]}
    path = os.path.join(vault, "Notes", f"{slugify(title)}.md")
    _write(path, dump_frontmatter(meta) + "\n\n" + (body or ""))
    return path


def daily_path(date: str | None = None, vault: str = DEFAULT_VAULT) -> str:
    return os.path.join(vault, "Daily", f"{date or today_str()}.md")


def write_daily(date: str | None, body: str, vault: str = DEFAULT_VAULT) -> str:
    ensure_vault(vault)
    date = date or today_str()
    meta = {"date": date, "type": "daily", "tags": ["freelab/daily"]}
    path = daily_path(date, vault)
    _write(path, dump_frontmatter(meta) + "\n\n" + (body or ""))
    return path


def append_daily(text: str, date: str | None = None, vault: str = DEFAULT_VAULT) -> str:
    date = date or today_str()
    path = daily_path(date, vault)
    if os.path.exists(path):
        _write(path, _read(path).rstrip() + "\n" + text + "\n")
    else:
        write_daily(date, text + "\n", vault)
    return path


# =============================================================================
# tasks  (Obsidian Tasks convention: `- [ ] text 📅 YYYY-MM-DD`)
# =============================================================================
_TASK_RE = re.compile(r"^\s*[-*]\s+\[(?P<done>[ xX])\]\s+(?P<text>.*)$")
_DUE_RE = re.compile(r"📅\s*(\d{4}-\d{2}-\d{2})")


def parse_tasks(vault: str = DEFAULT_VAULT):
    """Scan every note in the vault for `- [ ]` items. Returns dicts with
    text, done, due (or None), the source note path, and line number."""
    tasks = []
    for root, dirs, files in os.walk(vault):
        dirs[:] = [d for d in dirs if d != CACHE_DIR]
        for fn in files:
            if not fn.endswith(".md"):
                continue
            path = os.path.join(root, fn)
            for i, line in enumerate(_read(path).splitlines(), 1):
                m = _TASK_RE.match(line)
                if not m:
                    continue
                text = m.group("text").strip()
                due = _DUE_RE.search(text)
                tasks.append({
                    "text": text,
                    "done": m.group("done").lower() == "x",
                    "due": due.group(1) if due else None,
                    "note": os.path.relpath(path, vault),
                    "line": i,
                })
    return tasks


# =============================================================================
# app-state pointer (which chat is current)
# =============================================================================
def _meta_path(vault):
    return os.path.join(vault, CACHE_DIR, "_meta.json")


def set_current(cid: str, vault: str = DEFAULT_VAULT):
    ensure_vault(vault)
    _write(_meta_path(vault), json.dumps({"current": cid}))


def get_current(vault: str = DEFAULT_VAULT):
    p = _meta_path(vault)
    return json.loads(_read(p)).get("current") if os.path.exists(p) else None


# =============================================================================
# migration  (state.pkl  ->  vault)
# =============================================================================
def migrate_pickle(pickle_path: str, vault: str = DEFAULT_VAULT) -> dict:
    """One-way import of the legacy pickle into the Markdown vault.
    Returns a summary dict. Non-destructive: the pickle is left untouched."""
    import pickle
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    ensure_vault(vault)
    convs = data.get("conversations") or []
    written, sources_seen = 0, set()
    for c in convs:
        # normalize legacy field names: legacy used "history"; keep as-is
        conv = dict(c)
        conv.setdefault("history", conv.get("messages") or [])
        # a legacy chat carried its own index/pdf inline -> vault splits them out
        write_chat(conv, vault)
        written += 1
        idx = conv.get("index")
        if idx and idx.get("name") and idx["name"] not in sources_seen:
            write_source({"id": slugify(idx["name"]), "name": idx["name"],
                          "pages": idx.get("pages"), "orig_tok": idx.get("orig_tok")},
                         vault)
            sources_seen.add(idx["name"])
    if data.get("current"):
        set_current(data["current"], vault)
    return {"chats": written, "sources": len(sources_seen),
            "current": data.get("current"), "vault": vault}


# =============================================================================
# CLI
# =============================================================================
def _doctor(vault: str):
    chats = list_chats(vault)
    tasks = parse_tasks(vault)
    print(f"Vault: {vault}")
    print(f"  Chats : {len(chats)}")
    for c in chats:
        print(f"    - [{c['id']}] {c['title']}  ({len(c['history'])} msgs)"
              + (f"  doc={c['index']['name']}" if c.get("index") else ""))
    for sub in ("Sources", "Notes", "Daily"):
        d = os.path.join(vault, sub)
        n = len([f for f in os.listdir(d) if f.endswith('.md')]) if os.path.isdir(d) else 0
        print(f"  {sub:7}: {n}")
    print(f"  Tasks : {len(tasks)} "
          f"({sum(not t['done'] for t in tasks)} open)")
    print(f"  Current: {get_current(vault)}")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="FreeLAB vault tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("migrate", help="import state.pkl into the vault")
    m.add_argument("--pickle", default=os.path.join(_HERE, "freelab_data", "state.pkl"))
    m.add_argument("--vault", default=DEFAULT_VAULT)
    d = sub.add_parser("doctor", help="summarize a vault")
    d.add_argument("--vault", default=DEFAULT_VAULT)
    args = p.parse_args(argv)
    if args.cmd == "migrate":
        summary = migrate_pickle(args.pickle, args.vault)
        print("Migrated:", json.dumps(summary, indent=2))
    elif args.cmd == "doctor":
        _doctor(args.vault)


if __name__ == "__main__":
    main()

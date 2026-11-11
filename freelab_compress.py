#!/usr/bin/env python3
"""
FreeLAB compression front-end -- Stage 2: query-aware context compression.

WHY THIS IS THE REAL WIN FOR LARGE TEXT
---------------------------------------
For big documents, cost (and the memory that makes your Mac OOM) is dominated by
PREFILL -- the big model reading the whole input. This front-end does a cheap
step FIRST that keeps only the chunks relevant to the question, so the big model
reads a tiny fraction of the document. On 16 GB that's not just faster -- it's
often what makes the document fit in memory at all.

PIPELINE
  1. Split the document into overlapping word-chunks.
  2. Embed the query and every chunk with a small local embedding model.
  3. Rank chunks by cosine similarity to the query; keep the top-k (in doc order).
  4. Send ONLY those chunks + the question to the big model.

This is *query-aware* compression (the LongLLMLingua / RAG-reranking idea):
relevance is anchored to the actual question, so we're far less likely to throw
away the paragraph that held the answer.

SETUP
  ollama pull nomic-embed-text     # small (~275 MB) embedding model

USAGE
  python3 freelab_compress.py --demo
  python3 freelab_compress.py --file report.txt "What are the payment terms?"
  python3 freelab_compress.py --file report.txt --keep 8 "..."
"""

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"
VERIFIER = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"


def _post(path, payload, timeout=600):
    req = urllib.request.Request(
        OLLAMA + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(inputs, model=EMBED_MODEL):
    """/api/embed takes a list of strings and returns one vector per string."""
    body = _post("/api/embed", {"model": model, "input": inputs})
    return body["embeddings"]


def generate(prompt, model=VERIFIER, think=False, num_predict=None):
    opts = {"temperature": 0.2}
    if num_predict:
        opts["num_predict"] = num_predict
    body = _post("/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        "think": think, "keep_alive": "10m", "options": opts,
    })
    return (body.get("response") or "").strip(), body.get("eval_count") or 0


def chunk_text(text, size=120, overlap=20):
    """Split into ~120-word chunks with 20-word overlap so a fact spanning a
    boundary isn't lost."""
    words = text.split()
    chunks, i = [], 0
    step = size - overlap
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += step
    return chunks


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def approx_tokens(s):
    return max(1, len(s) // 4)   # rough: ~4 chars/token


def compress(query, chunks, keep):
    """Embed query + chunks, rank by similarity, keep top-k in document order."""
    vecs = embed([query] + chunks)
    qv, cvs = vecs[0], vecs[1:]
    scored = sorted(((cosine(qv, cv), i) for i, cv in enumerate(cvs)), reverse=True)
    top = scored[:keep]
    keep_idx = sorted(i for _, i in top)            # restore reading order
    return [chunks[i] for i in keep_idx], top


def answer(query, kept_chunks):
    context = "\n\n".join(f"[chunk {n+1}]\n{c}" for n, c in enumerate(kept_chunks))
    prompt = (
        "Use ONLY the context below to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    return generate(prompt)


# --- needle-in-a-haystack demo -------------------------------------------------
DEMO_NEEDLE = (
    "Note for the record: the internal codename for the 2027 infrastructure "
    "initiative is BLUEHERON, approved unanimously by the steering committee."
)


def build_demo():
    filler = ("The quarterly operations review covered staffing levels, budget "
              "variance, vendor timelines, and routine compliance items, with no "
              "material exceptions noted during the period. ")
    paras = [filler * 3 for _ in range(120)]
    paras.insert(73, DEMO_NEEDLE)          # bury the one fact that matters
    doc = "\n\n".join(paras)
    q = "What is the codename for the 2027 infrastructure initiative?"
    return doc, q


def main():
    ap = argparse.ArgumentParser(description="FreeLAB query-aware compressor.")
    ap.add_argument("query", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--keep", type=int, default=5, help="chunks to keep")
    args = ap.parse_args()

    if args.demo:
        doc, query = build_demo()
    elif args.file and args.query:
        if not os.path.isfile(args.file):
            print(f"No such file: {args.file}\n"
                  "Point --file at a real document, e.g. a .txt or .md on your Mac.")
            return
        with open(args.file, encoding="utf-8", errors="ignore") as f:
            doc = f.read()
        query = args.query
    else:
        ap.error("use --demo, OR --file FILE plus a query string")

    chunks = chunk_text(doc)
    orig_tok = approx_tokens(doc)

    t0 = time.time()
    try:
        kept, top = compress(query, chunks, args.keep)
    except urllib.error.HTTPError:
        print(f"Embedding call failed. Have you run:  ollama pull {EMBED_MODEL}")
        return
    kept_tok = approx_tokens("\n\n".join(kept))
    ans, _ = answer(query, kept)
    dt = time.time() - t0

    print(f"\nquery        : {query}")
    print(f"document     : {len(chunks)} chunks, ~{orig_tok} tokens")
    print(f"kept         : {len(kept)} chunks, ~{kept_tok} tokens "
          f"({100 - 100 * kept_tok // orig_tok}% smaller)")
    print(f"top scores   : {[round(s, 3) for s, _ in top]}")
    print(f"time         : {dt:.1f}s")
    print(f"\nANSWER:\n{ans}")
    print("\n(The big model never saw the other "
          f"{len(chunks) - len(kept)} chunks -- that's the compression.)")


if __name__ == "__main__":
    main()

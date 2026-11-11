#!/usr/bin/env python3
"""
FreeLAB CLI -- fast, free, local AI on a Mac.

Thin command-line front-end over the shared pipeline in `freelab_core.py`.
All the real logic (compress -> draft -> judge -> answer) lives in the core
module so the CLI, the eval harness, and the Gradio app stay in sync.

MODES
  (default)        compress + cascade  -- the full pipeline
  --no-cascade     compress, then big model answers directly (no draft/judge)
  --no-compress    NO compression: big model reads the FULL document (baseline)

SETUP
  ollama pull qwen3:0.6b
  ollama pull qwen3:8b
  ollama pull nomic-embed-text

USAGE
  python3 freelab.py --demo
  python3 freelab.py --file sample_doc.md "How many planets are in the solar system?"
  python3 freelab.py --file big.txt --no-compress "..."   # see the slow baseline
"""

import argparse
import os
import urllib.error

import freelab_core as fl


def main():
    ap = argparse.ArgumentParser(description="FreeLAB local AI pipeline.")
    ap.add_argument("query", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--keep", type=int, default=fl.DEFAULT_KEEP)
    ap.add_argument("--threshold", type=int, default=fl.DEFAULT_THRESHOLD)
    ap.add_argument("--no-compress", action="store_true")
    ap.add_argument("--no-cascade", action="store_true")
    args = ap.parse_args()

    if args.demo:
        doc, query = fl.build_demo()
    elif args.file and args.query:
        if not os.path.isfile(args.file):
            print(f"No such file: {args.file}")
            return
        with open(args.file, encoding="utf-8", errors="ignore") as f:
            doc = f.read()
        query = args.query
    else:
        ap.error("use --demo, OR --file FILE plus a query string")

    try:
        fl.warmup(not args.no_compress, not args.no_cascade,
                  log=lambda m: print(m))
        final, s = fl.pipeline(query, doc, args.keep, args.threshold,
                               use_compress=not args.no_compress,
                               use_cascade=not args.no_cascade)
    except urllib.error.HTTPError:
        print(f"A model call failed. Have you pulled: "
              f"{fl.DRAFTER}, {fl.VERIFIER}, {fl.EMBED_MODEL}?")
        return

    d, j, g = s["tokens"]
    print(f"\nquery      : {query}")
    if args.no_compress:
        print(f"compression: OFF  (big model reads all {s['orig_chunks']} chunks, "
              f"~{s['orig_tok']} tokens)")
    else:
        pct = 100 - 100 * s["kept_tok"] // s["orig_tok"]
        print(f"compression: {s['orig_chunks']}->{s['kept_chunks']} chunks, "
              f"~{s['orig_tok']}->~{s['kept_tok']} tokens ({pct}% smaller)")
        print(f"top scores : {s['top_scores']}")
    print(f"path       : {s['path']}" + (f"  (judge score {s['score']})"
          if s["score"] is not None else ""))
    print(f"tokens     : draft={d} judge={j} biggen={g}")
    print(f"time       : {s['time_s']:.1f}s")
    print(f"\nANSWER:\n{final}")


if __name__ == "__main__":
    main()

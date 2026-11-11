#!/usr/bin/env python3
"""
FreeLAB cascade harness -- Stage 1b: JUDGE-gated cascade.

THE IDEA
--------
1. A small fast model (the DRAFTER) answers the query.
2. The big model (the VERIFIER) does NOT regenerate -- it just JUDGES the draft,
   emitting a tiny verdict like "SCORE: 85". Judging is a few output tokens;
   generating is hundreds. Decode is the expensive part on a Mac, so judging is
   ~20x cheaper than answering. (This is "verification is cheaper than
   generation" -- from the FreeLAB plan -- applied at the answer level.)
3. If the score is high, we keep the draft (we paid: small gen + cheap judge).
   If it's low, we ESCALATE: the big model generates the full answer.

Why a big-model judge and not self-rating? A 0.6B model rates itself 100 on
everything, including garbage. A capable model is a far better-calibrated judge.

THE CATCH (watch for it in the benchmark)
-----------------------------------------
The big model runs a judge pass on EVERY query. This only wins if (a) judging is
much cheaper than answering -- so we cap judge output and turn thinking off --
and (b) most drafts get accepted, keeping full generation rare. If the judge
rejects everything, you pay judge + generate and it's SLOWER than the big model
alone. The token counts in the summary tell you which regime you're in.

No external deps -- standard library talking to local Ollama over HTTP.

USAGE
-----
  python3 freelab_cascade.py "What is the capital of France?"
  python3 freelab_cascade.py --benchmark
  python3 freelab_cascade.py --threshold 60 --benchmark      # accept more drafts
  python3 freelab_cascade.py --gate selfrated --benchmark    # the old, broken gate, for contrast
"""

import argparse
import json
import re
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"

DEFAULT_DRAFTER = "qwen3:0.6b"
DEFAULT_VERIFIER = "qwen3:8b"
DEFAULT_THRESHOLD = 70   # judge score >= this -> keep draft; else escalate
JUDGE_MAX_TOKENS = 16    # cap the judge's output so it stays cheap


def ollama_generate(model, prompt, think=False, num_predict=None, timeout=600):
    """One non-streaming Ollama call. Returns (text, tokens_per_sec, n_tokens).

    think=False keeps qwen3 from emitting a long <think> block -- essential for
    the judge, whose whole value is being cheap. num_predict caps output length.
    keep_alive holds the model in RAM so we don't pay reload cost every call.
    """
    options = {"temperature": 0.2}
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload = {"model": model, "prompt": prompt, "stream": False,
               "think": think, "keep_alive": "10m", "options": options}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    text = (body.get("response") or "").strip()
    ec = body.get("eval_count") or 0            # tokens generated (decode phase)
    ed = body.get("eval_duration") or 0         # nanoseconds
    tps = (ec / (ed / 1e9)) if ed else 0.0
    return text, tps, ec


DRAFT_MAX_TOKENS = 350  # bound the worst case so a hard prompt can't hang the 0.6B


def draft(drafter, query):
    """Small model answers. No self-rating -- the judge decides."""
    return ollama_generate(drafter, query, num_predict=DRAFT_MAX_TOKENS)


def warmup(*models):
    """Load each model into RAM once so cold-start isn't counted in timings."""
    print("warming up models...", end=" ", flush=True)
    for m in models:
        ollama_generate(m, "hi", num_predict=1)
    print("done.")


SCORE_RE = re.compile(r"SCORE:\s*(\d{1,3})", re.IGNORECASE)


def judge(verifier, query, draft_answer):
    """Big model grades the draft and emits ONLY a score. This is the cheap step.

    Capped to JUDGE_MAX_TOKENS output. If no score is parsed, we fail safe to 0
    (escalate) so an unparseable judge never silently keeps a bad draft.
    """
    prompt = (
        "You are grading a draft answer. Reply with ONLY 'SCORE: <integer 0-100>' "
        "for how correct and complete the draft is. Do not explain or rewrite.\n\n"
        f"Question:\n{query}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        "SCORE: "
    )
    text, tps, ntok = ollama_generate(verifier, prompt, num_predict=JUDGE_MAX_TOKENS)
    m = SCORE_RE.search(text)
    score = int(m.group(1)) if m else 0
    return score, ntok


def generate_full(verifier, query, draft_answer):
    """Escalation path: big model writes the real answer, correcting the draft."""
    prompt = (
        f"Question:\n{query}\n\n"
        f"A draft answer from a smaller model (judged low-quality):\n{draft_answer}\n\n"
        "Provide the best, correct, complete answer."
    )
    return ollama_generate(verifier, prompt)


def run(query, drafter, verifier, threshold, gate="judge", verbose=True):
    t0 = time.time()
    draft_answer, _, draft_tok = draft(drafter, query)
    judge_tok = gen_tok = 0

    if gate == "judge":
        score, judge_tok = judge(verifier, query, draft_answer)
        if score >= threshold:
            path, final = "DRAFTER", draft_answer
        else:
            final, _, gen_tok = generate_full(verifier, query, draft_answer)
            path = "VERIFIER"
    else:  # "selfrated" -- kept only to contrast with the broken approach
        score = 100  # (the drafter would just say 100; not re-implemented here)
        path, final = "DRAFTER", draft_answer

    total = time.time() - t0
    if verbose:
        print(f"\n[path: {path}  judge_score: {score}  "
              f"tokens(draft/judge/gen): {draft_tok}/{judge_tok}/{gen_tok}  "
              f"total: {total:.1f}s]\n")
        print(final)
    return {"query": query, "path": path, "score": score, "total_s": total,
            "draft_tok": draft_tok, "judge_tok": judge_tok, "gen_tok": gen_tok}


BENCHMARK = [
    "What is the capital of France?",
    "What is 17 + 25?",
    "Who wrote Romeo and Juliet?",
    "Convert 100 Fahrenheit to Celsius.",
    "Prove that the square root of 2 is irrational.",
    "Explain why the sky is blue, including the physics.",
    "Write a Python function to detect a cycle in a linked list.",
    "What are the main tradeoffs between REST and gRPC?",
]


def benchmark(drafter, verifier, threshold, gate):
    warmup(drafter, verifier)
    rows = []
    for q in BENCHMARK:
        print(f"\n=== {q}")
        rows.append(run(q, drafter, verifier, threshold, gate))

    n = len(rows)
    fast = [r for r in rows if r["path"] == "DRAFTER"]
    slow = [r for r in rows if r["path"] == "VERIFIER"]
    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"queries                 : {n}")
    print(f"kept draft (fast)       : {len(fast)}  ({100*len(fast)//n}%)")
    print(f"escalated (full big-gen): {len(slow)}")
    print(f"avg latency kept        : {avg([r['total_s'] for r in fast]):.1f}s")
    print(f"avg latency escalated   : {avg([r['total_s'] for r in slow]):.1f}s")
    print(f"avg latency overall     : {avg([r['total_s'] for r in rows]):.1f}s")
    print("-" * 62)
    print("EFFICIENCY (the heart of your idea -- judging vs generating):")
    print(f"  avg judge tokens      : {avg([r['judge_tok'] for r in rows]):.0f}")
    print(f"  avg full-gen tokens   : {avg([r['gen_tok'] for r in slow]):.0f}"
          f"  (on escalated queries)")
    print("  -> judge tokens should be MUCH smaller. That gap is why this can win.")


def main():
    ap = argparse.ArgumentParser(description="FreeLAB judge-gated cascade.")
    ap.add_argument("query", nargs="?")
    ap.add_argument("--drafter", default=DEFAULT_DRAFTER)
    ap.add_argument("--verifier", default=DEFAULT_VERIFIER)
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    ap.add_argument("--gate", choices=["judge", "selfrated"], default="judge")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    if args.benchmark:
        benchmark(args.drafter, args.verifier, args.threshold, args.gate)
    elif args.query:
        run(args.query, args.drafter, args.verifier, args.threshold, args.gate)
    else:
        ap.error("give a query, or use --benchmark")


if __name__ == "__main__":
    main()

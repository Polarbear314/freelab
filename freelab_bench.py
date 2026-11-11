#!/usr/bin/env python3
"""
FreeLAB benchmark -- FreeLAB vs a plain locally-hosted LLM, in numbers.

Answers the question "how much faster is the cascade+compression pipeline than
just running the big model directly on my Mac -- and does it stay accurate?"

Two head-to-head tests, each comparing:
    BASELINE   the big model (qwen3:8b) answering directly -- i.e. a "normal"
               local LLM with no compression and no draft/judge cascade.
    FREELAB    the full pipeline (compress -> draft -> judge -> escalate).

  (A) CHAT  -- 7 trick questions a small model tends to blow. Baseline = 8B
              direct. FreeLAB = draft/judge cascade. Shows the cascade keeps the
              big model's accuracy while the cheap drafter handles the easy ones.

  (B) DOC   -- a large document with facts buried in it. Baseline feeds the WHOLE
              document to the 8B (the normal way); FreeLAB compresses to the few
              relevant chunks first. This is where the speedup is biggest.

Both baseline and FreeLAB go through `freelab_core.pipeline()` for the doc test
(baseline just sets use_compress=False, use_cascade=False), so the comparison is
apples-to-apples on the same code path.

Run from the FreeLAB folder with Ollama running:
    python3 freelab_bench.py                 # both tests, default sizes
    python3 freelab_bench.py --chat          # trick-question test only
    python3 freelab_bench.py --doc --paras 400   # bigger haystack
    python3 freelab_bench.py --repeats 3     # median of 3 runs per query

Writes a Markdown report + CSV next to the script (--out to change).
"""

import argparse
import csv
import datetime
import os
import statistics
import time

import freelab_core as fl
from freelab_eval import HARD, NEEDLES, build_haystack, correct

HERE = os.path.dirname(os.path.abspath(__file__))


def _timed(fn):
    """Run fn(), return (result, wall_seconds)."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


# ============================================================ (A) CHAT trick Qs
def bench_chat(repeats):
    print("\n########## (A) CHAT  —  baseline 8B direct vs FreeLAB cascade ##########")
    fl.warmup(use_compress=False, use_cascade=True)
    base, free = [], []
    for item in HARD:
        q, exp = item["q"], item["expect"]

        bt, btok, bok, btext = [], 0, False, ""
        for _ in range(repeats):
            (text, tok), dt = _timed(
                lambda: fl.generate(fl._chat_prompt(q, concise=False), fl.VERIFIER))
            bt.append(dt); btok = tok; bok = correct(text, exp); btext = text

        ft, fok, fpath, fscore, fgen = [], False, "", None, 0
        for _ in range(repeats):
            (final, path, score, toks), dt = _timed(lambda: fl.chat_answer(q))
            ft.append(dt); fok = correct(final, exp); fpath = path
            fscore = score; fgen = toks[2]

        base.append({"q": q, "ok": bok, "t": _median(bt), "big_tok": btok})
        free.append({"q": q, "ok": fok, "t": _median(ft), "path": fpath,
                     "score": fscore, "big_tok": fgen})
        print(f"\nQ: {q}")
        print(f"   baseline 8B : {'OK ' if bok else 'WRONG'}  {_median(bt):5.1f}s")
        print(f"   FreeLAB     : {'OK ' if fok else 'WRONG'}  {_median(ft):5.1f}s  "
              f"[{fpath}, judge {fscore}]")
    return base, free


# ============================================================ (B) DOC needles
def bench_doc(paras, keep, threshold, repeats):
    print("\n########## (B) DOC  —  baseline 8B reads whole doc vs FreeLAB compresses ##########")
    doc = build_haystack(paras)
    approx = fl.approx_tokens(doc)
    print(f"document: ~{approx} tokens, {len(fl.chunk_text(doc))} chunks, "
          f"{len(NEEDLES)} facts buried")
    fl.warmup(use_compress=True, use_cascade=True)
    base, free = [], []
    for n in NEEDLES:
        q, exp = n["q"], n["expect"]

        # BASELINE: whole document into the 8B, no cascade
        bt, bok, bin_tok, berr = [], False, approx, None
        for _ in range(repeats):
            try:
                (res), dt = _timed(lambda: fl.pipeline(
                    q, doc, keep, threshold, use_compress=False, use_cascade=False))
                final, s = res
                bt.append(dt); bok = correct(final, exp); bin_tok = s["orig_tok"]
            except Exception as e:  # noqa: BLE001 -- baseline may OOM on a big doc
                berr = str(e); bt.append(None)

        # FREELAB: compress then cascade
        ft, fok, fin_tok, fpath, fpct = [], False, 0, "", 0
        for _ in range(repeats):
            (res), dt = _timed(lambda: fl.pipeline(
                q, doc, keep, threshold, use_compress=True, use_cascade=True))
            final, s = res
            ft.append(dt); fok = correct(final, exp); fin_tok = s["kept_tok"]
            fpath = s["path"]; fpct = 100 - 100 * s["kept_tok"] // max(s["orig_tok"], 1)

        base.append({"q": q, "ok": bok, "t": _median(bt), "in_tok": bin_tok, "err": berr})
        free.append({"q": q, "ok": fok, "t": _median(ft), "in_tok": fin_tok,
                     "path": fpath, "pct": fpct})
        bt_s = f"{_median(bt):5.1f}s" if _median(bt) is not None else "FAILED"
        print(f"\nQ: {q}")
        print(f"   baseline 8B (full doc): {'OK ' if bok else 'WRONG'}  {bt_s}  "
              f"(~{bin_tok} tok in)" + (f"  [{berr}]" if berr else ""))
        print(f"   FreeLAB (compressed)  : {'OK ' if fok else 'WRONG'}  "
              f"{_median(ft):5.1f}s  (~{fin_tok} tok in, {fpct}% smaller, {fpath})")
    return doc, base, free


# ============================================================ reporting
def _agg_time(rows):
    ts = [r["t"] for r in rows if r["t"] is not None]
    return (sum(ts), statistics.mean(ts)) if ts else (None, None)


def _speedup(b, f):
    return f"{b / f:.1f}×" if (b and f) else "—"


def build_report(chat, doc, args):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"# FreeLAB Benchmark — {now}",
         "",
         f"Baseline = **{fl.VERIFIER}** answering directly (a normal local LLM). "
         f"FreeLAB = compress → draft ({fl.DRAFTER}) → judge → escalate. "
         f"Median of {args.repeats} run(s) per query on this machine.",
         ""]

    if chat:
        base, free = chat
        bt_tot, bt_avg = _agg_time(base)
        ft_tot, ft_avg = _agg_time(free)
        n = len(base)
        esc = sum(1 for r in free if r["path"] == "ESCALATED")
        b_big = sum(r["big_tok"] for r in base)
        f_big = sum(r["big_tok"] for r in free)
        L += ["## (A) Chat — 7 trick questions", "",
              "| Method | Accuracy | Total time | Avg / question | Big-model tokens |",
              "|---|---|---|---|---|",
              f"| Baseline {fl.VERIFIER} | {sum(r['ok'] for r in base)}/{n} | "
              f"{bt_tot:.1f}s | {bt_avg:.1f}s | {b_big} |",
              f"| **FreeLAB** | {sum(r['ok'] for r in free)}/{n} | "
              f"{ft_tot:.1f}s | {ft_avg:.1f}s | {f_big} |",
              "",
              f"- **Speedup: {_speedup(bt_avg, ft_avg)}** (avg per question)",
              f"- Escalated to the big model: {esc}/{n} "
              f"(the other {n-esc} answered by the cheap drafter)",
              f"- Big-model tokens generated: {b_big} → {f_big} "
              f"({100 - 100*f_big//max(b_big,1)}% fewer)",
              ""]

    if doc:
        docstr, base, free = doc
        approx = fl.approx_tokens(docstr)
        bt_tot, bt_avg = _agg_time(base)
        ft_tot, ft_avg = _agg_time(free)
        n = len(base)
        pct = statistics.mean(r["pct"] for r in free) if free else 0
        b_in = statistics.mean(r["in_tok"] for r in base) if base else 0
        f_in = statistics.mean(r["in_tok"] for r in free) if free else 0
        b_fail = sum(1 for r in base if r["t"] is None)
        L += [f"## (B) Document Q&A — {len(NEEDLES)} facts in a ~{approx}-token doc", "",
              "| Method | Accuracy | Avg latency | Input tokens → big model |",
              "|---|---|---|---|",
              f"| Baseline {fl.VERIFIER} (whole doc) | {sum(r['ok'] for r in base)}/{n} | "
              f"{bt_avg:.1f}s" + (f" ({b_fail} failed)" if b_fail else "") + f" | ~{int(b_in)} |",
              f"| **FreeLAB** (compressed) | {sum(r['ok'] for r in free)}/{n} | "
              f"{ft_avg:.1f}s | ~{int(f_in)} |",
              "",
              f"- **Speedup: {_speedup(bt_avg, ft_avg)}** (avg latency)",
              f"- Average compression: **{pct:.0f}%** smaller input "
              f"(~{int(b_in)} → ~{int(f_in)} tokens)",
              ""]

    L += ["---",
          "*Generated by `freelab_bench.py`. Speed depends on the machine; "
          "accuracy on this fixed question set. See [[FreeLAB Build Log]].*"]
    return "\n".join(L)


def write_csv(path, chat, doc):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "question", "method", "correct", "seconds",
                    "big_model_tokens", "path", "extra"])
        if chat:
            base, free = chat
            for r in base:
                w.writerow(["chat", r["q"], "baseline", r["ok"], r["t"], r["big_tok"], "", ""])
            for r in free:
                w.writerow(["chat", r["q"], "freelab", r["ok"], r["t"], r["big_tok"],
                            r["path"], f"judge={r['score']}"])
        if doc:
            _, base, free = doc
            for r in base:
                w.writerow(["doc", r["q"], "baseline", r["ok"], r["t"], "", "",
                            f"in_tok={r['in_tok']}"])
            for r in free:
                w.writerow(["doc", r["q"], "freelab", r["ok"], r["t"], "", r["path"],
                            f"in_tok={r['in_tok']},compressed={r['pct']}%"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="FreeLAB vs plain local LLM benchmark.")
    ap.add_argument("--chat", action="store_true", help="run the chat test only")
    ap.add_argument("--doc", action="store_true", help="run the document test only")
    ap.add_argument("--paras", type=int, default=200, help="filler paragraphs (doc size)")
    ap.add_argument("--keep", type=int, default=fl.DEFAULT_KEEP)
    ap.add_argument("--threshold", type=int, default=fl.DEFAULT_THRESHOLD)
    ap.add_argument("--repeats", type=int, default=1, help="runs per query (median)")
    ap.add_argument("--out", default=HERE, help="folder for the report + CSV")
    args = ap.parse_args(argv)

    run_both = not (args.chat or args.doc)
    chat = bench_chat(args.repeats) if (args.chat or run_both) else None
    doc = bench_doc(args.paras, args.keep, args.threshold, args.repeats) \
        if (args.doc or run_both) else None

    report = build_report(chat, doc, args)
    print("\n" + "=" * 70 + "\n" + report + "\n" + "=" * 70)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    md_path = os.path.join(args.out, f"FreeLAB Benchmark {stamp}.md")
    csv_path = os.path.join(args.out, f"freelab_bench_{stamp}.csv")
    with open(md_path, "w") as f:
        f.write(report)
    write_csv(csv_path, chat, doc)
    print(f"\nSaved:\n  {md_path}\n  {csv_path}")


if __name__ == "__main__":
    main()

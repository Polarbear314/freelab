#!/usr/bin/env python3
"""
FreeLAB threshold sweep -- find the input size where the wrapped pipeline starts
to beat the raw 8B, so the `direct_below` cutoff can be set from data, not a guess.

Method: take ONE question with a known answer, pad its context with filler to a
ladder of token sizes, and at each size run BOTH:
    direct   = whole context -> 8B directly (no compression, no cascade)  [the raw 8B]
    wrapped  = compress -> draft -> judge -> escalate (full FreeLAB)
Because only the context *length* changes, the two curves isolate the effect of
size on accuracy and latency. Two crossovers emerge:
    * ACCURACY crossover -- where the raw 8B starts failing (context overflow /
      lost-in-the-middle) while the wrapped version stays correct. This is the
      important one: set `direct_below` just under it.
    * LATENCY crossover  -- where the wrapped version stops being slower.

Run from the FreeLAB folder with Ollama running:
    python3 freelab_threshold_sweep.py
    python3 freelab_threshold_sweep.py --sizes 128,256,512,1024,2048,4096,8192 --repeats 2

Prints a per-size table and a recommended `direct_below` (tokens).

NOTE on latency: `pipeline()` re-embeds the whole context each call, so the
wrapped latency here is an upper bound; the deployed app embeds once at ingest.
The ACCURACY crossover is unaffected by this and is the primary signal.
"""

import argparse
import statistics
import time

import freelab_core as fl
from freelab_eval import correct

# the buried fact + its question (known answer for objective grading)
NEEDLE = ("Operational note: the maximum rated voltage for the Orion relay module "
          "is 480 volts; exceeding it trips the safety interlock.")
QUESTION = "What is the maximum rated voltage for the Orion relay module?"
EXPECT = ["480"]

_FILLER = ("The quarterly operations review covered staffing levels, budget variance, "
           "vendor timelines, and routine compliance items, with no material exceptions "
           "noted during the reporting period. ")


def build_context(target_tokens):
    """Filler padded to ~target_tokens (approx_tokens = chars//4), needle in the middle."""
    target_chars = target_tokens * 4
    half = _FILLER * max(1, target_chars // (2 * len(_FILLER)))
    ctx = half + "\n\n" + NEEDLE + "\n\n" + half
    return ctx


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run_size(size, keep, threshold, repeats):
    ctx = build_context(size)
    actual = fl.approx_tokens(ctx)
    d_ok, d_lat, w_ok, w_lat, kept_tok = [], [], [], [], []
    for _ in range(repeats):
        # direct: whole context -> 8B, no compression, no cascade
        (res), dt = _timed(lambda: fl.pipeline(QUESTION, ctx, keep, threshold,
                                               use_compress=False, use_cascade=False))
        final, _ = res
        d_ok.append(float(correct(final, EXPECT))); d_lat.append(dt)
        # wrapped: full FreeLAB
        (res), wt = _timed(lambda: fl.pipeline(QUESTION, ctx, keep, threshold,
                                              use_compress=True, use_cascade=True))
        final, s = res
        w_ok.append(float(correct(final, EXPECT))); w_lat.append(wt)
        kept_tok.append(s["kept_tok"])
    return {"size": actual,
            "d_acc": statistics.mean(d_ok), "w_acc": statistics.mean(w_ok),
            "d_lat": statistics.median(d_lat), "w_lat": statistics.median(w_lat),
            "kept": int(statistics.mean(kept_tok))}


def recommend(rows):
    rows = sorted(rows, key=lambda r: r["size"])
    # accuracy crossover: first size where wrapped clearly beats direct
    acc_cross = next((r["size"] for r in rows if r["w_acc"] - r["d_acc"] >= 0.5), None)
    # largest size where direct is still as good as wrapped (safe to stay direct)
    safe_direct = [r["size"] for r in rows if r["d_acc"] >= r["w_acc"] - 1e-9]
    last_safe = max(safe_direct) if safe_direct else None
    # latency crossover: first size where wrapped is no slower than direct
    lat_cross = next((r["size"] for r in rows if r["w_lat"] <= r["d_lat"]), None)
    return acc_cross, last_safe, lat_cross


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find the direct-vs-cascade token threshold.")
    ap.add_argument("--sizes", default="128,256,512,1024,2048,4096,8192",
                    help="comma-separated target context sizes in tokens")
    ap.add_argument("--keep", type=int, default=fl.DEFAULT_KEEP)
    ap.add_argument("--threshold", type=int, default=fl.DEFAULT_THRESHOLD)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args(argv)

    sizes = [int(x) for x in args.sizes.split(",")]
    fl.warmup(use_compress=True, use_cascade=True)
    print(f"\nThreshold sweep — direct 8B vs wrapped FreeLAB, {len(sizes)} sizes, "
          f"{args.repeats} repeat(s)\n")
    print(f"{'ctx tok':>8} | {'direct acc':>10} | {'wrap acc':>9} | "
          f"{'direct lat':>10} | {'wrap lat':>9} | {'wrap input':>10}")
    print("-" * 70)
    rows = []
    for sz in sizes:
        r = run_size(sz, args.keep, args.threshold, args.repeats)
        rows.append(r)
        print(f"{r['size']:>8} | {r['d_acc']:>10.2f} | {r['w_acc']:>9.2f} | "
              f"{r['d_lat']:>9.1f}s | {r['w_lat']:>8.1f}s | {r['kept']:>8} tok")

    acc_cross, last_safe, lat_cross = recommend(rows)
    print("\n----- RECOMMENDATION -----")
    print(f"Accuracy crossover (direct starts failing): "
          f"{acc_cross if acc_cross else 'not reached in range'}")
    print(f"Latency crossover (wrapped no longer slower): "
          f"{lat_cross if lat_cross else 'not reached in range'}")
    if last_safe is not None and acc_cross:
        rec = last_safe
        print(f"\n>>> Suggested direct_below ~= {rec} tokens")
        print(f"    (stay on the fast raw 8B up to ~{rec} tok, switch to the wrapped "
              f"pipeline at/above ~{acc_cross} tok, where the raw 8B begins to fail).")
    elif acc_cross is None:
        print("\n>>> Raw 8B never failed in this range — either raise the sizes, or the "
              "direct path is safe across all tested sizes (set direct_below high).")
    print("\nCaveat: wrapped latency is inflated by per-query re-embedding; the app "
          "embeds once at ingest. Accuracy crossover is the reliable signal.")


if __name__ == "__main__":
    main()

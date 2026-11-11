#!/usr/bin/env python3
"""
FreeLAB code benchmark -- FreeLAB vs a plain local LLM on code tasks.

Two experiments over a synthetic codebase:

  (A) CODE RETRIEVAL Q&A  -- a large generated repo with specific facts buried in
      deep modules. BASELINE feeds the whole repo to the 8B (truncates when it
      overflows the context window); FREELAB compresses to the few relevant chunks
      first. Mirrors the document benchmark, but over source code.

  (B) CODE GENERATION with a TEST-AS-JUDGE cascade  -- the key idea for code: the
      judge does NOT have to be an LLM. Each task ships a hidden test suite, so
      correctness is ground truth. BASELINE = the 8B writes the function directly,
      then we run the tests. FREELAB = the 0.6B drafts, we run the tests; if they
      pass we keep it (cheap path, big model never runs); if they fail we escalate
      to the 8B and run the tests again. This is a stronger, cheaper judge than any
      model-scored verdict.

Run from the FreeLAB folder with Ollama running:
    python3 freelab_code_bench.py                # both
    python3 freelab_code_bench.py --retrieval    # part A only
    python3 freelab_code_bench.py --gen          # part B only
    python3 freelab_code_bench.py --modules 400  # bigger synthetic repo

SAFETY: part (B) executes model-generated code. Each candidate runs in a separate
Python subprocess with a timeout, but only run this on synthetic tasks you trust.
"""

import argparse
import csv
import datetime
import os
import subprocess
import sys
import tempfile
import textwrap
import time

import freelab_core as fl
from freelab_eval import correct

HERE = os.path.dirname(os.path.abspath(__file__))


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


# ============================================================ synthetic repo (A)
FILLER_FUNCS = [
    "def normalize_record(r):\n    return {k: str(v).strip() for k, v in r.items()}",
    "def batch(items, n):\n    return [items[i:i+n] for i in range(0, len(items), n)]",
    "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
    "def dedupe(seq):\n    seen = set(); out = []\n    for x in seq:\n        if x not in seen:\n            seen.add(x); out.append(x)\n    return out",
    "def percent(part, whole):\n    return 0 if not whole else round(100 * part / whole, 2)",
]

# facts buried in the repo; each is a gradable retrieval question
CODE_NEEDLES = [
    {"code": "ORION_MAX_VOLTAGE = 480  # tripping the interlock above this is fatal",
     "q": "What is the value of the ORION_MAX_VOLTAGE constant in the codebase?",
     "expect": ["480"]},
    {"code": "def falcon_launch_date():\n    # rescheduled after design review\n    return \"2029-03-14\"",
     "q": "What date does the function falcon_launch_date() return?",
     "expect": ["2029-03-14", "march 14", "2029"]},
    {"code": "def west_facility_passphrase():\n    # never share externally\n    return \"DELTA-SEVEN-SIGMA\"",
     "q": "What string does west_facility_passphrase() return?",
     "expect": ["delta-seven-sigma", "delta"]},
]


def build_repo(modules):
    """Return one big string simulating a repo: many modules of filler functions,
    with the needle definitions inserted at spread-out positions."""
    files = []
    step = max(modules // (len(CODE_NEEDLES) + 1), 1)
    needle_at = {(i + 1) * step: n for i, n in enumerate(CODE_NEEDLES)}
    for m in range(modules):
        body = [f"# module {m}", "import math, os"]
        for k, fn in enumerate(FILLER_FUNCS):
            body.append(fn.replace("(", f"_{m}_{k}(", 1))  # unique names
        if m in needle_at:
            body.append(needle_at[m]["code"])
        files.append(f"### file: pkg/mod_{m:04d}.py\n" + "\n\n".join(body))
    return "\n\n".join(files)


def bench_retrieval(modules, keep, threshold):
    print("\n########## (A) CODE RETRIEVAL — baseline whole-repo vs FreeLAB compressed ##########")
    repo = build_repo(modules)
    approx = fl.approx_tokens(repo)
    print(f"repo: ~{approx} tokens, {len(fl.chunk_text(repo))} chunks, "
          f"{len(CODE_NEEDLES)} facts buried")
    fl.warmup(use_compress=True, use_cascade=True)
    base, free = [], []
    for n in CODE_NEEDLES:
        q, exp = n["q"], n["expect"]
        try:
            (res), dt = _timed(lambda: fl.pipeline(q, repo, keep, threshold,
                                                   use_compress=False, use_cascade=False))
            final, s = res
            base.append({"q": q, "ok": correct(final, exp), "t": dt, "in_tok": s["orig_tok"]})
        except Exception as e:  # noqa: BLE001
            base.append({"q": q, "ok": False, "t": None, "in_tok": approx, "err": str(e)})
        (res), dt = _timed(lambda: fl.pipeline(q, repo, keep, threshold,
                                               use_compress=True, use_cascade=True))
        final, s = res
        free.append({"q": q, "ok": correct(final, exp), "t": dt, "in_tok": s["kept_tok"],
                     "pct": 100 - 100 * s["kept_tok"] // max(s["orig_tok"], 1)})
        bt = f"{base[-1]['t']:.1f}s" if base[-1]["t"] is not None else "FAILED"
        print(f"\nQ: {q}")
        print(f"   baseline 8B (whole repo): {'OK ' if base[-1]['ok'] else 'WRONG'}  {bt}")
        print(f"   FreeLAB (compressed)    : {'OK ' if free[-1]['ok'] else 'WRONG'}  "
              f"{free[-1]['t']:.1f}s  ({free[-1]['pct']}% smaller)")
    return repo, base, free


# ============================================================ code generation (B)
GEN_TASKS = [
    {"name": "two_sum",
     "spec": "Write a Python function `two_sum(nums, target)` returning a list [i, j] "
             "with i < j and nums[i] + nums[j] == target. Exactly one solution exists.",
     "entry": "two_sum",
     "test": "assert two_sum([2,7,11,15],9)==[0,1]\n"
             "assert two_sum([3,2,4],6)==[1,2]\nassert two_sum([3,3],6)==[0,1]"},
    {"name": "is_palindrome",
     "spec": "Write `is_palindrome(s)` returning True if s is a palindrome considering "
             "only alphanumeric characters and ignoring case, else False.",
     "entry": "is_palindrome",
     "test": "assert is_palindrome('A man, a plan, a canal: Panama')==True\n"
             "assert is_palindrome('race a car')==False\nassert is_palindrome('')==True"},
    {"name": "roman_to_int",
     "spec": "Write `roman_to_int(s)` converting a Roman numeral (I,V,X,L,C,D,M with "
             "subtractive notation) to an integer.",
     "entry": "roman_to_int",
     "test": "assert roman_to_int('III')==3\nassert roman_to_int('LVIII')==58\n"
             "assert roman_to_int('MCMXCIV')==1994"},
    {"name": "flatten",
     "spec": "Write `flatten(xs)` that flattens a list one level deep (only top-level "
             "sublists expanded), returning a new list.",
     "entry": "flatten",
     "test": "assert flatten([1,[2,3],4])==[1,2,3,4]\n"
             "assert flatten([[1],[2],[3]])==[1,2,3]\nassert flatten([1,2,3])==[1,2,3]"},
    {"name": "word_count",
     "spec": "Write `word_count(text)` returning a dict mapping each lowercased "
             "whitespace-split word to its count.",
     "entry": "word_count",
     "test": "assert word_count('a a b')=={'a':2,'b':1}\n"
             "assert word_count('Hi hi HELLO')=={'hi':2,'hello':1}"},
]


def _prompt(spec):
    return (f"{spec}\n\nReturn ONLY the complete Python function definition, no "
            f"explanation, no example usage, no markdown fences.")


def extract_code(text, entry):
    """Pull a runnable function definition out of a model response."""
    text = fl._clean(text)
    if "```" in text:                       # take the largest fenced block
        parts = text.split("```")
        blocks = [parts[i] for i in range(1, len(parts), 2)]
        if blocks:
            text = max(blocks, key=len)
            if text.lstrip().startswith(("python", "py")):
                text = text.split("\n", 1)[1] if "\n" in text else text
    idx = text.find(f"def {entry}")
    return text[idx:] if idx != -1 else text


def run_tests(code, entry, test, timeout=10):
    """Exec candidate code + its tests in an isolated subprocess. Returns True/False."""
    script = f"{code}\n\n# --- tests ---\n{test}\nprint('ALL_PASS')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        return "ALL_PASS" in r.stdout
    except Exception:  # noqa: BLE001 -- timeout / crash counts as fail
        return False
    finally:
        os.unlink(path)


def bench_gen(repeats):
    print("\n########## (B) CODE GENERATION — test suite as ground-truth judge ##########")
    fl.warmup(use_compress=False, use_cascade=True)
    base, free = [], []
    for task in GEN_TASKS:
        entry, test = task["entry"], task["test"]

        # BASELINE: big model writes it directly
        (btext, _), bdt = _timed(lambda: fl.generate(_prompt(task["spec"]), fl.VERIFIER))
        bok = run_tests(extract_code(btext, entry), entry, test)

        # FREELAB: drafter first; tests judge; escalate only on failure
        (dtext, _), fdt0 = _timed(lambda: fl.generate(_prompt(task["spec"]), fl.DRAFTER))
        draft_ok = run_tests(extract_code(dtext, entry), entry, test)
        if draft_ok:
            fok, path, fdt = True, "DRAFT-KEPT", fdt0
        else:
            (etext, _), edt = _timed(lambda: fl.generate(_prompt(task["spec"]), fl.VERIFIER))
            fok = run_tests(extract_code(etext, entry), entry, test)
            path, fdt = "ESCALATED", fdt0 + edt

        base.append({"name": task["name"], "ok": bok, "t": bdt})
        free.append({"name": task["name"], "ok": fok, "t": fdt, "path": path})
        print(f"\nTask: {task['name']}")
        print(f"   baseline 8B : {'PASS' if bok else 'FAIL'}  {bdt:5.1f}s")
        print(f"   FreeLAB     : {'PASS' if fok else 'FAIL'}  {fdt:5.1f}s  [{path}]")
    return base, free


# ============================================================ reporting
def _avg(rows, key="t"):
    xs = [r[key] for r in rows if r.get(key) is not None]
    return sum(xs) / len(xs) if xs else None


def _sp(b, f):
    return f"{b/f:.1f}x" if (b and f) else "-"


def build_report(retr, gen):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"# FreeLAB Code Benchmark - {now}", "",
         f"Baseline = **{fl.VERIFIER}** directly. FreeLAB = compression (retrieval) "
         f"and a draft/test/escalate cascade with the **test suite as judge** ({fl.DRAFTER} drafts).", ""]
    if retr:
        repo, base, free = retr
        approx = fl.approx_tokens(repo)
        bfail = sum(1 for r in base if r["t"] is None)
        pct = sum(r["pct"] for r in free) / len(free) if free else 0
        L += [f"## (A) Code retrieval - {len(CODE_NEEDLES)} facts in a ~{approx}-token repo", "",
              "| Method | Accuracy | Avg latency | Input tokens -> big model |",
              "|---|---|---|---|",
              f"| Baseline {fl.VERIFIER} (whole repo) | {sum(r['ok'] for r in base)}/{len(base)} | "
              f"{(_avg(base) or 0):.1f}s" + (f" ({bfail} failed)" if bfail else "") +
              f" | ~{int(_avg(base,'in_tok') or 0)} |",
              f"| **FreeLAB** (compressed) | {sum(r['ok'] for r in free)}/{len(free)} | "
              f"{(_avg(free) or 0):.1f}s | ~{int(_avg(free,'in_tok') or 0)} |",
              "",
              f"- Speedup: **{_sp(_avg(base), _avg(free))}**; average compression **{pct:.0f}%**.", ""]
    if gen:
        base, free = gen
        n = len(base)
        kept = sum(1 for r in free if r["path"] == "DRAFT-KEPT")
        L += ["## (B) Code generation - test suite as ground-truth judge", "",
              "| Method | Pass rate | Avg time / task |",
              "|---|---|---|",
              f"| Baseline {fl.VERIFIER} | {sum(r['ok'] for r in base)}/{n} | {(_avg(base) or 0):.1f}s |",
              f"| **FreeLAB** (draft/test/escalate) | {sum(r['ok'] for r in free)}/{n} | {(_avg(free) or 0):.1f}s |",
              "",
              f"- Cheap path (drafter passed the tests, big model never ran): **{kept}/{n}**.",
              f"- Speedup: **{_sp(_avg(base), _avg(free))}** (tests are ground truth, so no wrong answer "
              f"is kept unless it actually passes).", ""]
    L += ["---", "*Generated by `freelab_code_bench.py`. See [[FreeLAB Benchmark Results]].*"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="FreeLAB code benchmark.")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--modules", type=int, default=250, help="synthetic repo size")
    ap.add_argument("--keep", type=int, default=fl.DEFAULT_KEEP)
    ap.add_argument("--threshold", type=int, default=fl.DEFAULT_THRESHOLD)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args(argv)

    both = not (args.retrieval or args.gen)
    retr = bench_retrieval(args.modules, args.keep, args.threshold) \
        if (args.retrieval or both) else None
    gen = bench_gen(args.repeats) if (args.gen or both) else None

    report = build_report(retr, gen)
    print("\n" + "=" * 70 + "\n" + report + "\n" + "=" * 70)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    with open(os.path.join(args.out, f"FreeLAB Code Benchmark {stamp}.md"), "w") as f:
        f.write(report)
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()

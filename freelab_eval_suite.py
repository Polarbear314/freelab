#!/usr/bin/env python3
"""
FreeLAB evaluation suite -- the rigorous benchmark.

Turns the ad-hoc benchmarks into a reviewable evaluation: real datasets, an
ablation matrix, objective auto-scoring, multiple seeds, resumable runs, and
bootstrap 95% confidence intervals.

WHAT IT MEASURES
  Document QA (LongBench subset)  -- 2x2 ABLATION isolating each mechanism:
      naive          : whole context -> big model            (compress off, cascade off)
      compress-only  : retrieve top-k -> big model           (compress on,  cascade off)
      cascade-only   : whole context -> draft/judge/escalate (compress off, cascade on)
      freelab        : retrieve top-k -> draft/judge/escalate (compress on,  cascade on)
    Metric: SQuAD-style token-F1 and exact match vs. gold answers.

  Code generation (HumanEval, MBPP) -- the test suite is the ground-truth judge:
      baseline_big   : big model writes it directly
      drafter_only   : small model writes it (no verify)
      freelab        : small drafts -> RUN TESTS -> escalate to big on failure
    Metric: pass@1 (unit tests), plus cheap-path rate and escalation rate.

Efficiency for every config: median & p90 latency, big-model tokens, escalation
rate, and (QA) compression ratio. Accuracy is reported as mean [95% CI] via
bootstrap over items.

USAGE (from the FreeLAB folder, Ollama running)
    pip install datasets
    python3 freelab_eval_suite.py --smoke                 # tiny offline self-test (no download)
    python3 freelab_eval_suite.py --code --limit 40       # HumanEval+MBPP, 40 each
    python3 freelab_eval_suite.py --qa --limit 60 --seeds 3
    python3 freelab_eval_suite.py                          # everything, default sizes

Runs are RESUMABLE: per-item results are cached to <out>/cache/*.jsonl and skipped
on re-run, so you can stop and restart (or run overnight in chunks).

SAFETY: the code experiments execute model-generated code, each candidate in a
separate subprocess with a timeout. Only run on benchmark tasks you trust.
"""

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time

import freelab_core as fl

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================ generation (seedable)
def gen(prompt, model, temperature=0.0, seed=None, num_predict=None):
    """Like fl.generate but exposes temperature/seed for sampling & reproducibility."""
    opts = {"temperature": temperature}
    if num_predict:
        opts["num_predict"] = num_predict
    if seed is not None:
        opts["seed"] = seed
    body = fl._post("/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        "think": False, "keep_alive": "10m", "options": opts})
    return fl._clean(body.get("response") or ""), body.get("eval_count") or 0


# ============================================================ QA scoring (SQuAD-style)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^a-z0-9 ]")


def _norm(s):
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def em_score(pred, golds):
    p = _norm(pred)
    return float(any(p == _norm(g) for g in golds))


def f1_score(pred, golds):
    best = 0.0
    pt = _norm(pred).split()
    for g in golds:
        gt = _norm(g).split()
        if not pt or not gt:
            best = max(best, float(pt == gt))
            continue
        overlap = sum(min(pt.count(w), gt.count(w)) for w in set(pt))
        if overlap == 0:
            continue
        prec = overlap / len(pt)
        rec = overlap / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# ============================================================ bootstrap CI
def bootstrap_ci(values, iters=2000, alpha=0.05, seed=0):
    vals = [v for v in values if v is not None]
    if not vals:
        return (None, None, None)
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(iters):
        s = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return (sum(vals) / n, lo, hi)


def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = min(len(xs) - 1, int(p * len(xs)))
    return xs[k]


# ============================================================ code execution
def extract_code(text, entry):
    text = fl._clean(text)
    if "```" in text:
        parts = text.split("```")
        blocks = [parts[i] for i in range(1, len(parts), 2)]
        if blocks:
            text = max(blocks, key=len)
            if text.lstrip()[:6].lower().startswith(("python", "py\n", "py ")):
                text = text.split("\n", 1)[1] if "\n" in text else text
    idx = text.find(f"def {entry}")
    return text[idx:] if idx != -1 else text


def run_tests(code, program_tail, timeout=12):
    script = f"{code}\n\n{program_tail}\nprint('ALL_PASS')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        return "ALL_PASS" in r.stdout
    except Exception:  # noqa: BLE001
        return False
    finally:
        os.unlink(path)


# ============================================================ datasets
def load_qa(dataset, limit, smoke):
    """Reading-comprehension QA with gold answers for EM/F1.

    Default `squad_v2` (parquet, works with modern `datasets`). Long-context sets
    like THUDM/LongBench use dataset *scripts*, which `datasets>=3` refuses; to use
    them, `pip install "datasets<3"`. SQuAD still exercises the compress/cascade
    ablation on real questions with objective scoring.
    """
    if smoke:
        return [
            {"id": "qa0", "question": "What is the capital of France?",
             "context": ("Filler. " * 200) + " The capital of France is Paris. " + ("Filler. " * 200),
             "answers": ["Paris"]},
            {"id": "qa1", "question": "What year did the Orion module ship?",
             "context": ("Notes. " * 150) + " The Orion module shipped in 2029. " + ("Notes. " * 150),
             "answers": ["2029"]},
        ][:limit or 2]
    from datasets import load_dataset
    name = "rajpurkar/squad_v2" if dataset in ("squad_v2", "squad", "default") else dataset
    ds = load_dataset(name, split="validation")
    out = []
    for r in ds:
        ans = r.get("answers", {}).get("text") or []
        if not ans:                      # skip unanswerable SQuAD-v2 items
            continue
        out.append({"id": r.get("id") or f"q{len(out)}",
                    "question": r["question"], "context": r["context"],
                    "answers": list(dict.fromkeys(ans))})
        if limit and len(out) >= limit:
            break
    return out


def load_humaneval(limit, smoke):
    if smoke:
        return [{
            "id": "he_smoke", "entry": "add",
            "spec": "def add(a, b):\n    \"\"\"Return a+b.\"\"\"\n",
            "tail": "def check(c):\n    assert c(1,2)==3\n    assert c(-1,1)==0\ncheck(add)"}][:limit or 1]
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    out = []
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        out.append({"id": r["task_id"], "entry": r["entry_point"],
                    "spec": r["prompt"], "tail": r["test"] + f"\ncheck({r['entry_point']})"})
    return out


def load_mbpp(limit, smoke):
    if smoke:
        return [{
            "id": "mbpp_smoke", "entry": "sq",
            "spec": "Write a function sq(x) that returns x squared. Example: assert sq(3)==9",
            "tail": "assert sq(3)==9\nassert sq(0)==0"}][:limit or 1]
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    out = []
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        m = re.search(r"assert\s+(\w+)\s*\(", r["test_list"][0])
        entry = m.group(1) if m else "solution"
        spec = (r["text"] + f"\nYour function must be named `{entry}`. "
                f"Example test: {r['test_list'][0]}")
        out.append({"id": f"mbpp-{r['task_id']}", "entry": entry, "spec": spec,
                    "tail": "\n".join(r["test_list"])})
    return out


# ============================================================ runners
def _qa_prompt(q, chunks, concise):
    ctx = fl._context(chunks)
    style = " Answer with the shortest phrase that is correct." if concise else ""
    return (f"Answer the question using ONLY the context below.{style}\n\n"
            f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")


def run_qa(item, compress, cascade, keep, threshold, temperature, seed):
    q, ctx = item["question"], item["context"]
    chunks = fl.chunk_text(ctx)
    t0 = time.perf_counter()
    if compress:
        vecs = fl.embed(chunks)
        kept, _ = fl.rank_chunks(q, chunks, vecs, keep)
        orig_tok = fl.approx_tokens(ctx)
        kept_tok = fl.approx_tokens(fl._context(kept))
    else:
        kept, orig_tok = chunks, fl.approx_tokens(ctx)
        kept_tok = orig_tok
    big_tok, path = 0, "BIG-DIRECT"
    if cascade:
        draft, _ = gen(_qa_prompt(q, kept, True), fl.DRAFTER, temperature, seed, 350)
        score, _ = fl.judge(q, draft)
        if score >= threshold:
            final, path = draft, "DRAFT-KEPT"
        else:
            final, big_tok = gen(_qa_prompt(q, kept, False), fl.VERIFIER, temperature, seed)
            path = "ESCALATED"
    else:
        final, big_tok = gen(_qa_prompt(q, kept, False), fl.VERIFIER, temperature, seed)
    dt = time.perf_counter() - t0
    return {"pred": final, "f1": f1_score(final, item["answers"]),
            "em": em_score(final, item["answers"]), "t": dt, "big_tok": big_tok,
            "escalated": path == "ESCALATED", "kept_draft": path == "DRAFT-KEPT",
            "compression": 100 - 100 * kept_tok // max(orig_tok, 1)}


def _code_prompt(spec):
    return (f"{spec}\n\nComplete the function. Return ONLY the complete Python "
            f"function definition, no explanation, no markdown fences.")


def run_code(item, mode, threshold, temperature, seed):
    entry, tail, spec = item["entry"], item["tail"], item["spec"]

    def attempt(model):
        text, tok = gen(_code_prompt(spec), model, temperature, seed, 512)
        code = extract_code(text, entry)
        if f"def {entry}" not in code:      # model returned only a body -> prepend signature
            code = spec + "\n" + code
        return run_tests(code, tail), tok

    t0 = time.perf_counter()
    if mode == "baseline_big":
        ok, big = attempt(fl.VERIFIER); path = "BIG-DIRECT"
    elif mode == "drafter_only":
        ok, _ = attempt(fl.DRAFTER); big = 0; path = "DRAFT-ONLY"
    else:  # freelab: draft -> tests judge -> escalate
        ok, _ = attempt(fl.DRAFTER); big = 0; path = "DRAFT-KEPT"
        if not ok:
            ok, big = attempt(fl.VERIFIER); path = "ESCALATED"
    return {"passed": float(ok), "t": time.perf_counter() - t0, "big_tok": big,
            "escalated": path == "ESCALATED", "kept_draft": path == "DRAFT-KEPT"}


# ============================================================ resumable cache
class Cache:
    def __init__(self, path):
        self.path = path
        self.done = {}
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                    self.done[r["key"]] = r["val"]
                except Exception:  # noqa: BLE001
                    pass
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a")

    def get(self, key):
        return self.done.get(key)

    def put(self, key, val):
        self.done[key] = val
        self.fh.write(json.dumps({"key": key, "val": val}) + "\n")
        self.fh.flush()


# ============================================================ orchestration
QA_CONFIGS = [("naive", False, False), ("compress-only", True, False),
              ("cascade-only", False, True), ("freelab", True, True)]
CODE_MODES = ["baseline_big", "drafter_only", "freelab"]


def run_qa_bench(items, args, cache):
    print(f"\n########## DOCUMENT QA — {len(items)} items x {len(QA_CONFIGS)} configs "
          f"x {args.seeds} seed(s) ##########")
    results = {c[0]: [] for c in QA_CONFIGS}
    n = len(items)
    for name, comp, casc in QA_CONFIGS:
        for seed in range(args.seeds):
            for i, it in enumerate(items):
                key = f"qa|{name}|{seed}|{it['id']}"
                val = cache.get(key)
                cached = val is not None
                if not cached:
                    val = run_qa(it, comp, casc, args.keep, args.threshold,
                                 args.qa_temperature, seed)
                    cache.put(key, val)
                results[name].append(val)
                print(f"\r  {name:14} seed {seed+1}/{args.seeds}  item {i+1}/{n}"
                      f"{'  (cached)' if cached else '          '}", end="", flush=True)
        f1 = sum(r["f1"] for r in results[name]) / len(results[name])
        print(f"\r  {name:14} F1={f1:.3f}  (n={len(results[name])})" + " " * 20)
    return results


def run_code_bench(name, items, args, cache):
    print(f"\n########## CODE — {name} — {len(items)} items x {len(CODE_MODES)} modes "
          f"x {args.seeds} seed(s) ##########")
    results = {m: [] for m in CODE_MODES}
    n = len(items)
    for mode in CODE_MODES:
        for seed in range(args.seeds):
            for i, it in enumerate(items):
                key = f"code|{name}|{mode}|{seed}|{it['id']}"
                val = cache.get(key)
                cached = val is not None
                if not cached:
                    val = run_code(it, mode, args.threshold, args.code_temperature, seed)
                    cache.put(key, val)
                results[mode].append(val)
                print(f"\r  {mode:14} seed {seed+1}/{args.seeds}  item {i+1}/{n}"
                      f"{'  (cached)' if cached else '          '}", end="", flush=True)
        p = sum(r["passed"] for r in results[mode]) / len(results[mode])
        print(f"\r  {mode:14} pass@1={p:.3f}  (n={len(results[mode])})" + " " * 20)
    return results


# ============================================================ reporting
def _ci_str(vals):
    m, lo, hi = bootstrap_ci(vals)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]" if m is not None else "-"


def report_qa(results):
    L = ["## Document QA — 2x2 ablation (F1, exact match)", "",
         "| Config | F1 [95% CI] | EM | Median lat | p90 lat | Big tokens | Escalation | Compression |",
         "|---|---|---|---|---|---|---|---|"]
    for name, _, _ in QA_CONFIGS:
        rs = results[name]
        if not rs:
            continue
        f1 = _ci_str([r["f1"] for r in rs])
        em = sum(r["em"] for r in rs) / len(rs)
        med = _pct([r["t"] for r in rs], 0.5)
        p90 = _pct([r["t"] for r in rs], 0.9)
        big = sum(r["big_tok"] for r in rs) / len(rs)
        esc = 100 * sum(r["escalated"] for r in rs) / len(rs)
        comp = sum(r["compression"] for r in rs) / len(rs)
        star = "**" if name == "freelab" else ""
        L.append(f"| {star}{name}{star} | {f1} | {em:.3f} | {med:.1f}s | {p90:.1f}s | "
                 f"{big:.0f} | {esc:.0f}% | {comp:.0f}% |")
    L += ["", "*naive = whole context to the big model (a normal local LLM). "
          "The 2x2 isolates compression and the cascade; `freelab` is both.*", ""]
    return "\n".join(L)


def report_code(name, results):
    L = [f"## Code generation — {name} (pass@1, unit tests as judge)", "",
         "| Mode | pass@1 [95% CI] | Median lat | Escalation | Cheap-path |",
         "|---|---|---|---|---|"]
    for mode in CODE_MODES:
        rs = results[mode]
        if not rs:
            continue
        p = _ci_str([r["passed"] for r in rs])
        med = _pct([r["t"] for r in rs], 0.5)
        esc = 100 * sum(r["escalated"] for r in rs) / len(rs)
        cheap = 100 * sum(r["kept_draft"] for r in rs) / len(rs)
        star = "**" if mode == "freelab" else ""
        L.append(f"| {star}{mode}{star} | {p} | {med:.1f}s | {esc:.0f}% | {cheap:.0f}% |")
    L += ["", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="FreeLAB rigorous eval suite.")
    ap.add_argument("--qa", action="store_true")
    ap.add_argument("--code", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny inline data, no download")
    ap.add_argument("--limit", type=int, default=50, help="items per dataset")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--keep", type=int, default=fl.DEFAULT_KEEP)
    ap.add_argument("--threshold", type=int, default=fl.DEFAULT_THRESHOLD)
    ap.add_argument("--qa-temperature", type=float, default=0.0)
    ap.add_argument("--code-temperature", type=float, default=0.2)
    ap.add_argument("--qa-dataset", default="squad_v2",
                    help="HF QA dataset (default squad_v2). LongBench needs datasets<3.")
    ap.add_argument("--longbench-subset", default=None, help="(deprecated) use --qa-dataset")
    ap.add_argument("--out", default=os.path.join(HERE, "eval_results"))
    args = ap.parse_args(argv)

    both = not (args.qa or args.code)
    os.makedirs(args.out, exist_ok=True)
    cache = Cache(os.path.join(args.out, "cache", "results.jsonl"))
    sections = []

    if args.qa or both:
        try:
            if args.longbench_subset:
                print(f"note: --longbench-subset is deprecated and ignored; using "
                      f"--qa-dataset={args.qa_dataset}. For LongBench, pip install "
                      f"'datasets<3' and pass --qa-dataset THUDM/LongBench.")
            items = load_qa(args.qa_dataset, args.limit, args.smoke)
            sections.append(report_qa(run_qa_bench(items, args, cache)))
        except ImportError:
            print("QA needs `pip install datasets` (or use --smoke).")

    if args.code or both:
        for name, loader in [("HumanEval", load_humaneval), ("MBPP", load_mbpp)]:
            try:
                items = loader(args.limit, args.smoke)
            except ImportError:
                print("Code benches need `pip install datasets` (or use --smoke).")
                break
            sections.append(report_code(name, run_code_bench(name, items, args, cache)))

    import datetime
    head = (f"# FreeLAB Evaluation Suite — {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n"
            f"Models: {fl.DRAFTER} (drafter) · {fl.VERIFIER} (verifier/judge). "
            f"Accuracy = mean [bootstrap 95% CI] over items; {args.seeds} seed(s). "
            f"keep={args.keep}, threshold={args.threshold}.\n")
    report = head + "\n" + "\n".join(sections) + \
        "\n---\n*Generated by `freelab_eval_suite.py`. Resumable cache in "\
        "`eval_results/cache/`. See [[FreeLAB Benchmark Results]].*\n"
    print("\n" + "=" * 70 + "\n" + report + "=" * 70)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    with open(os.path.join(args.out, f"FreeLAB Eval Suite {stamp}.md"), "w") as f:
        f.write(report)
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()

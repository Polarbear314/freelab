#!/usr/bin/env python3
"""
FreeLAB eval -- the moment of truth.

Two tests:

(A) JUDGE TEST  (--judge)
    A set of questions a 0.6B model tends to get WRONG (letter counting, the
    bat-and-ball trap, 9.11 vs 9.9, etc.). We watch whether the big-model judge
    catches the bad drafts and escalates them -- and whether escalation fixes
    the answer. This is what decides if "big-model quality" is real or hoped-for.

(B) CHUNK TEST  (--chunk)
    Builds a genuinely large document (tens of thousands of tokens) with several
    facts buried at different positions, then asks for each. Tests whether
    query-aware compression still finds the needle at scale and how much it
    shrinks the input.

Reuses the real pipeline from freelab.py, so run this from the FreeLAB folder:
    python3 freelab_eval.py --judge
    python3 freelab_eval.py --chunk
    python3 freelab_eval.py            # both
"""

import argparse
import time
import freelab_core as fl


def judge_score(q, draft, deep):
    """Cheap judge = snap verdict (few tokens, no reasoning).
    Deep judge = the expensive model actually reasons through the check first,
    then outputs a score. This is the "spend compute on judging" idea."""
    if not deep:
        s, _ = fl.judge(q, draft)      # the original cheap judge
        return s
    prompt = ("Carefully check whether the draft answer is correct. Reason "
              "step by step, then on the FINAL line output exactly "
              "'SCORE: <integer 0-100>' for how correct the draft is.\n\n"
              f"Question: {q}\n\nDraft answer: {draft}\n")
    text, _ = fl.generate(prompt, fl.VERIFIER, num_predict=400)
    m = fl.SCORE_RE.search(text)
    return int(m.group(1)) if m else 0


def correct(answer, expected):
    """Loose check: does the answer contain any accepted form of the truth?"""
    a = answer.lower()
    return any(e.lower() in a for e in expected)


# --- (A) questions a small model tends to blow -------------------------------
HARD = [
    {"q": "How many times does the letter 'r' appear in the word 'strawberry'?",
     "expect": ["3", "three"]},
    {"q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
          "the ball. How much does the ball cost?",
     "expect": ["0.05", "5 cent", "five cent", "$.05"]},
    {"q": "Which number is larger, 9.11 or 9.9?", "expect": ["9.9"]},
    {"q": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
     "expect": ["9", "nine"]},
    {"q": "What is 47 multiplied by 53?", "expect": ["2491"]},
    {"q": "If today is Wednesday, what day of the week will it be 100 days from now?",
     "expect": ["friday"]},
    {"q": "I have 3 apples. I eat 2 bananas. How many apples do I have left?",
     "expect": ["3", "three"]},
]


def cascade_qa(q, threshold, deep):
    """Plain Q&A cascade (no document): draft -> judge -> maybe escalate."""
    draft, _ = fl.generate(f"Answer concisely. {q}", fl.DRAFTER,
                           num_predict=fl.DRAFT_MAX_TOKENS)
    score = judge_score(q, draft, deep)
    if score >= threshold:
        return {"path": "DRAFT-KEPT", "draft": draft, "final": draft, "score": score}
    final, _ = fl.generate(f"Answer concisely. {q}", fl.VERIFIER)
    return {"path": "ESCALATED", "draft": draft, "final": final, "score": score}


def run_judge_test(threshold, deep):
    print(f"\n########## (A) JUDGE TEST  ({'EXTENDED' if deep else 'STANDARD'} judge) ##########")
    fl.warmup(use_compress=False, use_cascade=True)
    rows = []
    t0 = time.time()
    for item in HARD:
        r = cascade_qa(item["q"], threshold, deep)
        r["draft_ok"] = correct(r["draft"], item["expect"])
        r["final_ok"] = correct(r["final"], item["expect"])
        rows.append(r)
        print(f"\nQ: {item['q']}")
        print(f"   drafter: {'OK ' if r['draft_ok'] else 'WRONG'} "
              f"| judge {r['score']:>3} -> {r['path']} "
              f"| final: {'OK' if r['final_ok'] else 'WRONG'}")
        print(f"   answer: {r['final'][:160].strip()}")

    n = len(rows)
    wrong = [r for r in rows if not r["draft_ok"]]
    caught = [r for r in wrong if r["path"] == "ESCALATED"]
    fixed = [r for r in caught if r["final_ok"]]
    leaked = [r for r in wrong if r["path"] == "DRAFT-KEPT"]  # bad draft waved through

    elapsed = time.time() - t0
    print("\n----- JUDGE TEST SUMMARY -----")
    print(f"judge mode            : {'EXTENDED (full reasoning)' if deep else 'STANDARD (bounded verify)'}")
    print(f"total time            : {elapsed:.1f}s  ({elapsed/n:.1f}s/question)")
    print(f"drafter alone correct : {n - len(wrong)}/{n}")
    print(f"pipeline correct      : {sum(r['final_ok'] for r in rows)}/{n}")
    print(f"bad drafts            : {len(wrong)}")
    print(f"  caught (escalated)  : {len(caught)}/{len(wrong)}   <- judge doing its job")
    print(f"  of those, fixed     : {len(fixed)}/{len(caught)}")
    print(f"  LEAKED (kept wrong) : {len(leaked)}/{len(wrong)}   <- judge failures")
    if leaked:
        print("  leaked questions:")
        for r in leaked:
            print(f"    - score {r['score']}: {r['final'][:80].strip()}")


# --- (B) large-document chunking ---------------------------------------------
NEEDLES = [
    {"text": "Operational note: the maximum rated voltage for the Orion relay "
             "module is 480 volts; exceeding it trips the safety interlock.",
     "q": "What is the maximum rated voltage for the Orion relay module?",
     "expect": ["480"]},
    {"text": "Schedule update: Project Falcon's launch has been rescheduled to "
             "March 14, 2029, pending a final design review.",
     "q": "What is the new launch date for Project Falcon?",
     "expect": ["march 14", "2029"]},
    {"text": "Security reminder: the emergency shutdown passphrase for the west "
             "facility is DELTA-SEVEN-SIGMA and must never be shared externally.",
     "q": "What is the emergency shutdown passphrase for the west facility?",
     "expect": ["delta-seven-sigma", "delta"]},
]


def build_haystack(paras):
    filler = ("The quarterly operations review covered staffing levels, budget "
              "variance, vendor timelines, and routine compliance items, with no "
              "material exceptions noted during the reporting period. ")
    base = [filler * 3 for _ in range(paras)]
    step = paras // (len(NEEDLES) + 1)
    for i, n in enumerate(NEEDLES):          # spread needles across the document
        base.insert((i + 1) * step, n["text"])
    return "\n\n".join(base)


def run_chunk_test(paras, keep, threshold):
    print("\n########## (B) CHUNK TEST ##########")
    doc = build_haystack(paras)
    approx = fl.approx_tokens(doc)
    print(f"document: ~{approx} tokens, {len(fl.chunk_text(doc))} chunks, "
          f"{len(NEEDLES)} needles buried")
    fl.warmup(use_compress=True, use_cascade=True)
    hits = 0
    for n in NEEDLES:
        final, s = fl.pipeline(n["q"], doc, keep, threshold,
                              use_compress=True, use_cascade=True)
        ok = correct(final, n["expect"])
        hits += ok
        pct = 100 - 100 * s["kept_tok"] // s["orig_tok"]
        print(f"\nQ: {n['q']}")
        print(f"   compressed {pct}% | path {s['path']} | "
              f"{'FOUND' if ok else 'MISSED'}")
        print(f"   answer: {final[:160].strip()}")
    print(f"\n----- CHUNK TEST SUMMARY -----")
    print(f"needles found: {hits}/{len(NEEDLES)}  in a ~{approx}-token document")


def main():
    ap = argparse.ArgumentParser(description="FreeLAB eval.")
    ap.add_argument("--judge", action="store_true", help="run the judge test only")
    ap.add_argument("--chunk", action="store_true", help="run the chunk test only")
    ap.add_argument("--deep-judge", action="store_true",
                    help="let the judge reason before scoring (vs cheap snap verdict)")
    ap.add_argument("--paras", type=int, default=300, help="filler paragraphs (doc size)")
    ap.add_argument("--keep", type=int, default=fl.DEFAULT_KEEP)
    ap.add_argument("--threshold", type=int, default=fl.DEFAULT_THRESHOLD)
    args = ap.parse_args()

    run_both = not (args.judge or args.chunk)
    if args.judge or run_both:
        run_judge_test(args.threshold, args.deep_judge)
    if args.chunk or run_both:
        run_chunk_test(args.paras, args.keep, args.threshold)


if __name__ == "__main__":
    main()

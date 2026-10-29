#!/usr/bin/env python3
"""Raw / lenient / strict judge-corrected accuracy for the four NV-Reason-CT input modes.

lenient = raw + rejected items the auditor labelled correct_paraphrase or correct_extra_detail
strict  = raw + correct_paraphrase + correct_extra_detail with extra_contradicts_report == false
Denominator is always the full eval file (judge-unparsed items count as wrong, reported separately).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[2] / "reports"
RUNS = {  # mode -> (judge file, auditor file)
    "blind": ("nv_reason_ct_blind_qwen_eval.json", "nv_reason_ct_blind_failure_categories_v2.json"),
    "image": ("nv_reason_ct_qwen_eval.json", "nv_reason_ct_image_failure_categories_v2.json"),
    "report": ("nv_reason_ct_report_qwen_eval.json", "nv_reason_ct_report_failure_categories_v2.json"),
    "image_report": ("nv_reason_ct_image_report_qwen_eval.json", "nv_reason_ct_image_report_failure_categories_v2.json"),
}
RECOVERABLE = {"correct_paraphrase", "correct_extra_detail"}


def wilson_free_rate(k: int, n: int) -> float:
    return k / n if n else float("nan")


def summarize(mode: str, eval_name: str, cat_name: str) -> dict:
    ev = json.load(open(REPORTS / "llm_eval" / eval_name))
    cats = {(r["case_id"], r["question"]): r for r in json.load(open(REPORTS / "llm_eval" / cat_name))}
    n = len(ev)
    raw = sum(r["is_correct"] is True for r in ev)
    unparsed = sum(r["is_correct"] is None for r in ev)
    lenient = strict = unaudited = 0
    versions = set()
    for r in ev:
        if r["is_correct"] is not False:
            continue
        c = cats.get((r["case_id"], r["question"]))
        if not c or not c.get("category"):
            unaudited += 1
            continue
        versions.add(c.get("prompt_version", "v1"))
        if c["category"] in RECOVERABLE:
            lenient += 1
            if c["category"] == "correct_paraphrase" or c.get("extra_contradicts_report") is False:
                strict += 1
    out = {
        "n": n, "raw": raw / n, "lenient": (raw + lenient) / n, "strict": (raw + strict) / n,
        "judge_unparsed": unparsed, "audit_unparsed": unaudited, "prompt_versions": sorted(versions),
    }
    acc_path = REPORTS / "llm_eval" / cat_name.replace(".json", "_accepted.json")
    if acc_path.exists():
        acc = [r for r in json.load(open(acc_path)) if r.get("judge_false_accept") is not None]
        fa = sum(r["judge_false_accept"] for r in acc)
        out.update(false_accept_rate=wilson_free_rate(fa, len(acc)), false_accept_n=len(acc),
                   # raw accuracy after removing the estimated false accepts
                   lenient_minus_fa=(raw * (1 - fa / len(acc)) + lenient) / n if acc else None,
                   strict_minus_fa=(raw * (1 - fa / len(acc)) + strict) / n if acc else None)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=REPORTS / "analysis" / "nv_reason_ct_corrected_accuracy.json")
    args = p.parse_args()
    res = {}
    for mode, (e, c) in RUNS.items():
        if not (REPORTS / "llm_eval" / c).exists():
            print(f"skip {mode}: {c} missing")
            continue
        res[mode] = summarize(mode, e, c)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2))
    f = lambda x: "  n/a" if x is None else f"{100 * x:5.1f}"
    print(f"{'mode':<13}{'raw':>6}{'lenient':>9}{'strict':>8}{'false-acc':>10}{'strict-FA':>10}  unparsed(judge/audit)  prompt")
    for m, r in res.items():
        print(f"{m:<13}{f(r['raw'])}{f(r['lenient']):>9}{f(r['strict']):>8}{f(r.get('false_accept_rate')):>10}"
              f"{f(r.get('strict_minus_fa')):>10}  {r['judge_unparsed']}/{r['audit_unparsed']}  {','.join(r['prompt_versions'])}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Split CT-SpatialVQA into easy / hard by a question-only (blind) baseline and
re-score every model's existing judge verdicts on each subset.

Easy = the blind text LLM answered correctly from the question alone (priors);
hard = it did not. Older judge files carry case_id=null, so they are aligned to
the dataset by question text in dataset order.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MODELS = {
    "ctchat": "CT-Chat", "merlin": "MERLIN", "med3dvlm": "Med3DVLM", "m3d": "M3D",
    "radfm": "RadFM", "vila_m3_vista3d": "VILA-M3", "medgemma": "MedGemma", "medevalkit": "MedEvalKit",
}
JUDGES = {"gpt": "{m}_gpt_eval.json", "gemini": "{m}_gemini2.5_eval_full.json", "qwen": "{m}_qwen_eval.json"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("dataset/ct_spatialvqa/spatial_qa_filtered_full_nifti.jsonl"))
    p.add_argument("--llm-eval-dir", type=Path, default=Path("benchmarking/reports/llm_eval"))
    p.add_argument("--blind-eval", type=Path,
                   default=Path("benchmarking/reports/llm_eval/qwen2.5_14b_blind_qwen_eval.json"))
    p.add_argument("--extra", nargs="*", default=["nv_reason_ct_qwen_eval.json"],
                   help="Additional single-judge eval files (keyed by case_id) to report.")
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def by_key(path: Path) -> dict:
    return {(r["case_id"], r["question"]): r.get("is_correct") for r in json.load(open(path))}


def aligned(path: Path, ds: list[dict]) -> list:
    out = [None] * len(ds)
    j = 0
    for r in json.load(open(path)):
        while j < len(ds) and ds[j]["question"] != r["question"]:
            j += 1
        if j >= len(ds):
            break
        out[j] = r.get("is_correct")
        j += 1
    return out


def acc(v: list, idx) -> float:
    kk = [i for i in idx if v[i] is not None]
    return sum(v[i] for i in kk) / len(kk) if kk else float("nan")


def main() -> None:
    args = parse_args()
    ds = [json.loads(l) for l in open(args.dataset) if l.strip()]
    n = len(ds)
    blind = by_key(args.blind_eval)
    qb = [blind.get((x["case_id"], x["question"])) for x in ds]
    easy = [i for i, v in enumerate(qb) if v is True]
    hard = [i for i, v in enumerate(qb) if v is False]
    print(f"blind baseline {args.blind_eval.name}: easy {len(easy)}  hard {len(hard)}  unjudged {qb.count(None)}")

    results = {"easy": len(easy), "hard": len(hard), "models": {}}
    rows = []
    for m, name in MODELS.items():
        V = {j: aligned(args.llm_eval_dir / pat.format(m=m), ds) for j, pat in JUDGES.items()}
        jury = [(sum(V[j][i] for j in JUDGES) >= 2) if all(V[j][i] is not None for j in JUDGES) else None
                for i in range(n)]
        row = {"all": acc(jury, range(n)), "easy": acc(jury, easy), "hard": acc(jury, hard),
               "per_judge_hard": {j: acc(V[j], hard) for j in JUDGES}}
        results["models"][name] = row
        rows.append((name, row))

    print("\nLLM-as-jury accuracy; easy/hard = blind baseline right/wrong")
    print("model | all | easy | hard")
    for name, r in sorted(rows, key=lambda x: -x[1]["hard"]):
        print(f"{name} | {100*r['all']:.1f} | {100*r['easy']:.1f} | {100*r['hard']:.1f}")
    print("mean of %d | %.1f | %.1f | %.1f" % (
        len(rows), 100 * statistics.mean(r["all"] for _, r in rows),
        100 * statistics.mean(r["easy"] for _, r in rows), 100 * statistics.mean(r["hard"] for _, r in rows)))

    for extra in args.extra:
        E = by_key(args.llm_eval_dir / extra)
        v = [E.get((x["case_id"], x["question"])) for x in ds]
        row = {"all": acc(v, range(n)), "easy": acc(v, easy), "hard": acc(v, hard)}
        results["models"][extra] = row
        print(f"{extra} (single judge) | {100*row['all']:.1f} | {100*row['easy']:.1f} | {100*row['hard']:.1f}")

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Categorize judge-rejected answers of the report-oracle run with a local Qwen model.

For each rejected item the classifier sees the source radiology report, the question,
the ground-truth answer and the model prediction, and assigns one failure category plus
two grounding flags (is the prediction / the ground truth supported by the report).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CATEGORIES = {
    "correct_extra_detail": "Prediction states the ground-truth fact correctly AND adds extra findings "
                            "(the judge was too strict).",
    "correct_paraphrase": "Prediction is semantically equivalent to the ground truth with no extra "
                          "findings and no omission (the judge was wrong).",
    "incomplete": "Prediction covers only part of a multi-part ground truth (omits a location, side, "
                  "lobe or segment).",
    "wrong_location": "Prediction names a different anatomical location, side, lobe, level or extent "
                      "than the ground truth.",
    "wrong_presence": "Prediction contradicts the ground truth about presence/absence or yes/no.",
    "non_answer": "Prediction hedges, refuses, or says it cannot determine the answer.",
    "gt_issue": "The ground truth is not supported by the report, or the question is ambiguous / "
                "unanswerable from the report (e.g. translation artefact).",
    "other": "None of the above.",
}

SYSTEM_PROMPT = "You are a meticulous radiology QA auditor."

PROMPT_VERSION = "v2"

INPUT_MODE_INTRO = {
    "blind": "A question-answering model was given ONLY the QUESTION below (no image, no report).",
    "image": "A question-answering model was given the CT volume and the QUESTION below (it did NOT see the report).",
    "report": "A question-answering model was given the radiology REPORT below and the QUESTION.",
    "image_report": "A question-answering model was given the CT volume, the radiology REPORT below and the QUESTION.",
}

PROMPT_TEMPLATE = """{intro}
The REPORT is shown to you, the auditor, as reference only. Its PREDICTION was marked wrong by an
automatic judge comparing it to the GROUND TRUTH (which was itself written from the same REPORT).

Classify why the prediction was rejected. Pick exactly one category:
{categories}

Also decide:
- pred_in_report: is the PREDICTION's main claim supported by the REPORT? (true/false)
- gt_in_report: is the GROUND TRUTH supported by the REPORT? (true/false)
- extra_contradicts_report: does the PREDICTION contain extra findings beyond the GROUND TRUTH that
  contradict or are absent from the REPORT (i.e. invented)? true/false, or null if it adds nothing.

Respond strictly as a JSON list, one object per item:
[{{"index": <int>, "category": "<category>", "pred_in_report": true/false, "gt_in_report": true/false, "extra_contradicts_report": true/false/null, "note": "<max 15 words>"}}]

Items:
{items}
"""

ACCEPT_TEMPLATE = """{intro}
The REPORT is shown to you, the auditor, as reference only. Its PREDICTION was marked CORRECT by an
automatic judge comparing it to the GROUND TRUTH (which was itself written from the same REPORT).

For each item decide whether the judge was right to accept it:
- judge_false_accept: true if the PREDICTION is actually wrong, contradicts the GROUND TRUTH, omits a key
  part of it, or is not supported by the REPORT; false if the acceptance was justified.

Respond strictly as a JSON list, one object per item:
[{{"index": <int>, "judge_false_accept": true/false, "note": "<max 15 words>"}}]

Items:
{items}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", type=Path, required=True, help="Judge output (list with is_correct).")
    p.add_argument("--reports-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input-mode", choices=sorted(INPUT_MODE_INTRO), required=True,
                   help="Input condition of the audited run; sets the first line of the prompt.")
    p.add_argument("--audit-accepted", type=int, default=0,
                   help="Also audit N judge-ACCEPTED items (seed 0) -> <output stem>_accepted.json.")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--batch-size", type=int, default=3, help="Items per prompt (each carries a report).")
    p.add_argument("--gen-batch", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def fmt_report(rec: dict) -> str:
    return f"Findings: {(rec.get('Findings_EN') or '').strip()}\nImpressions: {(rec.get('Impressions_EN') or '').strip()}"


def build_prompt(batch: list[dict], reports: dict, input_mode: str, accepted: bool = False) -> str:
    cats = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    items = []
    for i, r in enumerate(batch, start=1):
        items.append(
            f"### Item {i}\nREPORT:\n{fmt_report(reports[r['case_id']])}\n"
            f"QUESTION: {r['question']}\nGROUND TRUTH: {r['answer']}\nPREDICTION: {r['prediction']}\n"
        )
    intro = INPUT_MODE_INTRO[input_mode]
    if accepted:
        return ACCEPT_TEMPLATE.format(intro=intro, items="\n".join(items))
    return PROMPT_TEMPLATE.format(intro=intro, categories=cats, items="\n".join(items))


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`\n")
        if text.startswith("json\n"):
            text = text[5:]
    return text


def parse_reply(text: str, n: int, accepted: bool = False) -> list:
    out = [None] * n
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        return out
    try:
        parsed = json.loads(strip_code_fence(text[start:end + 1]))
    except json.JSONDecodeError:
        return out
    if not isinstance(parsed, list):
        return out
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not (isinstance(idx, int) and 1 <= idx <= n):
            continue
        if accepted:
            if isinstance(item.get("judge_false_accept"), bool):
                out[idx - 1] = {"judge_false_accept": item["judge_false_accept"], "note": item.get("note")}
        elif item.get("category") in CATEGORIES:
            out[idx - 1] = {
                "category": item["category"],
                "pred_in_report": item.get("pred_in_report"),
                "gt_in_report": item.get("gt_in_report"),
                "extra_contradicts_report": item.get("extra_contradicts_report"),
                "note": item.get("note"),
            }
    return out


def audit(items: list[dict], output: Path, key: str, accepted: bool, args, reports, tok, model) -> list:
    """Run the auditor over `items`, resuming from `output`; `key` is the field that marks a finished row."""
    accepted_items = []
    if args.audit_accepted > 0:
        import random
        ok = [r for r in evals if r.get("is_correct") is True]
        accepted_items = random.Random(0).sample(ok, min(args.audit_accepted, len(ok)))

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, quantization_config=quant, device_map={"": args.device},
    ).eval()

    results = audit(fails, args.output, "category", False, args, reports, tok, model)
    from collections import Counter
    print(json.dumps(Counter(r["category"] for r in results), indent=2))
    if accepted_items:
        acc_out = args.output.with_name(args.output.stem + "_accepted.json")
        acc = audit(accepted_items, acc_out, "judge_false_accept", True, args, reports, tok, model)
        print(json.dumps(Counter(r["judge_false_accept"] for r in acc), indent=2))


if __name__ == "__main__":
    main()

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

PROMPT_TEMPLATE = """A question-answering model was given the radiology REPORT below and asked the QUESTION.
Its PREDICTION was marked wrong by an automatic judge comparing it to the GROUND TRUTH
(which was itself written from the same REPORT).

Classify why the prediction was rejected. Pick exactly one category:
{categories}

Also decide:
- pred_in_report: is the PREDICTION's main claim supported by the REPORT? (true/false)
- gt_in_report: is the GROUND TRUTH supported by the REPORT? (true/false)

Respond strictly as a JSON list, one object per item:
[{{"index": <int>, "category": "<category>", "pred_in_report": true/false, "gt_in_report": true/false, "note": "<max 15 words>"}}]

Items:
{items}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", type=Path, required=True, help="Judge output (list with is_correct).")
    p.add_argument("--reports-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
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


def build_prompt(batch: list[dict], reports: dict) -> str:
    cats = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    items = []
    for i, r in enumerate(batch, start=1):
        items.append(
            f"### Item {i}\nREPORT:\n{fmt_report(reports[r['case_id']])}\n"
            f"QUESTION: {r['question']}\nGROUND TRUTH: {r['answer']}\nPREDICTION: {r['prediction']}\n"
        )
    return PROMPT_TEMPLATE.format(categories=cats, items="\n".join(items))


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`\n")
        if text.startswith("json\n"):
            text = text[5:]
    return text


def parse_reply(text: str, n: int) -> list:
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
        cat = item.get("category")
        if isinstance(idx, int) and 1 <= idx <= n and cat in CATEGORIES:
            out[idx - 1] = {
                "category": cat,
                "pred_in_report": item.get("pred_in_report"),
                "gt_in_report": item.get("gt_in_report"),
                "note": item.get("note"),
            }
    return out


def main() -> None:
    args = parse_args()
    reports = json.load(open(args.reports_json))
    evals = json.load(open(args.eval_json))
    fails = [r for r in evals if r.get("is_correct") is False]
    if args.limit > 0:
        fails = fails[:args.limit]

    results = []
    if args.resume and args.output.exists():
        results = [r for r in json.loads(args.output.read_text()) if r.get("category")]
    done = {(r["case_id"], r["question"]) for r in results}
    todo = [r for r in fails if (r["case_id"], r["question"]) not in done]
    print(f"{len(fails)} failures, {len(done)} already categorized, {len(todo)} to do")
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, quantization_config=quant, device_map={"": args.device},
    ).eval()

    groups = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for g0 in tqdm(range(0, len(groups), args.gen_batch), desc="Categorizing", unit="step"):
        chunk = groups[g0:g0 + args.gen_batch]
        texts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": build_prompt(b, reports)}],
                tokenize=False, add_generation_prompt=True,
            )
            for b in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        replies = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for batch, reply in zip(chunk, replies):
            for rec, parsed in zip(batch, parse_reply(reply, len(batch))):
                results.append({
                    "case_id": rec["case_id"], "question": rec["question"],
                    "answer": rec["answer"], "prediction": rec["prediction"],
                    **(parsed or {"category": None, "pred_in_report": None, "gt_in_report": None, "note": None}),
                })
        args.output.write_text(json.dumps(results, indent=2))

    from collections import Counter
    print(json.dumps(Counter(r["category"] for r in results), indent=2))


if __name__ == "__main__":
    main()

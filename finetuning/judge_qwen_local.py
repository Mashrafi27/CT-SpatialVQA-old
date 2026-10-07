#!/usr/bin/env python3
"""LLM-as-a-judge with a local Qwen model (no API key needed).

Same prompt, batching (N QA pairs per prompt) and output format as
benchmarking/eval_scripts/scripts/evaluate_with_qwen.py, but generation runs
on a local GPU with transformers. Verdicts are matched to items by the
"index" field the judge returns, not by list position.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarking/eval_scripts/scripts"))
from evaluate_with_qwen import PROMPT_TEMPLATE, load_predictions, sanitize_json, strip_code_fence  # noqa: E402

SYSTEM_PROMPT = "You are a strict medical QA evaluator."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local Qwen judge for QA predictions")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--batch-size", type=int, default=5, help="QA pairs per prompt (matches API judges)")
    p.add_argument("--gen-batch", type=int, default=8, help="Prompts generated in parallel")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--prediction-field", default="prediction")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0, help="Judge only the first N predictions (for testing)")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def build_prompt(batch: list[dict], field: str) -> str:
    entries = [
        f"{i}. Question: {r.get('question')}\n   Answer: {r.get('answer')}\n   Prediction: {r.get(field)}"
        for i, r in enumerate(batch, start=1)
    ]
    return PROMPT_TEMPLATE.format(batch="\n".join(entries))


def parse_verdicts(text: str, n: int) -> list:
    """Return a list of n verdicts (True/False/None), keyed by the judge's 1-based index."""
    verdicts = [None] * n
    try:
        parsed = json.loads(sanitize_json(strip_code_fence(text)))
    except json.JSONDecodeError:
        return verdicts
    if not isinstance(parsed, list):
        return verdicts
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx, val = item.get("index"), item.get("is_correct")
        if isinstance(idx, int) and 1 <= idx <= n and isinstance(val, bool):
            verdicts[idx - 1] = val
    return verdicts


def main():
    args = parse_args()
    preds = load_predictions(args.predictions)
    if args.limit > 0:
        preds = preds[:args.limit]

    judgments = []
    if args.resume and args.output.exists():
        judgments = json.loads(args.output.read_text())
        # Re-judge items whose verdict failed to parse last time
        judgments = [j for j in judgments if j.get("is_correct") is not None]
    done = {(j["case_id"], j["question"]) for j in judgments}
    todo = [r for r in preds if (r["case_id"], r["question"]) not in done]
    print(f"{len(preds)} predictions, {len(done)} already judged, {len(todo)} to judge")
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
    for g0 in tqdm(range(0, len(groups), args.gen_batch), desc="Judging", unit="step"):
        chunk = groups[g0:g0 + args.gen_batch]
        texts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": build_prompt(b, args.prediction_field)}],
                tokenize=False, add_generation_prompt=True,
            )
            for b in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        replies = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        for batch, reply in zip(chunk, replies):
            for rec, verdict in zip(batch, parse_verdicts(reply, len(batch))):
                judgments.append({
                    "case_id": rec.get("case_id"), "question": rec.get("question"),
                    "answer": rec.get("answer"), "prediction": rec.get(args.prediction_field),
                    "is_correct": verdict,
                })
        args.output.write_text(json.dumps(judgments, indent=2))

    judged = [j for j in judgments if j["is_correct"] is not None]
    correct = sum(j["is_correct"] for j in judged)
    print(json.dumps({"total": len(judgments), "judged": len(judged), "correct": correct,
                      "accuracy": correct / len(judged) if judged else 0.0}, indent=2))


if __name__ == "__main__":
    main()

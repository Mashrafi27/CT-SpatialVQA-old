#!/usr/bin/env python3
"""Question-only (blind) baseline with a plain text LLM.

Answers every CT-SpatialVQA question without any image or report, so the score
measures how much of the benchmark is answerable from priors and question wording
alone. Intended for a model that is NOT one of the evaluated 3D VLMs, so the
resulting easy/hard split is independent of the models under test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are a radiologist. The question below is about a chest CT scan that you cannot see. "
    "Give your single best answer in one short sentence. Do not say that you need the image."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = [json.loads(l) for l in args.dataset.open() if l.strip()]
    if args.limit > 0:
        records = records[:args.limit]

    done = set()
    if args.resume and args.output.exists():
        for l in args.output.open():
            if l.strip():
                r = json.loads(l)
                done.add((r["case_id"], r["question"]))
    todo = [r for r in records if (r["case_id"], r["question"]) not in done]
    print(f"{len(records)} questions, {len(done)} done, {len(todo)} to do")
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as f_out:
        for i in tqdm(range(0, len(todo), args.batch_size), desc="Blind LLM", unit="batch"):
            batch = todo[i:i + args.batch_size]
            texts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": r["question"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                for r in batch
            ]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
            replies = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, reply in zip(batch, replies):
                f_out.write(json.dumps({
                    "case_id": r["case_id"], "question": r["question"], "answer": r["answer"],
                    "prediction": reply.strip(), "model_id": args.model, "input_mode": "blind",
                }) + "\n")
            f_out.flush()


if __name__ == "__main__":
    main()

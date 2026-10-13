#!/usr/bin/env python3
"""Run NV-Reason-CT (native 3D CT VLM) on CT-SpatialVQA.

Unlike the slice-based harnesses, NV-Reason-CT consumes the raw NIfTI volume
directly: its processor reorients to LPS, resamples to 2 mm isotropic and takes
an anatomy-aware 192^3 crop (24x24x24 = 13,824 visual tokens).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing transformers. Install with: pip install transformers==5.6.2") from exc

THINK_END = "</think>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NV-Reason-CT on CT volumes.")
    parser.add_argument("--dataset", required=True, help="Input JSONL with case_id/question/answer.")
    parser.add_argument("--nifti-root", required=True, help="Root of raw NIfTI volumes (valid_fixed).")
    parser.add_argument("--output", required=True, help="Output JSONL for predictions.")
    parser.add_argument("--model-id", default="nvidia/NV-Reason-CT", help="HF model id.")
    parser.add_argument("--anatomy-region", default="chest", choices=["chest", "abdomen", "none"],
                        help="Anatomy-aware crop passed to the processor ('none' = centered crop).")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Let the model emit <think> reasoning before the answer.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N records (0 = all).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split the dataset into this many contiguous shards (for parallel SLURM jobs).")
    parser.add_argument("--shard-index", type=int, default=0, help="Which shard this process handles (0-based).")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error(f"--shard-index must be in [0, {args.num_shards})")
    return args


def derive_nifti_path(nifti_root: Path, case_id: str) -> Path:
    """Convert case_id like valid_1_a_1.nii.gz to valid_fixed path structure."""
    stem = case_id.replace(".nii.gz", "")
    subdir = stem.rsplit("_", 1)[0]  # valid_1_a
    base = subdir.rsplit("_", 1)[0] if "_" in subdir else subdir
    return nifti_root / base / subdir / case_id


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def cache_last_volume(image_processor_3d) -> None:
    """Memoize the most recent volume so consecutive questions on a case skip reloading.

    Resampling + anatomy cropping is CPU-heavy (seconds per volume) and the
    dataset has ~5.7 questions per volume, grouped by case.
    """
    original = image_processor_3d.load_image
    cache: dict = {}

    def load_image(filename, normalize_mode=None, anatomy_region=None):
        key = (str(filename), normalize_mode, anatomy_region)
        if key not in cache:
            cache.clear()
            cache[key] = original(filename, normalize_mode=normalize_mode, anatomy_region=anatomy_region)
        return cache[key]

    image_processor_3d.load_image = load_image


def split_reasoning(text: str) -> tuple[str, str, bool]:
    """Return (reasoning, answer, finished_thinking) from a raw generation."""
    if THINK_END not in text:
        return "", text.replace("<think>", "").strip(), False
    reasoning, answer = text.split(THINK_END, 1)
    return reasoning.replace("<think>", "").strip(), answer.strip(), True


def main() -> None:
    args = parse_args()
    nifti_root = Path(args.nifti_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anatomy_region = None if args.anatomy_region == "none" else args.anatomy_region

    processed = set()
    if args.resume and output_path.exists():
        for rec in iter_jsonl(output_path):
            processed.add((rec.get("case_id"), rec.get("question")))

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa",
    ).eval().to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    cache_last_volume(processor.image_processor_3d)

    total = sum(1 for _ in iter_jsonl(Path(args.dataset)))
    if args.limit and args.limit > 0:
        total = min(total, args.limit)
    # Contiguous shards: questions on the same case are consecutive in the dataset, so each
    # shard keeps the per-volume cache effective (round-robin would reload volumes constantly).
    shard_lo = total * args.shard_index // args.num_shards
    shard_hi = total * (args.shard_index + 1) // args.num_shards
    total = shard_hi - shard_lo

    with output_path.open("a") as f_out:
        # Slice to this shard before wrapping in tqdm so skipped rows do not inflate the bar.
        shard_iter = itertools.islice(enumerate(iter_jsonl(Path(args.dataset))), shard_lo, shard_hi)
        for idx, ex in tqdm(shard_iter, total=total, desc="Running NV-Reason-CT"):
            if args.limit and args.limit > 0 and idx >= args.limit:
                break
            case_id = ex.get("case_id")
            question = ex.get("question")
            answer = ex.get("answer")
            if (case_id, question) in processed:
                continue

            nifti_path = derive_nifti_path(nifti_root, case_id)
            if not nifti_path.exists():
                raise FileNotFoundError(f"Missing NIfTI: {nifti_path}")

            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            inputs = processor(
                text=prompt,
                images3d=[str(nifti_path)],
                anatomy_region=anatomy_region,
                return_tensors="pt",
            ).to(model.device)

            t0 = time.time()
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            if args.enable_thinking:
                reasoning, prediction, finished = split_reasoning(raw)
            else:
                reasoning, prediction, finished = "", raw.strip(), True

            record = dict(
                case_id=case_id,
                question=question,
                answer=answer,
                prediction=prediction,
                model_id=args.model_id,
                anatomy_region=args.anatomy_region,
                enable_thinking=args.enable_thinking,
                reasoning=reasoning,
                thinking_finished=finished,
                num_new_tokens=int(new_tokens.shape[1]),
                gen_seconds=round(time.time() - t0, 2),
            )
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()


if __name__ == "__main__":
    main()

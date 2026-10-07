#!/usr/bin/env python3
"""Run inference with fine-tuned Lingshu LoRA adapter on the test split.

Mirrors the original inference pipeline but loads the LoRA adapter on top
of the base model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from peft import PeftModel
except ImportError:
    raise SystemExit("Missing peft. Install with: pip install peft")

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise SystemExit("Missing qwen_vl_utils. Install with: pip install qwen-vl-utils")


DEFAULT_WINDOWS: List[Tuple[int, int]] = [
    (-1024, 1024),
    (-135, 215),
    (0, 80),
]

DEFAULT_INSTRUCTION = (
    "You are an instructor teaching medical students. You are analyzing a "
    "contiguous block of CT slices. Please review the slices provided below carefully."
)

DEFAULT_QUERY_SUFFIX = (
    "\n\nBased on the visual evidence in the slices provided above, "
    "answer the question below. Provide concise reasoning and conclude with a final answer."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned Lingshu LoRA")
    p.add_argument("--test-jsonl", type=Path, required=True)
    p.add_argument("--nifti-root", type=Path, default=None)
    p.add_argument("--npz-root", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-model-id", type=str, default="lingshu-medical-mllm/Lingshu-7B")
    p.add_argument("--adapter-path", type=Path, required=True,
                    help="Path to LoRA adapter (e.g. checkpoints/lingshu-lora/final)")
    p.add_argument("--num-slices", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


# ── Slice loading (same as training) ────────────────────────────────

def window_slice(s: np.ndarray, w: Tuple[int, int]) -> np.ndarray:
    lo, hi = w
    x = np.clip(s, lo, hi).astype(np.float32)
    return np.round((x - lo) / (hi - lo) * 255).astype(np.uint8)


def make_rgb(s: np.ndarray) -> np.ndarray:
    return np.stack([window_slice(s, w) for w in DEFAULT_WINDOWS], axis=-1)


def sample_indices(n: int, k: int) -> List[int]:
    if k <= 0: return []
    if n <= k: return list(range(n))
    return [int(round(i)) for i in np.linspace(0, n - 1, k)]


def derive_npz_path(root: Path, cid: str) -> Path:
    stem = cid.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        return root / "_".join(tokens[:2]) / "_".join(tokens[:3]) / f"{stem}.npz"
    return root / f"{stem}.npz"


def derive_nifti_path(root: Path, cid: str) -> Path:
    stem = cid.replace(".nii.gz", "")
    subdir = stem.rsplit("_", 1)[0]
    base = subdir.rsplit("_", 1)[0] if "_" in subdir else subdir
    return root / base / subdir / cid


def load_slices(cid: str, npz_root: Path | None, nifti_root: Path | None, n: int) -> List[np.ndarray]:
    if npz_root:
        p = derive_npz_path(npz_root, cid)
        if p.exists():
            data = np.load(p)
            slices = data["slices"]
            idxs = sample_indices(slices.shape[0], n)
            return [slices[i] for i in idxs]
    if nifti_root:
        import nibabel as nib
        p = derive_nifti_path(nifti_root, cid)
        if p.exists():
            vol = np.asarray(nib.load(str(p)).get_fdata())
            if vol.ndim == 3: vol = np.transpose(vol, (2, 1, 0))
            idxs = sample_indices(vol.shape[0], n)
            return [make_rgb(vol[i]) for i in idxs]
    raise FileNotFoundError(f"No volume for {cid}")


def build_messages(slices: List[np.ndarray], question: str) -> list:
    content = [{"type": "text", "text": DEFAULT_INSTRUCTION}]
    for i, rgb in enumerate(slices, 1):
        content.append({"type": "image", "image": Image.fromarray(rgb)})
        content.append({"type": "text", "text": f"SLICE {i}"})
    content.append({"type": "text", "text": f"{DEFAULT_QUERY_SUFFIX}\n\nQuestion: {question}"})
    return [{"role": "user", "content": content}]


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"Loading base model: {args.base_model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_id, torch_dtype=dtype, device_map={"": args.device},
    )

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, str(args.adapter_path))
    # Fold LoRA into the base weights: same outputs, but no extra LoRA activations
    # (32-slice prefill OOMs on a 32GB GPU otherwise)
    model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained(args.base_model_id, use_fast=True)

    # Resume support
    processed = set()
    if args.resume and args.output.exists():
        with args.output.open() as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                processed.add((rec.get("case_id"), rec.get("question")))

    records = []
    with args.test_jsonl.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.limit > 0:
        records = records[:args.limit]

    slice_cache = {}

    with args.output.open("a") as f_out:
        for rec in tqdm(records, desc="Evaluating Lingshu-LoRA"):
            cid = rec["case_id"]
            q = rec["question"]
            if (cid, q) in processed:
                continue

            if cid not in slice_cache:
                slice_cache[cid] = load_slices(cid, args.npz_root, args.nifti_root, args.num_slices)
            slices = slice_cache[cid]

            messages = build_messages(slices, q)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
            inputs = {k: v.to(args.device) for k, v in inputs.items() if hasattr(v, "to")}

            gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
            if args.temperature == 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = 0.001

            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            pred = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0].strip()

            result = dict(
                case_id=cid, question=q, answer=rec["answer"],
                prediction=pred, model_id=f"{args.base_model_id}+lora",
                adapter_path=str(args.adapter_path),
            )
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

    print(f"Done. Predictions saved to {args.output}")


if __name__ == "__main__":
    main()

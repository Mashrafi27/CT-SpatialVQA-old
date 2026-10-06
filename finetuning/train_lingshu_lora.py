#!/usr/bin/env python3
"""LoRA fine-tuning for Lingshu (Qwen2.5-VL 7B) on CT-SpatialVQA.

Loads preprocessed .npz slice packs (or raw NIfTI), builds Qwen2.5-VL
chat messages with multi-image input, and trains with LoRA using the
HuggingFace Trainer.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
except ImportError:
    raise SystemExit("Missing peft. Install with: pip install peft")

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise SystemExit("Missing qwen_vl_utils. Install with: pip install qwen-vl-utils")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CT windowing (same as inference) ────────────────────────────────
DEFAULT_WINDOWS: List[Tuple[int, int]] = [
    (-1024, 1024),  # wide
    (-135, 215),    # mediastinum
    (0, 80),        # brain
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
    p = argparse.ArgumentParser(description="LoRA fine-tune Lingshu on CT-SpatialVQA")

    # Data
    p.add_argument("--train-jsonl", type=Path, required=True,
                    help="Training split JSONL (case_id, question, answer).")
    p.add_argument("--nifti-root", type=Path, default=None,
                    help="Root of raw NIfTI volumes (fallback if no npz).")
    p.add_argument("--npz-root", type=Path, default=None,
                    help="Root of preprocessed .npz slice packs.")
    p.add_argument("--num-slices", type=int, default=8,
                    help="Number of slices per volume (fewer than inference to save VRAM).")

    # Model
    p.add_argument("--model-id", type=str, default="lingshu-medical-mllm/Lingshu-7B")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", type=str, nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"])

    # Training
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/lingshu-lora"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ── Volume / slice loading ──────────────────────────────────────────

def window_slice(slice_2d: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    lo, hi = window
    x = np.clip(slice_2d, lo, hi).astype(np.float32)
    x = (x - lo) / (hi - lo) * 255.0
    return np.round(x).astype(np.uint8)


def make_rgb(slice_2d: np.ndarray) -> np.ndarray:
    return np.stack([window_slice(slice_2d, w) for w in DEFAULT_WINDOWS], axis=-1)


def sample_indices(n_total: int, n_sample: int) -> List[int]:
    if n_sample <= 0:
        return []
    if n_total <= n_sample:
        return list(range(n_total))
    idxs = np.linspace(0, n_total - 1, n_sample)
    return [int(round(i)) for i in idxs]


def load_slices_npz(npz_path: Path, num_slices: int) -> List[np.ndarray]:
    data = np.load(npz_path)
    slices = data["slices"]  # (N, H, W, 3)
    idxs = sample_indices(slices.shape[0], num_slices)
    return [slices[i] for i in idxs]


def load_slices_nifti(nifti_path: Path, num_slices: int) -> List[np.ndarray]:
    import nibabel as nib
    vol = np.asarray(nib.load(str(nifti_path)).get_fdata())
    if vol.ndim == 3:
        vol = np.transpose(vol, (2, 1, 0))
    idxs = sample_indices(vol.shape[0], num_slices)
    return [make_rgb(vol[i]) for i in idxs]


def derive_npz_path(npz_root: Path, case_id: str) -> Path:
    stem = case_id.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])
        series = "_".join(tokens[:3])
        return npz_root / patient / series / f"{stem}.npz"
    return npz_root / f"{stem}.npz"


def derive_nifti_path(nifti_root: Path, case_id: str) -> Path:
    stem = case_id.replace(".nii.gz", "")
    subdir = stem.rsplit("_", 1)[0]
    base = subdir.rsplit("_", 1)[0] if "_" in subdir else subdir
    return nifti_root / base / subdir / case_id


# ── Dataset ─────────────────────────────────────────────────────────

class CTSpatialVQADataset(Dataset):
    """Loads QA pairs and returns tokenized inputs with labels."""

    def __init__(
        self,
        jsonl_path: Path,
        processor: AutoProcessor,
        npz_root: Path | None,
        nifti_root: Path | None,
        num_slices: int,
        max_seq_length: int,
    ):
        self.processor = processor
        self.npz_root = npz_root
        self.nifti_root = nifti_root
        self.num_slices = num_slices
        self.max_seq_length = max_seq_length

        # Load records and group by case_id for slice caching
        with jsonl_path.open() as f:
            self.records = [json.loads(l) for l in f if l.strip()]

        logger.info(f"Loaded {len(self.records)} training samples")

        # Pre-validate that we can find volumes for all case_ids
        case_ids = set(r["case_id"] for r in self.records)
        missing = []
        for cid in case_ids:
            if not self._find_volume_path(cid):
                missing.append(cid)
        if missing:
            logger.warning(f"{len(missing)} case_ids have no volume file (first 5): {missing[:5]}")

    def _find_volume_path(self, case_id: str) -> Path | None:
        if self.npz_root:
            p = derive_npz_path(self.npz_root, case_id)
            if p.exists():
                return p
        if self.nifti_root:
            p = derive_nifti_path(self.nifti_root, case_id)
            if p.exists():
                return p
        return None

    def _load_slices(self, case_id: str) -> List[np.ndarray]:
        if self.npz_root:
            p = derive_npz_path(self.npz_root, case_id)
            if p.exists():
                return load_slices_npz(p, self.num_slices)
        if self.nifti_root:
            p = derive_nifti_path(self.nifti_root, case_id)
            if p.exists():
                return load_slices_nifti(p, self.num_slices)
        raise FileNotFoundError(f"No volume found for {case_id}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        case_id = rec["case_id"]
        question = rec["question"]
        answer = rec["answer"]

        slice_rgbs = self._load_slices(case_id)

        # Build user message (same format as inference)
        user_content = [{"type": "text", "text": DEFAULT_INSTRUCTION}]
        for i, rgb in enumerate(slice_rgbs, 1):
            user_content.append({"type": "image", "image": Image.fromarray(rgb)})
            user_content.append({"type": "text", "text": f"SLICE {i}"})
        user_content.append({
            "type": "text",
            "text": f"{DEFAULT_QUERY_SUFFIX}\n\nQuestion: {question}",
        })

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]

        # Tokenize the full conversation (prompt + answer)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        # Tokenize prompt only (to find where labels start)
        prompt_messages = [{"role": "user", "content": user_content}]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )

        # Process with vision (no truncation — image tokens can't be split)
        image_inputs, video_inputs = process_vision_info(messages)
        full_inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_messages)
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=prompt_image_inputs,
            videos=prompt_video_inputs,
            padding=False,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Labels: -100 for prompt tokens, actual ids for answer tokens
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # Pass through pixel values and image grid info if present
        for key in full_inputs:
            if key not in result and key not in ("text",):
                val = full_inputs[key]
                if isinstance(val, torch.Tensor):
                    result[key] = val.squeeze(0)
                else:
                    result[key] = val

        return result


# ── Collator ────────────────────────────────────────────────────────

class VLCollator:
    """Pads variable-length inputs for the VL model."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        # Separate tensor fields from list fields
        tensor_keys = set()
        list_keys = set()
        for sample in batch:
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    tensor_keys.add(k)
                else:
                    list_keys.add(k)

        result = {}

        # Pad tensor fields
        for key in tensor_keys:
            tensors = [s[key] for s in batch]
            if key == "labels":
                result[key] = torch.nn.utils.rnn.pad_sequence(
                    tensors, batch_first=True, padding_value=-100
                )
            elif key == "attention_mask":
                result[key] = torch.nn.utils.rnn.pad_sequence(
                    tensors, batch_first=True, padding_value=0
                )
            elif key == "input_ids":
                result[key] = torch.nn.utils.rnn.pad_sequence(
                    tensors, batch_first=True, padding_value=self.pad_token_id
                )
            elif key == "pixel_values":
                # Pixel values: concatenate along batch dim
                result[key] = torch.cat(tensors, dim=0)
            else:
                # Try padding, fall back to cat/stack
                try:
                    result[key] = torch.nn.utils.rnn.pad_sequence(
                        tensors, batch_first=True, padding_value=0
                    )
                except Exception:
                    try:
                        result[key] = torch.cat(tensors, dim=0)
                    except Exception:
                        result[key] = tensors

        # List fields: concatenate
        for key in list_keys:
            vals = [s[key] for s in batch]
            if isinstance(vals[0], list):
                result[key] = sum(vals, [])
            else:
                result[key] = vals

        return result


# ── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger.info(f"Loading model: {args.model_id}")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # QLoRA: 4-bit quantization to fit 7B model on single 32GB GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    logger.info("Loaded model with 4-bit quantization (QLoRA)")

    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)

    # Ensure pad token
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    logger.info(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
    )
    # Save device map before PEFT wrapping
    base_device_map = getattr(model, "hf_device_map", None)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Propagate quantization flags to PeftModel so accelerate skips .to(device)
    model.is_loaded_in_4bit = True
    if base_device_map is not None:
        model.hf_device_map = base_device_map

    # Dataset
    logger.info("Loading dataset...")
    train_dataset = CTSpatialVQADataset(
        jsonl_path=args.train_jsonl,
        processor=processor,
        npz_root=args.npz_root,
        nifti_root=args.nifti_root,
        num_slices=args.num_slices,
        max_seq_length=args.max_seq_length,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="lingshu-lora-ct-spatialvqa",
    )

    collator = VLCollator(pad_token_id=processor.tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final adapter
    final_dir = args.output_dir / "final"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    logger.info(f"Saved final adapter to {final_dir}")


if __name__ == "__main__":
    main()

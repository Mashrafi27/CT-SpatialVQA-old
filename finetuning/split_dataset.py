#!/usr/bin/env python3
"""Split CT-SpatialVQA dataset into train/test at the volume level.

Ensures no CT volume appears in both splits. Outputs:
  - train.jsonl / test.jsonl  (case_id, question, answer)
  - split_info.json           (metadata: counts, volume lists, seed)

Also filters existing per-model prediction files to the test split
so baseline (pre-finetune) numbers can be computed on the same subset.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Volume-level train/test split")
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "dataset"
        / "ct_spatialvqa"
        / "spatial_qa_filtered_full.json",
        help="Path to the full QA JSON (keyed by volume).",
    )
    p.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "benchmarking"
        / "reports"
        / "predictions",
        help="Directory with per-model *_predictions_full.jsonl files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "splits",
        help="Output directory for split files.",
    )
    p.add_argument("--test-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def split_volumes(volumes: list[str], test_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(volumes)
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_ratio)
    return shuffled[n_test:], shuffled[:n_test]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def filter_predictions(pred_path: Path, test_case_ids: set[str], out_path: Path) -> int:
    """Filter a predictions JSONL to only test-split entries. Returns count."""
    count = 0
    with pred_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            # predictions use image_path containing the case_id, or case_id directly
            case_id = rec.get("case_id")
            if case_id is None:
                # fall back: extract from image_path
                img = rec.get("image_path", "")
                for cid in test_case_ids:
                    if cid.replace(".nii.gz", "") in img:
                        case_id = cid
                        break
            if case_id in test_case_ids:
                f_out.write(line)
                count += 1
    return count


def main() -> None:
    args = parse_args()

    with args.dataset.open() as f:
        data = json.load(f)

    volumes = sorted(data.keys())
    train_vols, test_vols = split_volumes(volumes, args.test_ratio, args.seed)
    train_set, test_set = set(train_vols), set(test_vols)

    # Build JSONL records
    train_records, test_records = [], []
    for vol, info in data.items():
        for qa in info["qa_pairs"]:
            rec = {"case_id": vol, "question": qa["question"], "answer": qa["answer"]}
            if vol in train_set:
                train_records.append(rec)
            else:
                test_records.append(rec)

    # Write splits
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_records, out / "train.jsonl")
    write_jsonl(test_records, out / "test.jsonl")

    # Filter predictions to test split
    test_case_ids = test_set
    baseline_dir = out / "baseline_predictions"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    pred_counts = {}
    if args.predictions_dir.exists():
        for pred_file in sorted(args.predictions_dir.glob("*_predictions_full.jsonl")):
            model_name = pred_file.name.replace("_predictions_full.jsonl", "")
            out_pred = baseline_dir / f"{model_name}_predictions_test.jsonl"
            count = filter_predictions(pred_file, test_case_ids, out_pred)
            pred_counts[model_name] = count
            print(f"  {model_name}: {count} test predictions")

    # Write metadata
    split_info = {
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "total_volumes": len(volumes),
        "train_volumes": len(train_vols),
        "test_volumes": len(test_vols),
        "train_qa_pairs": len(train_records),
        "test_qa_pairs": len(test_records),
        "train_volume_ids": train_vols,
        "test_volume_ids": test_vols,
        "baseline_prediction_counts": pred_counts,
    }
    with (out / "split_info.json").open("w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSplit complete:")
    print(f"  Train: {len(train_vols)} volumes, {len(train_records)} QA pairs")
    print(f"  Test:  {len(test_vols)} volumes, {len(test_records)} QA pairs")
    print(f"  Output: {out}")


if __name__ == "__main__":
    main()

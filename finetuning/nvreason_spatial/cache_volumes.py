#!/usr/bin/env python3
"""Crop CT-RATE train volumes the way NV-Reason-CT does and cache the model input, so training never reads raw scans.

For every volume in --volumes, once its raw .nii.gz is on disk: run the model's own ImageLoader3D (LPS orientation,
2 mm spacing, pad, chest crop to 192^3, transpose, CT normalisation to [-1, 1]) and save the [1, 192, 192, 192] tensor as
float16 .npy (~14 MB) in --cache-dir. After a successful save, the raw file is deleted only when --delete-raw is set AND
the file is listed in --owned (files our download script fetched; scans that were already on disk are never deleted).

Axis order of the cached tensor: [channel, z (inferior->superior), y (anterior->posterior), x (right->left)]
(LPS, then transpose (0, 3, 2, 1)). A left-right flip is therefore a flip of the last axis.

--wait keeps polling for new downloads until every listed volume is cached (pairs with download_ctrate_volumes.sh).
Failures are logged to <cache-dir>/failed.txt and skipped.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np


def raw_path(root: Path, volume: str) -> Path:
    scan = volume.rsplit("_", 1)[0]  # train_123_a_1.nii.gz -> train_123_a
    return root / "dataset" / "train_fixed" / scan.rsplit("_", 1)[0] / scan / volume


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True, help="Folder that mirrors the HF repo (contains dataset/).")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--owned", type=Path, default=None, help="Raw files our download created (may be deleted).")
    ap.add_argument("--delete-raw", action="store_true")
    ap.add_argument("--wait", action="store_true", help="Poll for downloads until all volumes are cached.")
    ap.add_argument("--snapshot", default=None)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    snap = a.snapshot or glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--NV-Reason-CT/snapshots/*"))[0]
    sys.path.insert(0, snap)
    from processor import ImageLoader3D  # the model's own preprocessing code
    import json
    cfg = json.load(open(f"{snap}/image_processor_3d/preprocessor_config.json"))
    loader = ImageLoader3D(**{k: v for k, v in cfg.items() if k != "image_processor_type"})

    vols = [v.strip() for v in open(a.volumes) if v.strip()]
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    failed_log = a.cache_dir / "failed.txt"
    failed = set(failed_log.read_text().split()) if failed_log.exists() else set()
    t0, n_done = time.time(), 0
    while True:
        owned = set(a.owned.read_text().split()) if a.owned and a.owned.exists() else set()
        if a.delete_raw:  # also catches files cropped before the download script listed them as ours
            for v in vols:
                src = raw_path(a.data_root, v)
                if str(src) in owned and src.exists() and (a.cache_dir / v.replace(".nii.gz", ".npy")).exists():
                    src.unlink()
        todo = [v for v in vols if v not in failed and not (a.cache_dir / v.replace(".nii.gz", ".npy")).exists()]
        ready = [v for v in todo if raw_path(a.data_root, v).exists()]
        if not todo:
            break
        if not ready:
            if not a.wait:
                break
            time.sleep(60)
            continue
        for v in ready:
            src, dst = raw_path(a.data_root, v), a.cache_dir / v.replace(".nii.gz", ".npy")
            try:
                x = loader.load_image(str(src), normalize_mode="ct", anatomy_region="chest")
                assert tuple(x.shape) == (1, 192, 192, 192), x.shape
                tmp = dst.with_suffix(".tmp.npy")
                np.save(tmp, x.numpy().astype(np.float16))
                os.replace(tmp, dst)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {v}: {type(e).__name__}: {e}", flush=True)
                failed.add(v)
                with open(failed_log, "a") as f:
                    f.write(v + "\n")
                continue
            n_done += 1
            if n_done % 25 == 0:
                print(f"{n_done} cached, {len(todo) - n_done} left, {(time.time() - t0) / n_done:.1f} s/volume", flush=True)
    n_cached = len(list(a.cache_dir.glob("*.npy")))
    print(f"CACHE_DONE {n_cached} cached, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()

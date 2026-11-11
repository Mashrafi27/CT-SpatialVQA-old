#!/usr/bin/env python3
"""Build the spatial fine-tuning set for NV-Reason-CT from CT-RATE TRAIN reports (pilot, task 7).

Every report (Findings_EN + Impressions_EN) is split into sentences. A sentence is "spatial" when it names exactly one
pole of a direction (left/right, upper/lower, front/back, central/peripheral) and is not a normal-anatomy filler
("... are normal", "no ... was observed"). Each selected volume becomes one training item:

  prompt  "Where are the abnormal findings in this CT? Give their side and location."
  target  its spatial sentences (up to --max-sentences), in report order

and, when the target says left or right anywhere, a flipped twin: the same volume flipped left-right at load time
(flip_lr=true) with every left/right word swapped in the target. The flip pairs teach that side must come from the image.

Selection: one reconstruction per scan (the lowest _<n>), scans with at least one one-sided left/right sentence first,
left- and right-sided items balanced, capped at --max-volumes. Seed 0.

Outputs (--out-dir):
  train.jsonl          {volume, case_id, prompt, target, flip_lr, directions, sentences}
  volumes.txt          one VolumeName per line, for the download step
  stats.json           counts per direction and pole
Never point --reports at validation_reports.csv for a real run: CT-SpatialVQA is built from the validation scans.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

PROMPT = "Where are the abnormal findings in this CT? Give their side and location."
POLES = {
    "lr": (r"\bleft\b", r"\bright\b"),
    "si": (r"\b(upper|superior|apical|apex|apices)\b", r"\b(lower|inferior|basal|bases?)\b"),
    "ap": (r"\b(anterior|anteriorly|antero\w+)\b", r"\b(posterior|posteriorly|postero\w+)\b"),
    "ml": (r"\b(medial|central|centrally|perihilar)\b", r"\b(lateral|peripheral|peripherally|subpleural)\b"),
}
BOTH_RE = re.compile(r"\b(both|bilateral|bilaterally)\b", re.I)
NORMAL_RE = re.compile(r"\b(normal|unremarkable|no |not (observed|detected|seen)|were not|was not|within normal)", re.I)
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text or "") if len(s.strip()) > 8]


def tag(sent: str) -> dict:
    """{axis: 'A' | 'B'} for every direction where the sentence names exactly one pole."""
    out = {}
    for axis, (a, b) in POLES.items():
        ha, hb = bool(re.search(a, sent, re.I)), bool(re.search(b, sent, re.I))
        if ha != hb and not (axis == "lr" and BOTH_RE.search(sent)):
            out[axis] = "A" if ha else "B"
    return out


def swap_lr(text: str) -> str:
    def rep(m):
        w = m.group(0)
        new = "right" if w.lower() == "left" else "left"
        return new.capitalize() if w[0].isupper() else new
    return re.sub(r"\b(left|right)\b", rep, text, flags=re.I)


def scan_id(volume: str) -> str:
    return re.sub(r"_\d+\.nii\.gz$", "", volume)  # train_123_a_1.nii.gz -> train_123_a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, required=True, help="CT-RATE train_reports.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-volumes", type=int, default=4000)
    ap.add_argument("--max-sentences", type=int, default=4)
    ap.add_argument("--allow-validation", action="store_true", help="Only for testing this script.")
    a = ap.parse_args()
    if "valid" in a.reports.name and not a.allow_validation:
        raise SystemExit("validation reports are the benchmark's source; pass --allow-validation only to test")
    rng = random.Random(0)

    by_scan = {}
    for r in csv.DictReader(open(a.reports, newline="")):
        s = scan_id(r["VolumeName"])
        if s not in by_scan or r["VolumeName"] < by_scan[s]["VolumeName"]:
            by_scan[s] = r

    items, stats = [], Counter()
    for s, r in sorted(by_scan.items()):
        sel = []
        for sent in sentences(r.get("Findings_EN", "")) + sentences(r.get("Impressions_EN", "")):
            t = tag(sent)
            if t and not NORMAL_RE.search(sent) and sent not in [x for x, _ in sel]:
                sel.append((sent, t))
        if not sel:
            continue
        sel = sel[: a.max_sentences]
        lr = [t["lr"] for _, t in sel if "lr" in t]
        items.append(dict(volume=r["VolumeName"], sel=sel, lr_side=(lr[0] if len(set(lr)) == 1 else None) if lr else None))

    with_lr = [it for it in items if it["lr_side"]]
    left = [it for it in with_lr if it["lr_side"] == "A"]
    right = [it for it in with_lr if it["lr_side"] == "B"]
    rng.shuffle(left), rng.shuffle(right)
    k = min(len(left), len(right), a.max_volumes // 2)
    chosen = left[:k] + right[:k]
    rest = [it for it in items if it not in chosen]
    rng.shuffle(rest)
    chosen += rest[: max(0, a.max_volumes - len(chosen))]
    chosen.sort(key=lambda it: it["volume"])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "train.jsonl", "w") as f:
        for it in chosen:
            sents = [x for x, _ in it["sel"]]
            dirs = sorted({ax for _, t in it["sel"] for ax in t})
            for _, t in it["sel"]:
                for ax, p in t.items():
                    stats[f"{ax}_{p}"] += 1
            row = dict(volume=it["volume"], case_id=it["volume"], prompt=PROMPT, target=" ".join(sents),
                       flip_lr=False, directions=dirs, sentences=sents)
            f.write(json.dumps(row) + "\n")
            stats["items"] += 1
            if re.search(r"\b(left|right)\b", row["target"], re.I):  # mirroring + swapping words is always consistent
                f.write(json.dumps(row | dict(target=swap_lr(row["target"]), flip_lr=True,
                                              sentences=[swap_lr(x) for x in sents])) + "\n")
                stats["items_flipped"] += 1
    (a.out_dir / "volumes.txt").write_text("".join(it["volume"] + "\n" for it in chosen))
    stats.update(scans_in_reports=len(by_scan), scans_with_spatial_sentences=len(items),
                 scans_one_sided_lr=len(with_lr), left_sided=len(left), right_sided=len(right), volumes=len(chosen))
    (a.out_dir / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1))
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()

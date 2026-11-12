#!/usr/bin/env python3
"""Does a CT carry a left/right lean in its image tokens, and do NV-Reason-CT's answers follow it?

Reads readout_probe.py --all-questions outputs (<stem>.shard*.npz/json); which-side questions with a one-sided GT (n=592).

1 Attention laterality. attn_img is the last prompt position's attention over the 13,824 visual tokens (mean over heads,
  renormalised), one map per full-attention layer. Token order is the 24^3 grid (z, y, x) of the LPS volume after the
  processor's transpose, x increasing from the patient's right to left, so x >= 12 is the patient's left half.
  LI = (left mass - right mass) / total, per question and layer (+1 = all attention on the left half).
  Reported: AUROC of LI for GT side and for the model's answer, per layer and for the mean over layers; LI split by
  (GT, answer).
2 Per-CT lean. At full-attention layers 3 and 7 the attention-weighted visual vector is ~identical for all questions on a
  CT (cosine 0.999), i.e. a question-independent image summary. Out-of-fold logistic probes (5-fold GroupKFold by CT)
  map it to the model's answer and to the GT side; the out-of-fold P(Left) of the answer probe is the CT's "lean".
  Reported: probe balanced accuracies, and how often the answer / the GT agree with the lean, split by whether GT and
  lean agree.
3 Probe on the attention-weighted visual vector at every full-attention layer -> model's answer (the missing dashed-blue
  line), next to -> GT side on the same questions.

Writes <out-dir>/side_lean.json and side_lean.png.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    meta, V, ATT = [], [], []
    for f in sorted(glob.glob(f"{a.stem}.shard*.npz")):
        m = json.load(open(f[:-4] + ".json"))
        d = np.load(f)
        idx = [i for i, r in enumerate(m) if r["side_question"] and r["gt_side"] in ("L", "R")]
        meta += [m[i] for i in idx]
        V.append(d["img_vec"][idx].astype(np.float32))
        ATT.append(d["attn_img"][idx].astype(np.float32))
    V, ATT = np.concatenate(V), np.concatenate(ATT)  # [n, 8, 2560], [n, 8, 13824]
    layers = meta[0]["layers"]
    gt = np.array([r["gt_side"] for r in meta])
    said = np.array([{"Left": "L", "Right": "R"}.get(r["top_token"], "-") for r in meta])
    case = np.array([r["case_id"] for r in meta])
    lr = np.isin(said, ["L", "R"])
    S = {"n": len(meta), "n_answer_L_or_R": int(lr.sum()), "layers": layers}

    # ---- 1 attention laterality
    grid = ATT.reshape(len(meta), len(layers), 24, 24, 24)
    left, right = grid[..., 12:].sum((2, 3, 4)), grid[..., :12].sum((2, 3, 4))
    LI = (left - right) / (left + right + 1e-12)  # [n, layers]
    LIm = LI.mean(1)
    auc = lambda s, y: round(float(roc_auc_score(y, s)), 3)  # noqa: E731
    S["attention_laterality"] = {
        "auroc_gt_per_layer": [auc(LI[:, j], gt == "L") for j in range(len(layers))],
        "auroc_answer_per_layer": [auc(LI[lr, j], said[lr] == "L") for j in range(len(layers))],
        "auroc_gt_mean": auc(LIm, gt == "L"), "auroc_answer_mean": auc(LIm[lr], said[lr] == "L"),
        "mean_LI_by_gt_answer": {f"GT {g} / answer {s}": [int(((gt == g) & (said == s)).sum()),
                                                         round(float(LIm[(gt == g) & (said == s)].mean()), 4)]
                                 for g in "LR" for s in "LR" if ((gt == g) & (said == s)).any()},
    }

    # ---- 2 per-CT lean from the (question-independent) early visual vector
    gkf = GroupKFold(n_splits=5)
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))  # noqa: E731
    lean = {}
    for j, l in enumerate(layers[:2]):
        X = V[:, j]
        ba_ans = float(cross_val_score(clf(), X[lr], said[lr] == "L", groups=case[lr], cv=gkf, scoring="balanced_accuracy").mean())
        ba_gt = float(cross_val_score(clf(), X, gt == "L", groups=case, cv=gkf, scoring="balanced_accuracy").mean())
        p = cross_val_predict(clf(), X[lr], said[lr] == "L", groups=case[lr], cv=gkf, method="predict_proba")[:, 1]
        lean_side = np.where(p >= 0.5, "L", "R")
        g, s = gt[lr], said[lr]
        agree = g == lean_side
        lean[str(l)] = {
            "probe_to_answer_balanced_acc": round(ba_ans, 3), "probe_to_gt_balanced_acc": round(ba_gt, 3),
            "answer_follows_lean": round(float((s == lean_side).mean()), 3),
            "gt_matches_lean": round(float(agree.mean()), 3),
            "when_gt_matches_lean": {"n": int(agree.sum()), "answer_correct": round(float((s == g)[agree].mean()), 3)},
            "when_gt_opposes_lean": {"n": int((~agree).sum()), "answer_correct": round(float((s == g)[~agree].mean()), 3),
                                     "answer_follows_lean": round(float((s == lean_side)[~agree].mean()), 3)},
        }
    S["ct_lean"] = lean

    # ---- 3 visual vector -> answer vs -> GT at every full-attention layer
    S["visual_vector_probe"] = {
        "to_answer": [round(float(cross_val_score(clf(), V[lr, j], said[lr] == "L", groups=case[lr], cv=gkf,
                                                  scoring="balanced_accuracy").mean()), 3) for j in range(len(layers))],
        "to_gt": [round(float(cross_val_score(clf(), V[:, j], gt == "L", groups=case, cv=gkf,
                                              scoring="balanced_accuracy").mean()), 3) for j in range(len(layers))],
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "side_lean.json").write_text(json.dumps(S, indent=1))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    al = S["attention_laterality"]
    axs[0].plot(layers, al["auroc_answer_per_layer"], "o-", color="tab:blue", label="LI -> model answer")
    axs[0].plot(layers, al["auroc_gt_per_layer"], "o-", color="gray", label="LI -> GT side")
    axs[0].axhline(0.5, color="lightgray", ls=":")
    axs[0].set(xlabel="full-attention layer", ylabel="AUROC", title="Attention laterality index (left - right half)")
    axs[0].legend()
    vp = S["visual_vector_probe"]
    axs[1].plot(layers, vp["to_answer"], "o--", color="tab:blue", label="v_l -> model answer")
    axs[1].plot(layers, vp["to_gt"], "o--", color="gray", label="v_l -> GT side")
    axs[1].axhline(0.5, color="lightgray", ls=":")
    axs[1].set(xlabel="full-attention layer", ylabel="balanced accuracy", title="Probe on attention-weighted visual vector")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(a.out_dir / "side_lean.png", dpi=120)
    print(json.dumps(S, indent=1))
    print("SIDE_LEAN_DONE")


if __name__ == "__main__":
    main()

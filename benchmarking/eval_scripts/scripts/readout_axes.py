#!/usr/bin/env python3
"""Readout analysis per spatial axis (left/right, upper/lower, anterior/posterior, central/peripheral) for NV-Reason-CT.

Reuses the readout_probe.py --all-questions outputs (<stem>.shard*.npz/json); no new forward pass. Two steps:

  1 lens (nvreason env, needs torch + the cached checkpoint):
        readout_axes.py lens --stem <stem> --out <stem>_axis_lens.npz
    Logit lens for the axis words: final norm + LM head (rows of the first token of each word only) applied to the
    stored readout state of every layer. Last layer uses the state as is (it is already normed), like readout_probe.py.
    Checks itself against the stored Left/Right lens and stops if they disagree.

  2 analyse (analysis venv, sklearn):
        readout_axes.py analyse --stem <stem> --lens <stem>_axis_lens.npz --out-dir <dir> \
            --judge-image <qwen eval json> --judge-blind <qwen eval json>
    For each axis, the questions whose question text does not give the answer away (it names neither pole, or both)
    and whose GT answer names exactly one pole:
      text_only        TF-IDF + logistic regression on the question text alone -> GT pole (control: how much of a
                       probe's accuracy the wording alone explains)
      probe            readout state per layer -> GT pole; question-weighted image vector per attention layer -> GT pole
      lens             per layer, AUC of logit(pole A) - logit(pole B) for separating GT A from GT B (all questions,
                       and the forced-choice "A or B?" subset), plus mean curves by GT pole
      leak             within-CT pairs: cosine of question-weighted image vectors / attention maps, same pole vs
                       opposite pole, different-CT pairs as baseline
      judge            Qwen-judge accuracy on the same questions, image vs blind
      mirror           image-token similarity to the mirror twin along this axis (from <stem>_mirror.json)
    All cross-validation is 5-fold GroupKFold by patient (valid_<n>), so scans of one patient never straddle folds.
    'lr_whichside' repeats the original which-side set (592 questions) so its probe can be checked against text_only.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SIDE_RE = re.compile(r"\b(which side|left or right|right or left|laterality)\b", re.I)
BOTH_RE = re.compile(r"\b(both|bilateral|bilaterally)\b", re.I)
# pole regexes (A, B) and the lens word for each pole (first token must be distinct and not a common answer opener)
AXES = {
    "lr": (r"\bleft\b", r"\bright\b", "Left", "Right"),
    "si": (r"\b(upper|superior|apical|apex|apices|cranial)\b", r"\b(lower|inferior|basal|bases?|caudal)\b", "Upper", "Lower"),
    "ap": (r"\b(anterior|anteriorly|ventral|antero\w+)\b", r"\b(posterior|posteriorly|dorsal|postero\w+)\b", "Ant", "Post"),
    "ml": (r"\b(medial|medially|central|centrally|perihilar|parahilar)\b",
           r"\b(lateral|laterally|peripheral|peripherally|subpleural)\b", "Central", "Peripheral"),
}
LENS_WORDS = ["Left", "Right", "Upper", "Lower", "Ant", "Post", "Central", "Peripheral", "Yes", "No"]
MIRROR_KEY = {"lr": "lr_mirror", "si": "si_mirror", "ap": "ap_mirror"}


def shard_files(stem: str) -> list[str]:
    return sorted(glob.glob(f"{stem}.shard*.npz")) or [f"{stem}.npz"]


def load_meta(stem: str) -> list[dict]:
    meta = []
    for f in shard_files(stem):
        meta += json.load(open(f[:-4] + ".json"))
    return meta


def load_arrays(stem: str, keys: list[str]) -> dict:
    parts = defaultdict(list)
    for f in shard_files(stem):
        d = np.load(f)
        n = len(json.load(open(f[:-4] + ".json")))
        for k in keys:
            parts[k].append(d[k][:n])
    return {k: np.concatenate(v) for k, v in parts.items()}


# ---------------------------------------------------------------- step 1: logit lens for the axis words
def cmd_lens(a) -> None:
    import os
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    from safetensors import safe_open
    from transformers import AutoTokenizer

    snap = a.snapshot or glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--NV-Reason-CT/snapshots/*"))[0]
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in LENS_WORDS]
    assert len(set(ids)) == len(ids), ids
    cfg = json.load(open(f"{snap}/config.json"))
    eps = cfg.get("text_config", cfg).get("rms_norm_eps", 1e-6)
    with safe_open(f"{snap}/model.safetensors", "pt") as g:
        W = g.get_slice("lm_head.weight")[ids].float()  # [words, d]
        nw = g.get_tensor("model.language_model.norm.weight").float()

    parts, stored = [], []
    for f in shard_files(a.stem):
        d = np.load(f)
        n = len(json.load(open(f[:-4] + ".json")))
        parts.append(d["readout"][:n])
        stored.append(d["lens"][:n])
    R = torch.from_numpy(np.concatenate(parts)).float()  # [n, layers, d]
    S = np.concatenate(stored)  # [n, layers, 6] log-probs of Left Right Both Bilateral No Yes
    ref = S[:, :-1, 0] - S[:, :-1, 1]  # stored Left - Right, all but the last layer

    def lens(scale):
        x = R[:, :-1]
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * scale
        return torch.cat([x @ W.T, R[:, -1:] @ W.T], 1).numpy()  # last layer: state is already normed

    best = None
    for name, scale in [("w", nw), ("1+w", 1 + nw)]:  # Qwen3.5 RMSNorm may scale by (1 + weight)
        out = lens(scale)
        err = float(np.abs((out[:, :-1, 0] - out[:, :-1, 1]) - ref).mean())
        print(f"norm scale {name}: mean |diff - stored| = {err:.4f} (stored mean |L-R| {np.abs(ref).mean():.3f})")
        if best is None or err < best[0]:
            best = (err, name, out)
    err, name, out = best
    last_err = float(np.abs((out[:, -1, 0] - out[:, -1, 1]) - (S[:, -1, 0] - S[:, -1, 1])).mean())
    print(f"using {name}; last layer mean |diff - stored| = {last_err:.4f}")
    assert err < 0.05 * np.abs(ref).mean() + 0.05, "lens does not reproduce the stored Left/Right lens"
    np.savez_compressed(a.out, logits=out.astype(np.float32), words=np.array(LENS_WORDS), norm_scale=name)
    print(f"wrote {a.out} {out.shape}")


# ---------------------------------------------------------------- step 2: analysis
def patient(case_id: str) -> str:
    return re.sub(r"_[a-z]+_\d+\.nii\.gz$", "", case_id)


def poles(text: str, axis: str) -> tuple[bool, bool]:
    A, B = AXES[axis][:2]
    return bool(re.search(A, text, re.I)), bool(re.search(B, text, re.I))


def select(meta: list[dict], axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices, labels (1 = pole A) and forced-choice flags for one axis."""
    idx, y, fc = [], [], []
    for i, m in enumerate(meta):
        if axis == "lr_whichside":
            if m.get("side_question") and m["gt_side"] in ("L", "R"):
                idx.append(i), y.append(int(m["gt_side"] == "L")), fc.append(True)
            continue
        qa, qb = poles(m["question"], axis)
        if qa != qb:  # the question names one pole, so the wording gives the answer away
            continue
        aa, ab = poles(m["answer"], axis)
        if aa == ab or (axis == "lr" and BOTH_RE.search(m["answer"])):
            continue
        idx.append(i), y.append(int(aa)), fc.append(qa and qb)
    return np.array(idx), np.array(y), np.array(fc, bool)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def cmd_analyse(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    a.out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(a.stem)
    X = load_arrays(a.stem, ["readout", "img_vec", "attn_img"])
    LG = np.load(a.lens)["logits"]
    assert LG.shape[0] == len(meta)
    mirror = {}
    for f in shard_files(a.stem):
        mirror.update(json.load(open(f[:-4] + "_mirror.json")))
    layers = meta[0]["layers"]
    case = np.array([m["case_id"] for m in meta])
    pat = np.array([patient(c) for c in case])

    def judge(path):
        if not path:
            return {}
        out = {}
        for r in json.load(open(path)):
            out.setdefault((r["case_id"], r["question"]), str(r["is_correct"]).lower() == "true")
        return out
    J = {"image": judge(a.judge_image), "blind": judge(a.judge_blind)}

    Vn = X["img_vec"].astype(np.float32)
    Vn /= np.linalg.norm(Vn, axis=-1, keepdims=True) + 1e-12
    Att = X["attn_img"].astype(np.float32).mean(1)
    Att /= np.linalg.norm(Att, axis=-1, keepdims=True) + 1e-12
    rng = np.random.default_rng(0)
    gkf = GroupKFold(n_splits=5)
    S = {"n_questions": len(meta), "grouping": "patient", "axes": {}}

    for axis in ["lr_whichside", "lr", "si", "ap", "ml"]:
        ax_ = "lr" if axis.startswith("lr") else axis
        idx, y, fc = select(meta, axis)
        groups = pat[idx]
        print(f"== {axis}: n={len(idx)} (pole A {y.sum()}, B {len(y) - y.sum()}, forced-choice {fc.sum()})", flush=True)
        res = {"n": int(len(idx)), "pole_A": int(y.sum()), "pole_B": int(len(y) - y.sum()), "forced_choice": int(fc.sum())}

        def cv(clf, F):
            return round(float(cross_val_score(clf, F, y, groups=groups, cv=gkf, scoring="balanced_accuracy",
                                               n_jobs=5).mean()), 3)

        qtext = [meta[i]["question"] for i in idx]
        res["text_only"] = cv(make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                            LogisticRegression(max_iter=2000, C=1.0)), qtext)
        lin = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))  # noqa: E731
        R = X["readout"][idx].astype(np.float32)
        res["probe_readout"] = [cv(lin(), R[:, l]) for l in range(R.shape[1])]
        V = X["img_vec"][idx].astype(np.float32)
        res["probe_img_vec"] = dict(zip(map(str, layers), [cv(lin(), V[:, j]) for j in range(V.shape[1])]))
        print(f"   text_only {res['text_only']}  readout best {max(res['probe_readout'])}  "
              f"img_vec best {max(res['probe_img_vec'].values())}", flush=True)

        wa, wb = AXES[ax_][2:]
        diff = LG[idx][:, :, LENS_WORDS.index(wa)] - LG[idx][:, :, LENS_WORDS.index(wb)]
        res["lens_auc"] = [round(auc(diff[:, l], y), 3) for l in range(diff.shape[1])]
        res["lens_auc_forced_choice"] = [round(auc(diff[fc, l], y[fc]), 3) for l in range(diff.shape[1])] if fc.sum() >= 10 else None
        res["lens_mean"] = {p: [round(float(x), 2) for x in diff[y == v].mean(0)] for p, v in [("gt_A", 1), ("gt_B", 0)]}

        for mode, jd in J.items():
            hit = [jd.get((meta[i]["case_id"], meta[i]["question"])) for i in idx]
            hit = [h for h in hit if h is not None]
            res[f"judge_acc_{mode}"] = round(float(np.mean(hit)), 3) if hit else None
        mk = MIRROR_KEY.get(ax_)
        if mk:
            res["mirror"] = {d: round(float(np.mean([v[d][mk] for v in mirror.values() if d in v])), 3)
                             for d in sorted({d for v in mirror.values() for d in v}, key=int)}

        # within-CT pairs
        by_case = defaultdict(list)
        for k, i in enumerate(idx):
            by_case[case[i]].append(k)
        pairs = {"same CT, same pole": [], "same CT, opposite poles": []}
        for ks in by_case.values():
            for p in range(len(ks)):
                for q in range(p + 1, len(ks)):
                    pairs["same CT, same pole" if y[ks[p]] == y[ks[q]] else "same CT, opposite poles"].append((ks[p], ks[q]))
        diffct = []
        while len(diffct) < min(3000, len(idx) * 2):
            p, q = rng.choice(len(idx), 2, replace=False)
            if case[idx[p]] != case[idx[q]]:
                diffct.append((p, q))
        pairs["different CTs"] = diffct
        res["leak"] = {}
        for name, pr in pairs.items():
            if len(pr) < 5:
                continue
            P = idx[np.array(pr)]
            res["leak"][name] = dict(n=len(pr), img_vec_cos_last=round(float((Vn[P[:, 0], -1] * Vn[P[:, 1], -1]).sum(-1).mean()), 3),
                                     attn_cos=round(float((Att[P[:, 0]] * Att[P[:, 1]]).sum(-1).mean()), 3))
        S["axes"][axis] = res

        fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
        axs[0].plot(range(len(res["probe_readout"])), res["probe_readout"], marker=".", label="readout -> GT pole")
        axs[0].plot([l + 1 for l in layers], list(res["probe_img_vec"].values()), marker="s", label="image vector -> GT pole")
        axs[0].axhline(res["text_only"], color="tab:red", ls="--", label=f"question text only ({res['text_only']})")
        axs[0].axhline(0.5, color="gray", ls=":")
        axs[0].set_xlabel("layer")
        axs[0].set_ylabel("balanced accuracy (5-fold, grouped by patient)")
        axs[0].set_title(f"{axis}: linear probes (n={len(idx)})")
        axs[0].legend(fontsize=8)
        axs[1].plot(res["lens_auc"], marker=".", label="all")
        if res["lens_auc_forced_choice"]:
            axs[1].plot(res["lens_auc_forced_choice"], marker=".", label=f"forced choice (n={fc.sum()})")
        axs[1].axhline(0.5, color="gray", ls=":")
        axs[1].set_xlabel("layer")
        axs[1].set_ylabel(f"AUC of logit({wa}) - logit({wb}) vs GT")
        axs[1].set_title(f"{axis}: logit lens")
        axs[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.out_dir / f"{axis}.png", dpi=120)
        plt.close(fig)
        (a.out_dir / "summary.json").write_text(json.dumps(S, indent=1))

    # one overview figure
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    for axis, r in S["axes"].items():
        line, = axs[0].plot(r["probe_readout"], label=f"{axis} (n={r['n']})")
        axs[0].axhline(r["text_only"], color=line.get_color(), ls="--", lw=0.8)
        axs[1].plot(r["lens_auc"], color=line.get_color(), label=axis)
    axs[0].axhline(0.5, color="gray", ls=":")
    axs[0].set_title("readout probe -> GT pole (dashed: question text only)")
    axs[0].set_xlabel("layer")
    axs[0].legend(fontsize=8)
    axs[1].axhline(0.5, color="gray", ls=":")
    axs[1].set_title("logit lens AUC vs GT")
    axs[1].set_xlabel("layer")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out_dir / "overview.png", dpi=120)
    plt.close(fig)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))} | {
        "readout_best": max(v["probe_readout"]), "img_vec_best": max(v["probe_img_vec"].values()),
        "lens_auc_last": v["lens_auc"][-1]} for k, v in S["axes"].items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lens")
    p.add_argument("--stem", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--snapshot", default=None)
    p = sub.add_parser("analyse")
    p.add_argument("--stem", required=True)
    p.add_argument("--lens", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--judge-image", default=None)
    p.add_argument("--judge-blind", default=None)
    a = ap.parse_args()
    cmd_lens(a) if a.cmd == "lens" else cmd_analyse(a)


if __name__ == "__main__":
    main()

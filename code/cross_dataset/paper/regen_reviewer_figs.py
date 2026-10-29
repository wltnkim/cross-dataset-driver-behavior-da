#!/usr/bin/env python3
"""Reviewer-response figure regen (3-fold):
  1) cross_target_heatmap.pdf  -- 3-fold mean off-diagonals + Table-II DANN diagonal
  2) perclass_combined.pdf     -- per-class F1 as 3-fold MEAN (was fold-0)

Styling matches the existing builders (generate_figures.plot_cross_target_heatmap
and regen_figures.make_perclass_combined) verbatim except for the data source.
Non-destructive: outputs to figures_new/ only; live figures copied separately.
"""
import os, re, glob, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))          # .../code
ARR = os.path.dirname(CODE)                            # release-tree root
# In the pre-release working tree this script came from, results/ sat under code/results/;
# in the released layout, results/ sits at the repo root instead. Prefer whichever exists.
_res_under_code = os.path.join(CODE, "results")
RES = os.environ.get("RESULTS_ROOT",
                      _res_under_code if os.path.isdir(_res_under_code)
                      else os.path.join(ARR, "results"))
OUT = os.path.join(HERE, "figures_new"); os.makedirs(OUT, exist_ok=True)

CLASS_NAMES = ["drinking", "phone_call", "phone_texting",
               "reaching", "adjusting_controls", "seatbelt"]
TARGETS = ["driveandact", "dmd", "3mdad"]
TARGET_LABEL = {"driveandact": "Drive&Act", "dmd": "DMD", "3mdad": "3MDAD"}
PERCLASS_METHODS = ["target_only", "finetune", "dann", "cdan", "mcc", "bnm",
                    "shot", "cleanadapt", "csfda", "bn_adapt", "tent", "tta", "vitta"]
METHOD_DISPLAY = {
    "source_only": "Source-Only", "target_only": "Target-Only", "finetune": "Fine-Tune",
    "dann": "DANN", "cdan": "CDAN", "mcc": "MCC", "bnm": "BNM",
    "shot": "SHOT", "cleanadapt": "CleanAdapt", "csfda": "C-SFDA",
    "bn_adapt": "BN-Adapt", "tent": "TENT", "tta": "LN-Ent", "vitta": "ViTTA",
}
EXC = re.compile(r"reverse|bbSmall|bbRandom|bbsmall|bbrandom|item1|srconly|"
                 r"tier2sanity|sanity|smoke|_uni_|_r\d{3}(_|$)")


def tgt_of(dn, d):
    return d.get("target_dataset") or (
        "dmd" if "_dmd_" in dn or dn.endswith("_dmd")
        else "3mdad" if "_3mdad_" in dn or dn.endswith("_3mdad")
        else "driveandact")


def clean_best(fold):
    best = {}
    for path in glob.glob(os.path.join(RES, f"*_cross_*_fold{fold}", "results.json")):
        dn = os.path.basename(os.path.dirname(path))
        if EXC.search(dn):
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        m = d.get("mode", d.get("method", "?"))
        if m == "cbdann":
            continue
        k = (tgt_of(dn, d), m)
        # Prefer the run with more trained epochs, then the latest dirname. A 3-epoch
        # June re-verification run must not outrank the complete 23-epoch April run
        # (dmd/target_only/fold2, 2026-09-11). Missing epochs_trained ranks lowest.
        ep = d.get("epochs_trained")
        sk = (ep if ep is not None else -1, dn)
        if k not in best or sk > best[k][0]:
            best[k] = (sk, os.path.dirname(path))
    return {k: v[1] for k, v in best.items()}


def parse_report(path):
    f1 = {}
    text = open(path).read()
    for cn in CLASS_NAMES:
        m = re.search(re.escape(cn) + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", text)
        if m:
            f1[cn] = float(m.group(3))
    return f1


CROSS_TARGETS = ["driveandact", "dmd", "3mdad"]


def cross_target_matrix():
    """Compute the 3x3 cross-target DANN matrix from data.

    CORRECTED 2026-09-11 (I-4): this used to be a hardcoded literal because the
    March-2026 cross-target eval_only run directories were believed not cleanly
    reconstructible (see generate_figures.py's plot_cross_target_heatmap() docstring
    for the belief as it was stated at the time).
    That belief was wrong -- the eval-on-other-targets data was never in separate
    "eval_only" directories; it's nested inside the plain per-target DANN run
    directories as extra results_eval_{other_target}[_fold{N}].json files
    (the DANN-adapted model's own evaluation on the other two targets, saved
    alongside its own results.json). Diagonal = that run's own results.json;
    off-diagonal = the nested results_eval_*.json siblings. All 3 folds meaned.
    Recomputing this way reproduced the old literal to 3 decimals exactly for
    every one of the 9 cells (verified 2026-09-11).
    """
    diag = {}
    offdiag = {t: {} for t in CROSS_TARGETS}
    for fold in (0, 1, 2):
        best = clean_best(fold)
        for adapted_to in CROSS_TARGETS:
            dp = best.get((adapted_to, "dann"))
            if not dp:
                continue
            own = json.load(open(os.path.join(dp, "results.json")))
            diag.setdefault(adapted_to, []).append(own["test_balanced_accuracy"])
            for evalf in glob.glob(os.path.join(dp, "results_eval_*.json")):
                d = json.load(open(evalf))
                other = d["eval_target"]
                offdiag[adapted_to].setdefault(other, []).append(d["test_balanced_accuracy"])
    matrix = np.zeros((3, 3))
    for i, a in enumerate(CROSS_TARGETS):
        for j, e in enumerate(CROSS_TARGETS):
            vals = diag[a] if a == e else offdiag[a][e]
            matrix[i, j] = np.mean(vals)
    return matrix


# ---------- 1) cross-target heatmap (3-fold) ----------
def plot_cross_target_heatmap():
    """Cross-target DANN transferability, computed from data (I-4, 2026-09-11) --
    see cross_target_matrix()'s docstring for what changed and why."""
    labels = ["D&A", "DMD", "3MDAD"]
    # rows = adapted to, cols = evaluated on.
    matrix = cross_target_matrix()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Evaluated on", fontsize=8); ax.set_ylabel("Adapted to", fontsize=8)
    for i in range(3):
        for j in range(3):
            rgb = im.cmap(im.norm(matrix[i, j]))[:3]
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            color = "white" if lum < 0.45 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Balanced Accuracy", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    base = os.path.join(OUT, "cross_target_heatmap")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + "_preview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[cross-target-3fold] -> {base}.pdf")


# ---------- 2) per-class combined (3-fold mean) ----------
def f1mean_3fold(tgt):
    folds = {f: clean_best(f) for f in (0, 1, 2)}
    methods, rows = [], []
    for m in PERCLASS_METHODS:
        vals = []
        for cn in CLASS_NAMES:
            cell = []
            for f in (0, 1, 2):
                dp = folds[f].get((tgt, m))
                if dp:
                    rep = os.path.join(dp, "classification_report.txt")
                    if os.path.exists(rep):
                        v = parse_report(rep).get(cn)
                        if v is not None:
                            cell.append(v)
            vals.append(float(np.mean(cell)) if cell else 0.0)
        # include method only if it has any fold present for this target
        if any(folds[f].get((tgt, m)) for f in (0, 1, 2)):
            methods.append(m); rows.append(vals)
    return methods, np.array(rows)


def make_perclass_combined():
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6))
    cov = []
    for ax, tgt in zip(axes, TARGETS):
        methods, f1mat = f1mean_3fold(tgt)
        f1mat = f1mat.T          # classes x methods
        im = ax.imshow(f1mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(np.arange(len(CLASS_NAMES)))
        ax.set_yticklabels([cn.replace("_", " ") for cn in CLASS_NAMES], fontsize=8.5)
        ax.set_xticks(np.arange(len(methods)))
        if tgt == TARGETS[-1]:
            ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in methods],
                               fontsize=8.5, rotation=45, ha="right")
        else:
            ax.set_xticklabels([])
        ax.set_title(TARGET_LABEL[tgt], fontsize=12, pad=3)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(methods)):
                v = f1mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if v < 0.6 else "black")
        cov.append((tgt, len(methods)))
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("F1 score", fontsize=10); cb.ax.tick_params(labelsize=8)
    base = os.path.join(OUT, "perclass_combined")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + "_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[perclass-combined-3fold] -> {base}.pdf  (methods/target: {cov})")


if __name__ == "__main__":
    plot_cross_target_heatmap()
    make_perclass_combined()
    print("Done.")

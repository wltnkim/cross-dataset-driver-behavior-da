"""Generate paper-specific figures: SSDA vs LE line chart.

CONVERTED 2026-09-11 (K-2): the 5/10/25/50% points below are now computed from
the copied SSDA/LE-ratio run population in `results/` (see `ratio_point()`),
not hardcoded. Verified to match the old literals to 4 decimal places on every
one of the 14 checkable cells.

The r=0% point ("ssda" col) and the two 100%-row reference values
("full_dann"/"full_to") remain literals, deliberately -- they are NOT the same
population as the ratio sweep:
  - r=0% (zero target labels) is Table 5's own DANN-UDA value (Section
    sec:uda_failure), which uses a *different* protocol (fixed 50-epoch,
    patience disabled) than the ratio sweep (patience=15). There IS a
    `*_cross_dann_r000_fold*`-named population in `results/` that looks like a
    plausible r=0% source, but it uses patience=15 -- a DIFFERENT protocol --
    and its 3-fold mean (e.g. 0.256 for Drive&Act) does not match Table 5's
    printed value (0.232) or this literal. Using it would silently swap in the
    wrong population. Table 5 itself has no generator script yet (tracked in
    `TABLE_PROVENANCE_INVENTORY.md`), so there is nothing to point this value
    at other than the verified-correct literal until that's resolved.
  - full_dann/full_to (100% row) are Table 4's own DANN/Target-Only values,
    re-verified 2026-09-11 against the copied population via `clean_best()`:
    exact match for Drive&Act and 3MDAD; DMD's full_to (0.9666) is within
    0.0005 of the directly-computed value (0.9661) -- a known, separate,
    already-reported drift in exactly one candidate run for dmd/target_only/
    fold2 (a 3-epoch run outranks the correct 23-epoch run under naive
    "latest wins" selection; using the correct run reproduces 0.9666 exactly).
    Not fixed here -- reported, zero display-level impact, tracked separately.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import json
import re

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures_new")  # E-0/E-6: consistent with the other regen_*.py generators
os.makedirs(OUT_DIR, exist_ok=True)

_ARR = os.environ.get("ARR_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_RES = os.path.join(_ARR, "results")
_CAMPAIGN_TAG = re.compile(r"_E[1-4]L?_|da573_random|broad573|narrow573|_uni_|smoke", re.I)


def ratio_point(mode_token, target, ratio):
    """3-fold mean test_balanced_accuracy for (mode_token, target, ratio) from
    the original (pre-revision) SSDA/LE-ratio run population."""
    ratio_tag = "r%03d" % round(ratio * 100)
    patt_a = os.path.join(_RES, "*_cross_%s_%s_fold*" % (mode_token, ratio_tag), "results.json")
    patt_b = os.path.join(_RES, "*_cross_%s_*_%s_fold*" % (mode_token, ratio_tag), "results.json")
    by_fold = {}
    for path in glob.glob(patt_a) + glob.glob(patt_b):
        dn = os.path.basename(os.path.dirname(path))
        if _CAMPAIGN_TAG.search(dn):
            continue
        d = json.load(open(path))
        if d.get("mode") != mode_token:
            continue
        tgt = d.get("target_dataset") or ("dmd" if "_dmd_" in dn else "3mdad" if "_3mdad_" in dn else "driveandact")
        if tgt != target:
            continue
        fold = d.get("fold")
        by_fold[fold] = d["test_balanced_accuracy"]
    if len(by_fold) != 3:
        raise RuntimeError("expected 3 folds for (%s, %s, %.3f), found %d" %
                            (mode_token, target, ratio, len(by_fold)))
    return sum(by_fold.values()) / 3


def plot_ssda_vs_le():
    """3-panel line chart: SSDA-DANN vs Target-Only (LE) at each label ratio."""

    # Table 7 (main.tex), per target. 5/10/25/50% points computed from data
    # (see ratio_point() above); r=0 and the 100% reference values are
    # documented literals -- see the module docstring for why.
    TGT_KEYS = {"Drive&Act": "driveandact", "DMD": "dmd", "3MDAD": "3mdad"}
    data = {
        "Drive&Act": {
            "ratios":    [0.0,    5,      10,     25,     50],
            "ssda":      [0.232,  None, None, None, None],   # Table 7, D&A SSDA col (r=0 literal, see docstring)
            "le":        [None,   None, None, None, None],
            "full_dann": 0.9438,  # Table 7, D&A 100% row, SSDA col (== Table 4 DANN)
            "full_to":   0.9468,  # Table 7, D&A 100% row, LE col (== Table 4 Target-Only)
        },
        "DMD": {
            "ratios":    [0.0,    5,      10,     25,     50],
            "ssda":      [0.312,  None, None, None, None],   # Table 7, DMD SSDA col (r=0 literal, see docstring)
            "le":        [None,   None, None, None, None],
            "full_dann": 0.9691,  # Table 7, DMD 100% row, SSDA col
            "full_to":   0.9666,  # Table 7, DMD 100% row, LE col
        },
        "3MDAD": {
            "ratios":    [0.0,    5,      10,     25,     50],
            "ssda":      [0.308,  None, None, None, None],   # Table 7, 3MDAD SSDA col (r=0 literal, see docstring)
            "le":        [None,   None, None, None, None],
            "full_dann": 0.9740,  # Table 7, 3MDAD 100% row, SSDA col
            "full_to":   0.9496,  # Table 7, 3MDAD 100% row, LE col
        },
    }
    # Fill in the ratio points that exist from data. No target has an SSDA
    # point at 50% (no such run in the original campaign); 3MDAD additionally
    # has no LE point at 5% -- both match the original hardcoded table's own
    # None placements exactly, not an oversight introduced here.
    HAS_LE_5PCT = {"Drive&Act": True, "DMD": True, "3MDAD": False}
    for name, d in data.items():
        tgt = TGT_KEYS[name]
        for i, r in ((1, 5), (2, 10), (3, 25)):
            d["ssda"][i] = ratio_point("dann", tgt, r / 100.0)
        d["le"][1] = ratio_point("target_only", tgt, 0.05) if HAS_LE_5PCT[name] else None
        d["le"][2] = ratio_point("target_only", tgt, 0.10)
        d["le"][3] = ratio_point("target_only", tgt, 0.25)
        d["le"][4] = ratio_point("target_only", tgt, 0.50)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), sharey=True)

    for ax, (name, d) in zip(axes, data.items()):
        ratios = d["ratios"]

        # SSDA line
        ssda_r = [r for r, v in zip(ratios, d["ssda"]) if v is not None]
        ssda_v = [v for v in d["ssda"] if v is not None]
        ax.plot(ssda_r, ssda_v, "s-", color="#d62728", linewidth=1.3, markersize=3.5,
                label="SSDA-DANN", zorder=3)

        # LE line
        le_r = [r for r, v in zip(ratios, d["le"]) if v is not None]
        le_v = [v for v in d["le"] if v is not None]
        ax.plot(le_r, le_v, "o-", color="#1f77b4", linewidth=1.3, markersize=3.5,
                label="Target-Only (LE)", zorder=3)

        # Full supervision reference lines
        ax.axhline(d["full_dann"], color="#d62728", linestyle="--", alpha=0.5, linewidth=0.8,
                   label="DANN (100%)")
        ax.axhline(d["full_to"], color="#1f77b4", linestyle="--", alpha=0.5, linewidth=0.8,
                   label="TO (100%)")

        # Shade the region where LE > SSDA
        for i in range(len(ssda_r)):
            r = ssda_r[i]
            sv = ssda_v[i]
            # find matching LE
            if r in le_r:
                lv = le_v[le_r.index(r)]
                if lv > sv:
                    gap = lv - sv
                    ax.text(r + 0.8, sv - 0.03, f"+{gap:.1%}".replace("%", " pp"),
                            fontsize=6, fontweight="bold", color="#2ca02c",
                            ha="left", va="top")

        ax.set_title(name, fontsize=8, fontweight="bold")
        ax.set_xlabel("Target label ratio (%)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_xticks([0, 5, 10, 25, 50])
        ax.set_xticklabels(["0", "5", "10", "25", "50"])
        ax.set_xlim(-3, 55)
        ax.set_ylim(0.15, 1.02)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Balanced Accuracy", fontsize=7)
    axes[0].legend(fontsize=6, loc="lower right")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "ssda_vs_le.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    path_png = os.path.join(OUT_DIR, "ssda_vs_le.png")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")
    print(f"[Saved] {path_png}")


# =====================================================================
# Figure: Cross-target transferability heatmap
# =====================================================================

def plot_cross_target_heatmap():
    """Heatmap of cross-target balanced accuracy.

    Values are carried directly from Table 9 (main.tex), not read from a
    committed CSV: the March-2026 cross-target eval_only run directories that
    originally produced these numbers were not available in this tree when this
    literal was written, and were treated as not cleanly reconstructible at the
    time. Re-verified against the current Table 9 before being kept.
    regen_reviewer_figs.py's
    plot_cross_target_heatmap() carries the identical literal and is the
    authoritative copy for this figure.
    """
    labels = ["D&A", "DMD", "3MDAD"]
    # rows = adapted to, cols = evaluated on. Table 9.
    matrix = np.array([
        [0.944, 0.499, 0.345],   # adapted to D&A (row): D&A/DMD/3MDAD columns
        [0.364, 0.969, 0.727],   # adapted to DMD
        [0.405, 0.723, 0.974],   # adapted to 3MDAD
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.2, vmax=1.0, aspect="auto")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Evaluated on", fontsize=12)
    ax.set_ylabel("Adapted to", fontsize=12)

    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] < 0.5 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                    fontsize=12, color=color, fontweight=weight)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Balanced Accuracy", fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cross_target_heatmap.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    path_png = os.path.join(OUT_DIR, "cross_target_heatmap.png")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")
    print(f"[Saved] {path_png}")


if __name__ == "__main__":
    plot_ssda_vs_le()
    plot_cross_target_heatmap()
    print("Done.")

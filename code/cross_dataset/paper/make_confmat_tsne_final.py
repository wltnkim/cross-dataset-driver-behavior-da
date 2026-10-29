"""Generate Fig. 5 (confusion matrices) and Fig. 2 (t-SNE) of the final IEEE Access version.

Inputs (all in this directory):
  cm_data.py         fold-0 confusion matrices, transcribed from the per-run
                     confusion_matrix*.png files under results/ and checked against
                     each run's classification_report.txt by validate_cm.py.
  tsne_points/*.csv  t-SNE point positions of the published figure, extracted from
                     the vector PDF of the accepted manuscript (the embeddings
                     themselves are not part of this release). Coordinates are in
                     arbitrary t-SNE display units.
Output: figures_new/confmat_*.pdf and figures_new/tsne_domain_*.pdf
"""
import os, csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_new"); os.makedirs(OUT, exist_ok=True)
import sys; sys.path.insert(0, HERE)
from cm_data import CM
LBL = ["drinking", "phone_call", "phone_texting", "reaching", "adj._controls", "seatbelt"]


def confmat_row(tgt, methods, fname):
    n = len(methods); W = 7.0
    fig, axes = plt.subplots(1, n, figsize=(W * n / 4, 1.95), gridspec_kw=dict(wspace=0.08))
    for j, (ax, m) in enumerate(zip(axes, methods)):
        cm = np.array(CM[(tgt, m)][1])
        if cm.shape[0] == 5:
            cm = np.hstack([cm, np.zeros((5, 1), int)])   # six-way head: seatbelt column, never predicted
        elif tgt != "driveandact":
            cm = cm[:5]                                    # seatbelt absent from DMD/3MDAD ground truth
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max(), aspect="auto")
        r, c = cm.shape
        for i in range(r):
            for k in range(c):
                v = cm[i, k]
                ax.text(k, i, str(v), ha="center", va="center", fontsize=5.6,
                        color="white" if v > 0.55 * cm.max() else "#222222")
        ax.set_title(m, fontsize=7.5, pad=3)
        ax.set_xticks(range(c)); ax.set_xticklabels(LBL[:c], rotation=40, ha="right", fontsize=5.8)
        ax.set_yticks(range(r)); ax.set_yticklabels(LBL[:r] if j == 0 else [], fontsize=5.8)
        ax.tick_params(length=1.5, pad=1)
        if j == 0:
            ax.set_ylabel("True", fontsize=6.5, labelpad=4)
        ax.set_xlabel("Predicted", fontsize=6.5, labelpad=1)
    fig.savefig(os.path.join(OUT, fname + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


confmat_row("driveandact", ["Source-Only", "DANN", "Target-Only", "CleanAdapt"], "confmat_driveandact")
confmat_row("dmd", ["Source-Only", "DANN", "Target-Only", "CleanAdapt"], "confmat_dmd")
confmat_row("3mdad", ["DANN", "Target-Only", "CleanAdapt"], "confmat_3mdad")

NAME = {"driveandact": "D&A", "dmd": "DMD", "3mdad": "3MDAD"}
COL = {"source": (0.122, 0.467, 0.706), "target": (1.0, 0.498, 0.055)}
for t, name in NAME.items():
    pts = {"source": [], "target": []}
    with open(os.path.join(HERE, "tsne_points", f"tsne_points_{t}.csv")) as fh:
        for row in csv.DictReader(fh):
            pts[row["domain"]].append((float(row["x"]), float(row["y"])))
    fig, ax = plt.subplots(figsize=(2.3, 2.0))
    for dom in ("source", "target"):                       # same drawing order as the original figure
        a = np.array(pts[dom])
        lab = "OBC (source)" if dom == "source" else f"{name} (target)"
        ax.scatter(a[:, 0], a[:, 1], s=3, color=COL[dom], alpha=0.5, linewidths=0, label=lab)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=7); ax.set_ylabel("t-SNE 2", fontsize=7)
    leg = ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, markerscale=3,
                    handletextpad=0.1, columnspacing=0.8, borderpad=0.2, frameon=False)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.savefig(os.path.join(OUT, f"tsne_domain_{t}.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
print("wrote", OUT)

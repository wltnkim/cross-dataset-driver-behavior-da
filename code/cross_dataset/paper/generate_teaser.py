"""
Generate Teaser Figure (Figure 1) for the Cross-Dataset DA paper.
Four-panel figure (2x2, Group E / E-3):
  (a) Benchmark overview: OBC -> 3 targets, 14 methods across 4 paradigms
  (b) UDA catastrophic failure vs supervised performance
  (c) Label efficiency: LE > SSDA at low labels
  (d) Selection-rule separation: apparent adversarial benefit requires oracle
      (target-label) selection; collapses under every label-free rule. Values
      for (d) ARE read from a released analysis artifact.

CONVERTED 2026-09-11 (K-2): panel (c)'s 5/10/25/50% points are now computed
from the copied SSDA/LE-ratio run population (see `ratio_point()`), not
hardcoded -- verified to match the old literals to within floating-point
rounding. Panel (c)'s r=0 point and 100%-reference value, and ALL of panel
(b), remain literals: (c)'s r=0 point is Table 5's own DANN-UDA value (a
different protocol than the ratio sweep, see `generate_figures.py`'s
docstring for the full reasoning); panel (b) needs Table 5's full UDA
population (7 methods x 3 targets x 3 folds, mean/min/max) or Table 4's, and
neither table has its own generator/traced population yet (tracked in
`TABLE_PROVENANCE_INVENTORY.md`) -- converting (b) would mean first solving
that, a materially larger task than this round's scope, not attempted here.

Outputs:
  fig_teaser.{pdf,png}    — combined 4-panel (2x2) figure
  fig_teaser_a.{pdf,png}  — panel (a) only
  fig_teaser_b.{pdf,png}  — panel (b) only
  fig_teaser_c.{pdf,png}  — panel (c) only
  fig_teaser_d.{pdf,png}  — panel (d) only
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import csv
import glob
import json
import re

_ARR = os.environ.get("ARR_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_RES = os.path.join(_ARR, "results")
_CAMPAIGN_TAG = re.compile(r"_E[1-4]L?_|da573_random|broad573|narrow573|_uni_|smoke", re.I)


def _ratio_point(mode_token, target, ratio):
    """3-fold mean test_balanced_accuracy for (mode_token, target, ratio) from
    the original (pre-revision) SSDA/LE-ratio run population. Same logic as
    generate_figures.py::ratio_point(), duplicated here to keep this script
    self-contained (matches this file's existing pattern of not importing
    from its sibling generators)."""
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
        by_fold[d.get("fold")] = d["test_balanced_accuracy"]
    if len(by_fold) != 3:
        raise RuntimeError("expected 3 folds for (%s, %s, %.3f), found %d" %
                            (mode_token, target, ratio, len(by_fold)))
    return sum(by_fold.values()) / 3

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
})

# Colors
c_da = '#e74c3c'
c_le = '#2ecc71'
c_supervised = '#3498db'
c_chance = '#bdc3c7'
c_uda = '#e74c3c'

targets_short = ['D&A', 'DMD', '3MDAD']

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures_new')
os.makedirs(OUTDIR, exist_ok=True)

# Panel (d) data source: the released campaign cell table (D-1/D-4's population,
# 27 adversarial + 10 source-only cells). Repo-relative: this script now lives inside the
# release tree at code/cross_dataset/paper/, so ARR is two directories up. If/when the
# D-2 rename (swan_campaign_2026-09-07 -> campaign_2026-09-07) executes, update this path.
_ARR = os.environ.get("ARR_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
CELLS_CSV = os.path.join(_ARR, "analysis", "swan_campaign_2026-09-07", "reeval_cells.csv")
NOISE_FLOOR = 0.077445  # Section V-B; same figure the manuscript uses throughout.

def _chance_of(target):
    return 1 / 6 if target == 'obc' else 0.2


def _load_selection_rule_cells():
    """Returns (adversarial, source_only), each a list of 4-tuples
    (oracle, src_val, last_epoch, min_tgt_ent) minus that cell's own chance."""
    rule_cols = ['oracle_target_val_mean', 'source_val_mean',
                 'last_epoch_mean', 'min_target_entropy_mean']
    adversarial, source_only = [], []
    with open(CELLS_CSV, newline='') as f:
        for r in csv.DictReader(f):
            chance = _chance_of(r['target'])
            diffs = tuple(float(r[c]) - chance for c in rule_cols)
            if r['method'] in ('dann', 'cdan'):
                adversarial.append(diffs)
            elif r['method'] == 'source_only':
                source_only.append(diffs)
    return adversarial, source_only


def draw_panel_a(ax, title=True):
    """Benchmark Overview: source → 3 targets + methods summary."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    if title:
        ax.set_title('(a) Benchmark Overview', fontweight='bold', fontsize=9, pad=4)

    # Source box
    src_box = mpatches.FancyBboxPatch((0.3, 6.8), 3.0, 2.2, boxstyle="round,pad=0.2",
                                       facecolor='#3498db', edgecolor='#2980b9',
                                       linewidth=1.5, alpha=0.9)
    ax.add_patch(src_box)
    ax.text(1.8, 8.2, 'OBC', fontsize=10, fontweight='bold', ha='center', va='center', color='white')
    ax.text(1.8, 7.5, '42 subj · 6 cls\nAzure Kinect', fontsize=6.5, ha='center', va='center', color='white')

    # Target boxes — y centers 8.0 / 5.6 / 3.2, box height 1.6 (y-0.8 .. y+0.8)
    tgt_info = [
        ('Drive&Act', '15 subj\nKinect RGB', '#e74c3c', 8.0),
        ('DMD',       '31 subj\nBody RGB',    '#e67e22', 5.6),
        ('3MDAD',     '50 subj\nKinect v2',   '#9b59b6', 3.2),
    ]
    for name, info, color, y in tgt_info:
        box = mpatches.FancyBboxPatch((5.8, y - 0.8), 3.8, 1.6, boxstyle="round,pad=0.15",
                                       facecolor=color, edgecolor='none', alpha=0.85)
        ax.add_patch(box)
        ax.text(7.7, y + 0.3, name, fontsize=8, fontweight='bold',
                ha='center', va='center', color='white')
        ax.text(7.7, y - 0.3, info, fontsize=6, ha='center', va='center', color='white')
        # Arrow from source box right edge (~x=3.4, y=7.9) to target box left edge
        ax.annotate('', xy=(5.7, y), xytext=(3.4, 7.9),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0,
                                    connectionstyle='arc3,rad=0.1'))

    # Methods summary box — rect (0.3,0.2)..(9.5,1.9), pad 0.18 keeps it inside the
    # axes (visual bottom ~0.02) and clear of the 3MDAD box (visual bottom ~2.25).
    # The paradigm list is split across two lines so it no longer overflows the box.
    mbox = mpatches.FancyBboxPatch((0.3, 0.2), 9.2, 1.7, boxstyle="round,pad=0.18",
                                    facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=1.0)
    ax.add_patch(mbox)
    ax.text(4.9, 1.60, '14 Methods · 4 Paradigms', fontsize=7.5, fontweight='bold',
            ha='center', va='center', color='#2c3e50')
    ax.text(4.9, 1.18, 'Baselines (3) · Source-Avail. (4)', fontsize=6,
            ha='center', va='center', color='#555555')
    ax.text(4.9, 0.80, 'Source-Free (3) · Test-Time (4)', fontsize=6,
            ha='center', va='center', color='#555555')
    ax.text(4.9, 0.40, '3-fold CV · Balanced Accuracy', fontsize=6, ha='center', va='center',
            color='#7f8c8d', fontstyle='italic')


def draw_panel_b(ax, title=True):
    """UDA Fails Catastrophically: supervised vs UDA balanced accuracy.

    Values are carried directly from Table 4 (Target-Only row) and Table 5 (the
    seven UDA methods' three-fold means, both tables in main.tex), NOT read from
    a committed CSV: no such artifact exists for this March-2026 campaign (the
    raw per-run results.json files this table was originally typeset from are
    not cleanly reconstructible -- the same provenance gap noted for the
    cross-target heatmap). Every value below was independently re-verified
    against the current tables before being kept.
    """
    supervised = [0.947, 0.967, 0.950]            # Table 4, Target-Only row (D&A, DMD, 3MDAD)
    uda_mean   = [0.220, 0.263, 0.295]            # Table 5: mean over the 7 UDA methods, per target
    uda_min    = [0.174, 0.221, 0.200]            # Table 5: min over the 7 UDA methods, per target
    uda_max    = [0.254, 0.312, 0.392]            # Table 5: max over the 7 UDA methods, per target
    chance     = [1/6, 0.200, 0.200]              # per-target chance (Table 5's own "Chance" row): D&A=1/6, DMD/3MDAD=1/5

    x = np.arange(3)
    w = 0.28

    ax.bar(x - w/2, supervised, w, color=c_supervised, label='Target-Only (100% labels)',
           edgecolor='white', linewidth=0.5)
    yerr = [[m - lo for m, lo in zip(uda_mean, uda_min)],
            [hi - m for m, hi in zip(uda_mean, uda_max)]]
    ax.bar(x + w/2, uda_mean, w, color=c_uda, label='UDA, 7 methods (0% labels)',
           edgecolor='white', linewidth=0.5,
           yerr=yerr, capsize=3, error_kw=dict(ecolor='#333', lw=0.9, capthick=0.9))

    # Per-target chance: a short segment spanning each bar group, not one line
    # across the whole axis, since D&A's chance (1/6) differs from DMD/3MDAD's (1/5).
    for xi, c in zip(x, chance):
        ax.plot([xi - 0.42, xi + 0.42], [c, c], color=c_chance, linestyle='--',
                linewidth=1.0, zorder=0)
    # Anchored just past the 3MDAD group's right edge (outside its bars/error
    # whisker) with ha='left' so the text grows into the empty margin rather
    # than back over the bar -- ha='right' at this x previously grew leftward
    # and sat on top of the 3MDAD UDA bar and its error-bar whisker.
    ax.text(2.48, chance[2] + 0.006, 'chance', fontsize=6.5, color='#999999',
            fontstyle='italic', ha='left', va='center')

    ax.set_xticks(x)
    ax.set_xticklabels(targets_short)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if title:
        ax.set_title('(b) UDA Underperforms Substantially', fontweight='bold', fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), framealpha=0.9, handlelength=1.2, handletextpad=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Single range annotation: the supervised gap spans 55-77 pp across the seven methods
    ax.text(0.5, 0.55, '55–77 pp\nbelow full\nsupervision', fontsize=6.5, color='#333',
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#999999', alpha=0.85))


def draw_panel_c(ax, title=True):
    """Label Efficiency: LE > SSDA at low labels (D&A).

    Values carried from Table 7 (Drive&Act columns, SSDA and LE), same
    provenance note as draw_panel_b above -- re-verified against the current
    table, not read from a committed CSV.
    """
    ratios_ssda = [0, 5, 10, 25]
    ssda_da = [0.232] + [_ratio_point("dann", "driveandact", r / 100.0) for r in (5, 10, 25)]

    ax.plot(ratios_ssda, ssda_da, 'o-', color=c_uda, markersize=4, linewidth=1.5,
            label='SSDA-DANN', zorder=3)
    le_da = [_ratio_point("target_only", "driveandact", r / 100.0) for r in (5, 10, 25, 50)]
    ax.plot([5, 10, 25, 50], le_da,
            's-', color=c_le, markersize=4, linewidth=1.5, label='Target-only (LE)', zorder=3)

    # Supervised reference: Table 7, D&A 100% row, LE column (== Table 4 Target-Only D&A)
    ax.axhline(y=0.947, color=c_supervised, linestyle=':', linewidth=1.0, alpha=0.7)
    ax.text(48, 0.955, '100%', fontsize=6, color=c_supervised,
            ha='right', fontstyle='italic')

    # Highlight the gap at 5%
    ax.annotate('+12 pp', xy=(5, 0.867), xytext=(12, 0.78),
                fontsize=7, fontweight='bold', color=c_le,
                arrowprops=dict(arrowstyle='->', color=c_le, lw=1.0))

    ax.set_xlabel('Target Label Ratio (%)')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_ylim(0.2, 1.02)
    ax.set_xlim(-3, 55)
    ax.set_xticks([0, 5, 10, 25, 50])
    ax.set_xticklabels(['0', '5', '10', '25', '50'])
    if title:
        ax.set_title('(c) LE > SSDA (D&A)', fontweight='bold', fontsize=9)
    ax.legend(loc='lower right', framealpha=0.9, handlelength=1.5, handletextpad=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


c_adv = '#E69F00'   # colorblind-safe orange (adversarial cells)
c_so = '#0072B2'    # colorblind-safe blue (source-only cells)


def draw_panel_d(ax, title=True):
    """Selection-rule separation: apparent adversarial benefit requires oracle
    (target-label) selection and collapses under every label-free rule.

    Every value is read from CELLS_CSV (via _load_selection_rule_cells), not
    hardcoded -- this is the one teaser panel with a released source artifact.
    """
    adversarial, source_only = _load_selection_rule_cells()
    rng = np.random.RandomState(0)
    xpos = [0.0, 1.4, 1.9, 2.4]
    rule_labels = ['Oracle', 'Src-val', 'Last-epoch', 'Min-tgt-ent.']

    # Noise-floor band around chance (labelled in the caption, not on the axes)
    # and the chance line itself.
    ax.axhspan(-NOISE_FLOOR, NOISE_FLOOR, color='#cccccc', alpha=0.35, zorder=0)
    ax.axhline(0, color='#888888', linestyle='--', linewidth=0.9, zorder=1)
    # Visual separator between the oracle rule and the three label-free rules.
    ax.axvline(0.7, color='#bbbbbb', linestyle=':', linewidth=1.0, zorder=1)

    for i, x in enumerate(xpos):
        adv_vals = [c[i] for c in adversarial]
        so_vals = [c[i] for c in source_only]
        jitter_adv = rng.uniform(-0.10, 0.10, len(adv_vals))
        jitter_so = rng.uniform(-0.10, 0.10, len(so_vals))
        ax.scatter(np.full(len(adv_vals), x - 0.13) + jitter_adv, adv_vals,
                   s=13, color=c_adv, marker='o', alpha=0.7, zorder=3,
                   edgecolors='none',
                   label=f'Adversarial (n={len(adv_vals)})' if i == 0 else None)
        ax.scatter(np.full(len(so_vals), x + 0.13) + jitter_so, so_vals,
                   s=15, color=c_so, marker='^', alpha=0.8, zorder=3,
                   edgecolors='none',
                   label=f'Source-only (n={len(so_vals)})' if i == 0 else None)
        ax.plot([x - 0.24, x - 0.02], [np.mean(adv_vals)] * 2, color=c_adv, lw=2.2, zorder=4)
        ax.plot([x + 0.02, x + 0.24], [np.mean(so_vals)] * 2, color=c_so, lw=2.2, zorder=4)

    ax.set_xticks(xpos)
    ax.set_xticklabels(rule_labels, fontsize=7.5, rotation=20, ha='right')
    ax.set_ylabel("Balanced accuracy $-$\ntarget's own chance", fontsize=8.5)
    if title:
        ax.set_title('(d) Benefit Requires Oracle Selection', fontweight='bold', fontsize=9)
    ax.legend(fontsize=6.5, loc='upper right', framealpha=0.9, handletextpad=0.3, markerscale=1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Combined 4-panel (2x2) figure ────────────────────────────────────
# E-3: widened from 1x3 to 2x2 to add panel (d) without shrinking (a)-(c).
# Accepted cost: roughly a third of a page more height (per-panel width is
# larger than the old 1x3 layout, not smaller).
fig = plt.figure(figsize=(7.6, 6.4))
gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, top=0.95, bottom=0.09,
                       wspace=0.38, hspace=0.55)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
draw_panel_a(ax1)
draw_panel_b(ax2)
draw_panel_c(ax3)
draw_panel_d(ax4)
for ext in ['pdf', 'png']:
    fig.savefig(f'{OUTDIR}/fig_teaser.{ext}', bbox_inches='tight', dpi=300)
    print(f'Saved fig_teaser.{ext}')
plt.close(fig)


# ── Individual panels (for rearrangement by user) ───────────────────

# Panel (a) — standalone, slightly taller canvas to breathe
fig_a = plt.figure(figsize=(3.0, 3.2))
ax_a = fig_a.add_axes([0.02, 0.04, 0.96, 0.92])
draw_panel_a(ax_a)
for ext in ['pdf', 'png']:
    fig_a.savefig(f'{OUTDIR}/fig_teaser_a.{ext}', bbox_inches='tight', dpi=300)
    print(f'Saved fig_teaser_a.{ext}')
plt.close(fig_a)

# Panel (b) — bar chart
fig_b = plt.figure(figsize=(2.8, 2.8))
ax_b = fig_b.add_axes([0.18, 0.14, 0.78, 0.78])
draw_panel_b(ax_b)
for ext in ['pdf', 'png']:
    fig_b.savefig(f'{OUTDIR}/fig_teaser_b.{ext}', bbox_inches='tight', dpi=300)
    print(f'Saved fig_teaser_b.{ext}')
plt.close(fig_b)

# Panel (c) — line chart
fig_c = plt.figure(figsize=(2.8, 2.8))
ax_c = fig_c.add_axes([0.18, 0.14, 0.78, 0.78])
draw_panel_c(ax_c)
for ext in ['pdf', 'png']:
    fig_c.savefig(f'{OUTDIR}/fig_teaser_c.{ext}', bbox_inches='tight', dpi=300)
    print(f'Saved fig_teaser_c.{ext}')
plt.close(fig_c)

# Panel (d) — selection-rule separation strip plot
fig_d = plt.figure(figsize=(3.4, 2.8))
ax_d = fig_d.add_axes([0.20, 0.22, 0.76, 0.66])
draw_panel_d(ax_d)
for ext in ['pdf', 'png']:
    fig_d.savefig(f'{OUTDIR}/fig_teaser_d.{ext}', bbox_inches='tight', dpi=300)
    print(f'Saved fig_teaser_d.{ext}')
plt.close(fig_d)

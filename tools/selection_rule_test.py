#!/usr/bin/env python3
"""D-4: one statistical test for the selection-rule finding (Table 16,
tab:selection_rule_finding, Section V-K, sec:campaign_evidence).

Question: under label-free (target-label-free) checkpoint selection, do
adversarial cells (DANN/CDAN) sit further below their target's chance level
than source-only cells sit above theirs?

Data structure and why this test:
  - Population: the same 27 adversarial cells and 10 source-only cells as
    Table 16 (analysis/campaign_2026-09-07/reeval_cells.csv, oracle-selected
    is NOT used here -- the three *label-free* rule columns are: source_val,
    last_epoch, min_target_entropy).
  - Five-class (DMD, 3MDAD, OBC-as-source-in-E2 is 6-class) and six-class
    targets are made commensurable by scoring each cell as (rule_mean -
    chance), chance = 1/6 for Drive&Act/OBC targets, 1/5 = 0.2 for DMD/3MDAD.
  - The three label-free rules are three views of the same underlying runs
    (same checkpoints, different selection criteria applied post hoc to the
    same training trajectory), so they are NOT independent samples. Rather
    than silently pool 3 x 37 = 111 pseudo-independent points, each cell is
    first collapsed to ONE summary value: the MEDIAN of its three
    (rule_mean - chance) values. This makes the analysis unit the cell
    (n=27 vs n=10), matching Table 16's own population and avoiding
    within-cell pseudo-replication across rules.
  - Adversarial and source-only cells still share sources/targets/folds to
    some degree (e.g. several methods on the same source/target pair), which
    a rank-based two-sample test does not fully model; this is a known,
    stated assumption, not something this script corrects for.
  - Mann-Whitney U (two-sided) is used: distribution-free, appropriate given
    the visible skew and floor effects near chance (several source_val/
    last_epoch/min_target_entropy values are exactly at the chance floor for
    collapsed adversarial runs), and does not assume normality or equal
    variance.
  - This test was not registered in PREREGISTRATION.md and is reported as a
    post hoc / exploratory analysis, not a confirmatory one.

Run: python3 tools/selection_rule_test.py
"""
import csv
import sys
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parent.parent
# TODO at release packaging: rename to analysis/campaign_2026-09-07/ (D-2).
SRC = REPO / "analysis" / "swan_campaign_2026-09-07" / "reeval_cells.csv"

LABEL_FREE_RULES = ["source_val_mean", "last_epoch_mean", "min_target_entropy_mean"]


def chance_of(target):
    return 1 / 6 if target in ("obc", "driveandact") else 0.2


def main():
    adversarial, source_only = [], []
    with open(SRC, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] in ("dann", "cdan"):
                bucket = adversarial
            elif r["method"] == "source_only":
                bucket = source_only
            else:
                continue
            chance = chance_of(r["target"])
            diffs = [float(r[c]) - chance for c in LABEL_FREE_RULES]
            diffs.sort()
            median_diff = diffs[1]  # median of 3
            bucket.append({
                "group": r["group"], "method": r["method"], "source": r["source"],
                "target": r["target"], "backbone": r["backbone"],
                "per_rule_diffs": diffs, "median_diff": median_diff,
            })

    n_adv, n_so = len(adversarial), len(source_only)
    adv_vals = [c["median_diff"] for c in adversarial]
    so_vals = [c["median_diff"] for c in source_only]

    print(f"n adversarial cells = {n_adv} (expect 27)")
    print(f"n source-only cells = {n_so} (expect 10)")
    print()
    print("adversarial cells (method, source->target[/backbone], median rule-minus-chance):")
    for c in sorted(adversarial, key=lambda c: c["median_diff"]):
        tag = f"{c['source']}->{c['target']}" + (f"/{c['backbone']}" if c["backbone"] != "videomae" else "")
        print(f"  {c['group']:5s} {c['method']:6s} {tag:28s} {c['median_diff']:+.4f}  "
              f"(rules: {['%.3f' % d for d in c['per_rule_diffs']]})")
    print()
    print("source-only cells:")
    for c in sorted(source_only, key=lambda c: c["median_diff"]):
        tag = f"{c['source']}->{c['target']}"
        print(f"  {c['group']:5s} {c['method']:6s} {tag:28s} {c['median_diff']:+.4f}  "
              f"(rules: {['%.3f' % d for d in c['per_rule_diffs']]})")

    u_stat, p_two_sided = stats.mannwhitneyu(adv_vals, so_vals, alternative="two-sided")
    # rank-biserial effect size from U (Wendt 1972): r = 1 - 2U/(n1*n2)
    r_rb = 1 - (2 * u_stat) / (n_adv * n_so)
    # common-language effect size: P(source-only value > adversarial value)
    cles = u_stat / (n_adv * n_so)

    print()
    print(f"median (adversarial) = {sorted(adv_vals)[len(adv_vals)//2]:.4f}")
    print(f"median (source-only) = {sorted(so_vals)[len(so_vals)//2]:.4f}")
    print(f"Mann-Whitney U (adversarial vs source-only, two-sided) = {u_stat:.1f}")
    print(f"p-value (two-sided) = {p_two_sided:.3e}")
    print(f"rank-biserial correlation r = {r_rb:.4f}")
    print(f"common-language effect size (P[source-only > adversarial]) = {cles:.4f}")

    print()
    print("--- D-4a robustness check: same test, per individual rule (no median-collapse) ---")
    # per_rule_diffs above is sorted ascending per cell (for the median lookup), so the
    # per-rule values are re-read from the raw rows here rather than un-sorted.
    adversarial_raw, source_only_raw = [], []
    with open(SRC, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] in ("dann", "cdan"):
                bucket = adversarial_raw
            elif r["method"] == "source_only":
                bucket = source_only_raw
            else:
                continue
            chance = chance_of(r["target"])
            bucket.append({rule: float(r[rule]) - chance for rule in LABEL_FREE_RULES})

    for rule in LABEL_FREE_RULES:
        adv_vals_r = [c[rule] for c in adversarial_raw]
        so_vals_r = [c[rule] for c in source_only_raw]
        u_r, p_r = stats.mannwhitneyu(adv_vals_r, so_vals_r, alternative="two-sided")
        complete = max(adv_vals_r) < min(so_vals_r)
        overlap_pairs = sum(1 for a in adv_vals_r for s in so_vals_r if a >= s)
        total_pairs = len(adv_vals_r) * len(so_vals_r)
        print(f"{rule:25s} median_adv={sorted(adv_vals_r)[len(adv_vals_r)//2]:+.4f} "
              f"median_so={sorted(so_vals_r)[len(so_vals_r)//2]:+.4f} "
              f"U={u_r:.1f} p={p_r:.3e} complete_separation={complete} "
              f"(adv>=so pairs: {overlap_pairs}/{total_pairs} = {100*overlap_pairs/total_pairs:.1f}%)")


if __name__ == "__main__":
    sys.exit(main())

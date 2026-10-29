#!/usr/bin/env python3
"""Recompute the campaign fold-to-fold spread statistic quoted in the noise-floor
paragraph (Section V-B, sec:noise_floor).

Cell definition (Section V-A, sec:protocol, "This paper reports several different
counts of cells..."): one (group, method, source, target[, backbone]) combination,
its oracle (target-validation) fold spread taken from the ``oracle_target_val_spread``
column of the campaign cell table.

Population, per the recovered methodology note in
analysis/NOISE_FLOOR_SUBSECTION_DRAFT_2026-09-08.tex (no runnable generator script
for this statistic was found elsewhere; this script is the first committed one):
  - source: analysis/swan_campaign_2026-09-07/reeval_cells.csv -- at release, this
    directory is renamed analysis/campaign_2026-09-07/ (D-2: neutralized of cluster
    naming, matching the name main.tex now cites); SRC below still points at the
    current on-disk path since nothing has been moved yet.
  - include groups E1 (source composition + mechanism pair), E4 (native label
    space), R2-4/IR (instrumented rerun) -- these are the four groups the
    noise-floor paragraph now names (corrected from an earlier draft's incorrect
    "cross-target-transfer", which has no rows in this cell table at all).
  - exclude E2 (zero-label reverse): its spread column is across three *seed*
    replicates at OBC's one fixed split, not folds -- not a "three-fold cell".
  - exclude E3 (backbone-family-ladder Swan-replication rows): single-fold
    calibration rows in this table, oracle_target_val_n == 1, no fold spread defined.

Run: python3 tools/noise_floor_fold_spread.py
"""
import csv
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# TODO at release packaging: rename to analysis/campaign_2026-09-07/reeval_cells.csv
# (D-2 canonical path) and update this constant to match.
SRC = REPO / "analysis" / "swan_campaign_2026-09-07" / "reeval_cells.csv"
INCLUDE_GROUPS = {"E1", "E4", "R2-4"}
NOISE_FLOOR = 0.077445


def main():
    rows = []
    with open(SRC, newline="") as f:
        for r in csv.DictReader(f):
            if r["group"] not in INCLUDE_GROUPS:
                continue
            n = int(r["oracle_target_val_n"])
            if n < 2:
                continue  # no fold spread defined (single-fold / calibration row)
            rows.append(r)

    spreads = [float(r["oracle_target_val_spread"]) for r in rows]
    n = len(spreads)
    mean = st.mean(spreads)
    median = st.median(spreads)

    print(f"{'group':6s} {'method':12s} {'source':14s} {'target':7s} {'backbone':9s} "
          f"{'n_folds':7s} {'spread':8s}")
    for r in sorted(rows, key=lambda r: -float(r["oracle_target_val_spread"])):
        print(f"{r['group']:6s} {r['method']:12s} {r['source']:14s} {r['target']:7s} "
              f"{r['backbone']:9s} {r['oracle_target_val_n']:7s} "
              f"{float(r['oracle_target_val_spread']):.4f}")

    print()
    print(f"n = {n}")
    print(f"mean   = {mean:.6f}  (rounds to {mean:.3f})")
    print(f"median = {median:.6f}  (rounds to {median:.3f})")
    exceeding = [r for r in rows if float(r["oracle_target_val_spread"]) > NOISE_FLOOR]
    print(f"cells exceeding the noise floor ({NOISE_FLOOR}): {len(exceeding)}")
    for r in exceeding:
        print(f"  {r['group']} {r['method']} {r['source']}->{r['target']} "
              f"({r['backbone']}): {float(r['oracle_target_val_spread']):.4f}")

    print()
    print("NOTE: this table has no distinct 'cross-target-transfer' group code; "
          "Table 9 (tab:cross_target) is a separate DANN-cross-eval pipeline not "
          "present in this campaign cell table at all. main.tex's noise-floor "
          "paragraph names the actual four groups (source composition, mechanism "
          "pair, native label space, instrumented rerun) as of D-1c.")


if __name__ == "__main__":
    sys.exit(main())

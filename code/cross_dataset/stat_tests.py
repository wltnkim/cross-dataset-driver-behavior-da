"""Paired t-tests for the cross-dataset DA paper.

Computes paired t-tests across (target, fold) pairs for the key claims:
  1) UDA(DANN ratio=0) vs Source-Only      -- DANN adapts but to near-chance only
  2) LE(target_only) vs SSDA(DANN) @ {0.5%, 1%, 2.5%} -- labels beat source DA
  3) LE(target_only) @ 0.5% vs UDA          -- a few labels >> full UDA
  4) LP-K400 vs LP-DANN @ 100%              -- DANN destroys linear separability
  5) Target-Only(full) vs UDA               -- supervised target dominates

Usage:
    python code/cross_dataset/stat_tests.py
"""
import csv
import glob
import json
import statistics
from collections import defaultdict
from scipy.stats import ttest_rel


MASTER_CSV = "code/results/analysis/master_results.csv"
TARGETS = ["driveandact", "dmd", "3mdad"]
FOLDS = ["0", "1", "2"]


def load_master():
    lookup = {}
    with open(MASTER_CSV) as f:
        for r in csv.DictReader(f):
            if r["bal_acc"]:
                lookup[(r["target"], r["method"], r["ratio"], r["fold"])] = float(r["bal_acc"])
    return lookup


def collect(lookup, method, ratio):
    out = []
    for t in TARGETS:
        for f in FOLDS:
            out.append(lookup.get((t, method, ratio, f)))
    return out


def load_lp(prefix):
    """Load linear-probe 100% bal_acc across (target, fold)."""
    out = []
    for t_alias in ["", "dmd_", "3mdad_"]:
        for f in FOLDS:
            patt = f"code/results/*_{prefix}_{t_alias}fold{f}/linear_probe_results.json"
            paths = glob.glob(patt)
            if prefix == "linear_probe":
                paths = [p for p in paths if "linear_probe_dann" not in p]
            if not paths:
                out.append(None)
                continue
            with open(paths[0]) as fh:
                d = json.load(fh)
            out.append(d["ratios"]["1.0"]["balanced_accuracy"])
    return out


def paired(name, a_label, a, b_label, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        print(f"{name}: n<2, skip")
        return
    av, bv = [p[0] for p in pairs], [p[1] for p in pairs]
    t, p = ttest_rel(av, bv)
    m1, m2 = sum(av) / len(av), sum(bv) / len(bv)
    print(f"{name}: n={len(pairs)}  {a_label}={m1:.4f}  {b_label}={m2:.4f}  Δ={m1-m2:+.4f}  t={t:.3f}  p={p:.4g}")


def main():
    lookup = load_master()

    print("# Paired t-tests (mean over 3-fold cross-validation)\n")

    print("## Test 1: UDA (DANN r=0) vs Source-Only")
    paired("UDA vs SO", "UDA", collect(lookup, "dann", "0.00"),
           "SO", collect(lookup, "source_only", ""))

    print("\n## Test 2: LE vs SSDA at low label ratios")
    for ratio_csv, ratio_paper in [("0.05", "0.5%"), ("0.10", "1%"), ("0.25", "2.5%")]:
        paired(f"LE vs SSDA @ {ratio_paper}",
               "LE", collect(lookup, "target_only", ratio_csv),
               "SSDA", collect(lookup, "dann", ratio_csv))

    print("\n## Test 3: LE 0.5% vs UDA")
    paired("LE@0.5% vs UDA",
           "LE", collect(lookup, "target_only", "0.05"),
           "UDA", collect(lookup, "dann", "0.00"))

    print("\n## Test 4: LP-K400 vs LP-DANN @ 100% labels")
    paired("LP-K400 vs LP-DANN",
           "K400", load_lp("linear_probe"),
           "DANN", load_lp("linear_probe_dann"))

    print("\n## Test 5: Target-Only (full) vs UDA")
    paired("TO vs UDA",
           "TO", collect(lookup, "target_only", ""),
           "UDA", collect(lookup, "dann", "0.00"))


if __name__ == "__main__":
    main()

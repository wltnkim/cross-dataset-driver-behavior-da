"""Comprehensive result collection for cross-dataset DA experiments.

Collects ALL results including SSDA/LE/cross-target into unified CSVs.
Run on login node (no GPU needed).

Usage:
    python -m cross_dataset.collect_all_results
"""

import json
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")


def _parse_dir_name(dir_name):
    """Parse experiment metadata from directory name.

    Patterns:
      {timestamp}_cross_{method}[_{target}][_r{ratio}]_fold{fold}
      {timestamp}_source_only_eval
    """
    info = {"dir_name": dir_name}

    # Extract fold
    m = re.search(r"_fold(\d+)$", dir_name)
    info["fold"] = int(m.group(1)) if m else None

    # Extract target: _dmd_ or _3mdad_ → else driveandact
    if "_dmd_" in dir_name or "_dmd_r" in dir_name:
        info["target"] = "dmd"
    elif "_3mdad_" in dir_name or "_3mdad_r" in dir_name:
        info["target"] = "3mdad"
    else:
        info["target"] = "driveandact"

    # Extract ratio (SSDA/LE): _r000, _r005, _r010, _r025, _r050
    m = re.search(r"_r(\d{3})_", dir_name)
    info["ratio"] = int(m.group(1)) / 100 if m else None  # r005 → 0.05

    # Extract method
    m = re.search(r"_cross_(\w+?)(?:_dmd|_3mdad)?(?:_r\d{3})?_fold\d+$", dir_name)
    info["method"] = m.group(1) if m else "unknown"

    return info


# Directories that are NOT part of any table population this script aggregates. Same
# list as paper/regen_*_figs.py and analyze_results.py MINUS the `_r\d{3}` clause, because
# this script deliberately keeps the SSDA/LE label-ratio runs (that is where
# master_results.csv's ratio rows come from). Added 2026-09-16: without it, a re-run
# today would sweep in reverse-direction, backbone-ablation, item1, source-only-start,
# sanity/smoke and `_uni_` runs that did not exist when master_results.csv was written
# (2026-04-13). NOTE: do not re-run this script to regenerate master_results.csv --
# that file is kept as a dated record (see stat_tests_extended.py's population table).
_EXC = re.compile(r"reverse|bbSmall|bbRandom|bbsmall|bbrandom|item1|srconly|"
                  r"tier2sanity|sanity|smoke|calib|_uni_|"
                  # revision-campaign runs (2026-08/09; separate CSVs under analysis/), which
                  # also carry an `_r000_` token that this script would otherwise read as ratio 0
                  r"_E[1-4]L?_|da573_random|broad573|narrow573|_R2-?4_")


def _dedup_key(rec):
    """Prefer the run with more trained epochs; only then the latest directory name
    (a 3-epoch June re-verification run must not outrank the complete 23-epoch April
    run for dmd/target_only/fold2). Runs without epochs_trained rank below any that have it."""
    ep = rec.get("epochs")
    return (ep if ep is not None else -1, rec["dir_name"])


def collect_all():
    """Collect all results into structured records."""
    records = []

    # 1. Standard experiments: results.json
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_cross_*_fold*", "results.json"))):
        dir_path = os.path.dirname(path)
        dir_name = os.path.basename(dir_path)
        if _EXC.search(dir_name):
            continue
        info = _parse_dir_name(dir_name)

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        # cbdann → cbdann_ent / cbdann_freq
        method = info["method"]
        if method == "cbdann":
            config_path = os.path.join(dir_path, "config.yaml")
            cbw = data.get("cbdann_class_weight")
            if not cbw and os.path.exists(config_path):
                with open(config_path) as cf:
                    for line in cf:
                        if "cbdann_class_weight:" in line:
                            cbw = line.strip().split(":", 1)[1].strip()
                            break
            if cbw == "entropy":
                method = "cbdann_ent"
            elif cbw == "frequency":
                method = "cbdann_freq"

        records.append({
            "target": info["target"],
            "method": method,
            "fold": info["fold"],
            "ratio": info["ratio"],
            "bal_acc": data.get("test_balanced_accuracy", float("nan")),
            "acc": data.get("test_accuracy", float("nan")),
            "epochs": data.get("epochs_trained"),
            "dir_name": dir_name,
            "dir_path": dir_path,
        })

    # 2. Cross-target eval: results_eval_*.json
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_cross_*_fold*", "results_eval_*.json"))):
        dir_path = os.path.dirname(path)
        dir_name = os.path.basename(dir_path)
        fname = os.path.basename(path)  # results_eval_dmd.json
        eval_target = fname.replace("results_eval_", "").replace(".json", "")
        info = _parse_dir_name(dir_name)

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        records.append({
            "target": eval_target,  # eval target, not training target
            "method": f"cross_{info['target']}_{info['method']}",
            "fold": info["fold"],
            "ratio": None,
            "bal_acc": data.get("test_balanced_accuracy", float("nan")),
            "acc": data.get("test_accuracy", float("nan")),
            "dir_name": dir_name,
            "dir_path": dir_path,
            "cross_eval": True,
            "train_target": info["target"],
        })

    # 3. Source-only (all targets)
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_source_only_eval*", "source_only_results.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        dir_name = os.path.basename(os.path.dirname(path))
        # Determine target and fold from directory name or result keys
        for tgt_key, tgt_name in [
            ("driveandact_test", "driveandact"),
            ("dmd_test", "dmd"),
            ("3mdad_test", "3mdad"),
        ]:
            tgt_data = data.get(tgt_key, {})
            if not tgt_data:
                continue
            # Extract fold from dir name if present (e.g., _fold0, _fold1)
            m = re.search(r"_fold(\d+)", dir_name)
            fold = int(m.group(1)) if m else 0
            records.append({
                "target": tgt_name,
                "method": "source_only",
                "fold": fold,
                "ratio": None,
                "bal_acc": tgt_data.get("balanced_accuracy", float("nan")),
                "acc": tgt_data.get("accuracy", float("nan")),
                "dir_name": dir_name,
                "dir_path": os.path.dirname(path),
            })

    return records


def dedup(records):
    """Keep latest run per unique key. SSDA/LE use (target,method,fold,ratio)."""
    latest = {}
    for rec in records:
        key = (rec["target"], rec["method"], rec["fold"], rec.get("ratio"))
        if rec.get("cross_eval"):
            key = (rec["target"], rec["method"], rec["fold"], "cross_eval")
        if key not in latest or _dedup_key(rec) > _dedup_key(latest[key]):
            latest[key] = rec
    return list(latest.values())


def print_table(title, records, extra_cols=None):
    """Print a formatted table."""
    if not records:
        print(f"\n{title}: (no data)\n")
        return

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Group by method
    by_method = defaultdict(dict)
    for r in records:
        by_method[r["method"]][r["fold"]] = r

    folds = sorted(set(r["fold"] for r in records))

    # Header
    hdr = f"{'Method':<20}"
    for f in folds:
        hdr += f"{'Fold '+str(f):>10}"
    hdr += f"{'Mean':>10}{'Std':>8}"
    print(hdr)
    print("-" * len(hdr))

    # Rows in order
    METHOD_ORDER = [
        "source_only", "target_only", "finetune",
        "dann", "cdan", "mcc", "bnm", "cbdann_ent", "cbdann_freq",
        "shot", "cleanadapt", "csfda",
        "bn_adapt", "tent", "tta", "vitta",
    ]
    methods = [m for m in METHOD_ORDER if m in by_method]
    methods += sorted(m for m in by_method if m not in METHOD_ORDER)

    for method in methods:
        fold_data = by_method[method]
        vals = []
        row = f"{method:<20}"
        for f in folds:
            if f in fold_data:
                v = fold_data[f]["bal_acc"]
                vals.append(v)
                row += f"{v:>10.4f}"
            else:
                row += f"{'—':>10}"
        if len(vals) > 0:
            row += f"{np.mean(vals):>10.4f}"
            row += f"{np.std(vals):>8.4f}" if len(vals) > 1 else f"{'—':>8}"
        print(row)
    print()


def save_master_csv(records, path):
    """Save all records as CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("target,method,fold,ratio,bal_acc,acc,dir_name\n")
        for r in sorted(records, key=lambda x: (x["target"], x["method"], x.get("fold", 0), x.get("ratio") or 0)):
            ratio = r.get("ratio")
            ratio_str = f"{ratio:.2f}" if ratio is not None else ""
            f.write(f"{r['target']},{r['method']},{r['fold']},{ratio_str},"
                    f"{r['bal_acc']:.6f},{r['acc']:.6f},{r['dir_name']}\n")
    print(f"[Saved] {path}")


def main():
    records = collect_all()
    records = dedup(records)
    print(f"Total unique results: {len(records)}")

    # Split by category
    standard = [r for r in records if r.get("ratio") is None and not r.get("cross_eval")]
    ssda = [r for r in records if r.get("ratio") is not None and "dann" in r["method"]]
    le = [r for r in records if r.get("ratio") is not None and "target_only" in r["method"]]
    cross_eval = [r for r in records if r.get("cross_eval")]

    # Per-target standard tables
    for tgt in ["driveandact", "dmd", "3mdad"]:
        tgt_recs = [r for r in standard if r["target"] == tgt]
        print_table(f"Source-Available + SFDA/TTA — {tgt.upper()}", tgt_recs)

    # SSDA tables per target
    for tgt in ["driveandact", "dmd", "3mdad"]:
        tgt_ssda = [r for r in ssda if r["target"] == tgt]
        if tgt_ssda:
            # Group by ratio
            by_ratio = defaultdict(list)
            for r in tgt_ssda:
                by_ratio[r["ratio"]].append(r)
            print(f"\n{'='*80}")
            print(f"  SSDA (DANN) — {tgt.upper()}")
            print(f"{'='*80}")
            hdr = f"{'Ratio':<10}"
            folds = sorted(set(r["fold"] for r in tgt_ssda))
            for f in folds:
                hdr += f"{'Fold '+str(f):>10}"
            hdr += f"{'Mean':>10}{'Std':>8}"
            print(hdr)
            print("-" * len(hdr))
            for ratio in sorted(by_ratio.keys()):
                recs = by_ratio[ratio]
                vals = []
                row = f"{ratio:<10.2f}"
                fold_map = {r["fold"]: r for r in recs}
                for f in folds:
                    if f in fold_map:
                        v = fold_map[f]["bal_acc"]
                        vals.append(v)
                        row += f"{v:>10.4f}"
                    else:
                        row += f"{'—':>10}"
                if vals:
                    row += f"{np.mean(vals):>10.4f}"
                    row += f"{np.std(vals):>8.4f}" if len(vals) > 1 else f"{'—':>8}"
                print(row)
            print()

    # Label Efficiency tables per target
    for tgt in ["driveandact", "dmd", "3mdad"]:
        tgt_le = [r for r in le if r["target"] == tgt]
        if tgt_le:
            by_ratio = defaultdict(list)
            for r in tgt_le:
                by_ratio[r["ratio"]].append(r)
            print(f"\n{'='*80}")
            print(f"  Label Efficiency (Target-Only) — {tgt.upper()}")
            print(f"{'='*80}")
            folds = sorted(set(r["fold"] for r in tgt_le))
            hdr = f"{'Ratio':<10}"
            for f in folds:
                hdr += f"{'Fold '+str(f):>10}"
            hdr += f"{'Mean':>10}{'Std':>8}"
            print(hdr)
            print("-" * len(hdr))
            for ratio in sorted(by_ratio.keys()):
                recs = by_ratio[ratio]
                vals = []
                row = f"{ratio:<10.2f}"
                fold_map = {r["fold"]: r for r in recs}
                for f in folds:
                    if f in fold_map:
                        v = fold_map[f]["bal_acc"]
                        vals.append(v)
                        row += f"{v:>10.4f}"
                    else:
                        row += f"{'—':>10}"
                if vals:
                    row += f"{np.mean(vals):>10.4f}"
                    row += f"{np.std(vals):>8.4f}" if len(vals) > 1 else f"{'—':>8}"
                print(row)
            print()

    # Cross-target eval
    if cross_eval:
        print(f"\n{'='*80}")
        print(f"  Cross-Target Evaluation")
        print(f"{'='*80}")
        for r in sorted(cross_eval, key=lambda x: (x["method"], x["fold"])):
            print(f"  {r['method']} fold{r['fold']} → {r['target']}: BalAcc={r['bal_acc']:.4f}")
        print()

    # Missing results summary
    print(f"\n{'='*80}")
    print(f"  Missing Results Summary")
    print(f"{'='*80}")
    EXPECTED_METHODS = ["target_only", "finetune", "dann", "cdan", "mcc", "bnm",
                        "cbdann_ent", "cbdann_freq", "shot", "cleanadapt", "csfda",
                        "bn_adapt", "tent", "tta", "vitta"]
    for tgt in ["driveandact", "dmd", "3mdad"]:
        tgt_std = {(r["method"], r["fold"]) for r in standard if r["target"] == tgt}
        missing = []
        for m in EXPECTED_METHODS:
            for f in [0, 1, 2]:
                if (m, f) not in tgt_std:
                    missing.append(f"{m}_fold{f}")
        if missing:
            print(f"  {tgt}: {', '.join(missing)}")

    EXPECTED_RATIOS_SSDA = [0.0, 0.05, 0.10, 0.25]
    EXPECTED_RATIOS_LE = [0.05, 0.10, 0.25, 0.50]
    for tgt in ["driveandact", "dmd", "3mdad"]:
        ssda_keys = {(r["fold"], r["ratio"]) for r in ssda if r["target"] == tgt}
        le_keys = {(r["fold"], r["ratio"]) for r in le if r["target"] == tgt}
        missing_ssda = []
        missing_le = []
        for f in [0, 1, 2]:
            for ratio in EXPECTED_RATIOS_SSDA:
                if (f, ratio) not in ssda_keys:
                    missing_ssda.append(f"r{int(ratio*100):03d}_fold{f}")
            for ratio in EXPECTED_RATIOS_LE:
                if (f, ratio) not in le_keys:
                    missing_le.append(f"r{int(ratio*100):03d}_fold{f}")
        if missing_ssda:
            print(f"  {tgt} SSDA: {', '.join(missing_ssda)}")
        if missing_le:
            print(f"  {tgt} LE: {', '.join(missing_le)}")

    print()

    # Save master CSV
    save_master_csv(records, os.path.join(ANALYSIS_DIR, "master_results.csv"))


if __name__ == "__main__":
    main()

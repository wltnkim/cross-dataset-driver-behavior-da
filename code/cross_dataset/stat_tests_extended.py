"""Extended statistical rigor for the cross-dataset DA paper (reviewer-defense).

Adds, on top of the paired t-tests in stat_tests.py:
  - Cohen's d_z (paired effect size) + 95% bootstrap CI
  - 95% bootstrap CI for the mean paired difference
  - Holm-Bonferroni correction across the full family of confirmatory tests

Non-destructive: writes nothing, only prints. Deterministic (fixed RNG seed).
To compare the output against the manuscript line by line, run
tools/verify_paper_stats.py, which wraps this module.

Usage:
    python code/cross_dataset/stat_tests_extended.py
    (needs numpy + scipy; on this cluster: `module load anaconda3/2024.06`)

======================================================================
POPULATION-TO-TEST BINDING  (read this before trusting any number)
======================================================================
The seven tests read from FOUR sources. Which test reads which is stated
here, not implied by which helper each test calls. Three of the seven
bypass the aggregate CSV for one side; the reason is chronological, not a
preference: the CSV is dated 2026-04-13 and two of the populations the
manuscript reports were produced after that date (2026-06-07, 2026-07-15).

  Source A  MASTER      results/analysis/master_results.csv  (dated 2026-04-13,
            a frozen record; never regenerated). Its zero-label DANN rows
            (method="dann", ratio="0.00") are the `*_cross_dann_*_r000_fold*`
            runs (patience=15). Its "source_only" rows are the April
            source-only evaluations, superseded by Source D below.
  Source B  UNI_DANN    the `*_cross_dann_*_r000_uni_fold*` run directories,
            read directly (9 dirs: 3 targets x 3 folds). Created 2026-06-07,
            after Source A was written. This is the zero-label DANN population
            that Table 5 (tab:uda_failure) prints (0.232 / 0.312 / 0.308).
  Source C  LP          linear_probe_results.json per run directory, ratio "1.0":
            LP(linear_probe)          K400 frozen features, no adaptation
            LP(linear_probe_uda_dann) frozen features of the UNI_DANN backbones
            (NOT linear_probe_dann: that prefix probes the fully-supervised
            DANN of Table 4 and is the sensitivity check in the footnote of
            Sec. VI-B, not this family's test.)
  Source D  SRCONLY_33  the `20260715fix_source_only_eval_<target>_fold<f>`
            directories, read directly (9 dirs): the Source-Only (33-class,
            probability-averaged) model of Table 4 (0.249 / 0.240 / 0.266),
            re-evaluated 2026-07-15 after a fix. Source A's older April
            source_only rows do NOT reproduce Table 4 and are not used.

  test  name                          side A                   side B
  ----  ----------------------------  -----------------------  ------------------------
   1    UDA vs Source-Only            MASTER dann   r=0.00     SRCONLY_33 (Source D)
   2    LE vs SSDA @5%                MASTER target_only 0.05  MASTER dann 0.05
   3    LE vs SSDA @10%               MASTER target_only 0.10  MASTER dann 0.10
   4    LE vs SSDA @25%               MASTER target_only 0.25  MASTER dann 0.25
   5    LE@5% vs UDA                  MASTER target_only 0.05  UNI_DANN   (Source B)
   6    LP-K400 vs LP-UDA-DANN        LP(linear_probe)         LP(linear_probe_uda_dann)
   7    Target-Only(full) vs UDA      MASTER target_only ""    UNI_DANN   (Source B)

  Tests 1 and 7 both compare against "zero-label DANN" but from DIFFERENT
  populations (A vs B). That is what reproduces the manuscript: test 1's
  printed Delta=+0.039 comes from Source A's DANN rows against Source D,
  test 7's printed +0.670 / t=33.4 from Source B; swapping either breaks
  the other (verified 2026-09-16). This is a
  documented historical asymmetry, not an inconsistency to "fix" by
  forcing one population.
======================================================================
"""
import csv
import glob
import json
import os
import re
import numpy as np
from scipy.stats import ttest_rel

# Self-locating: this file lives at <repo>/code/cross_dataset/. Override with ARR_ROOT.
ARR = os.environ.get("ARR_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(ARR, "results")
MASTER_CSV = os.path.join(RES, "analysis", "master_results.csv")
TARGETS = ["driveandact", "dmd", "3mdad"]
FOLDS = ["0", "1", "2"]
B = 10000           # bootstrap resamples
SEED = 0

# Manuscript location of each test's printed statistics (for readers cross-checking).
TEST_SITES = {
    "UDA vs SO":            "main.tex Sec. V-D, tab:uda_failure prose (Delta, Holm p)",
    "LE vs SSDA @0.5%":     "main.tex Sec. V-E, label-ratio prose (n, Delta, t, p, d_z, CI, Holm p)",
    "LE vs SSDA @1%":       "main.tex Sec. V-E, label-ratio prose (n, Delta, t, p, d_z, Holm p)",
    "LE vs SSDA @2.5%":     "not printed (family member only)",
    "LE@0.5% vs UDA":       "not printed (family member only)",
    "LP-K400 vs LP-DANN":   "main.tex Sec. VI-B linear-probing paragraph + intro bullet (Delta, t, p, d_z, Holm p)",
    "TO(full) vs UDA":      "main.tex Sec. V-D, tab:uda_failure prose (Delta, t, p, d_z, Holm p)",
}


def load_master():
    if not os.path.isfile(MASTER_CSV):
        raise FileNotFoundError(f"{MASTER_CSV} not found (set ARR_ROOT to the repository root)")
    lookup = {}
    with open(MASTER_CSV) as f:
        for r in csv.DictReader(f):
            if r["bal_acc"]:
                lookup[(r["target"], r["method"], r["ratio"], r["fold"])] = float(r["bal_acc"])
    return lookup


def collect(lookup, method, ratio):
    return [lookup.get((t, method, ratio, f)) for t in TARGETS for f in FOLDS]


def _exactly_one(paths, what):
    """A selection rule that silently picks among several candidates is how a
    2-epoch smoke run and a superseded pre-fix run got into figures/scripts in
    this project. Refuse to guess."""
    if len(paths) != 1:
        raise RuntimeError(f"{what}: expected exactly 1 candidate directory, found {len(paths)}: "
                           f"{[os.path.basename(os.path.dirname(p)) for p in paths]}")
    return paths[0]


def load_lp(prefix):
    """Linear-probe balanced accuracy at ratio 1.0, per (target, fold), Source C.
    Matches ONLY '<ts>_<prefix>_[<target>_]fold<f>' -- the prefix must be followed
    directly by the target/fold suffix, so linear_probe does not pick up
    linear_probe_dann / linear_probe_uda_dann / linear_probe_bbsmall."""
    out = []
    for t_alias in ["", "dmd_", "3mdad_"]:
        for f in FOLDS:
            patt = os.path.join(RES, f"*_{prefix}_{t_alias}fold{f}", "linear_probe_results.json")
            paths = sorted(glob.glob(patt))
            tail = f"_{prefix}_{t_alias}fold{f}"
            paths = [p for p in paths if os.path.basename(os.path.dirname(p)).endswith(tail)]
            p = _exactly_one(paths, f"load_lp({prefix}) target={t_alias or 'driveandact_'}fold{f}")
            with open(p) as fh:
                d = json.load(fh)
            out.append(d["ratios"]["1.0"]["balanced_accuracy"])
    return out


def load_uni_dann():
    """Zero-label DANN test balanced accuracy from the `_r000_uni_` run directories
    (Source B), per (target, fold). Bypasses master_results.csv on purpose: that
    CSV is dated 2026-04-13 and these runs were created 2026-06-07, so it cannot
    contain them; the manuscript's test-5/test-7 statistics were computed from
    these runs (verified 2026-09-16), and the CSV is kept as a dated record
    rather than regenerated."""
    out = []
    for t in TARGETS:
        mid = "" if t == "driveandact" else f"{t}_"
        for f in FOLDS:
            patt = os.path.join(RES, f"*_cross_dann_{mid}r000_uni_fold{f}", "results.json")
            paths = sorted(glob.glob(patt))
            p = _exactly_one(paths, f"load_uni_dann target={t} fold{f}")
            with open(p) as fh:
                d = json.load(fh)
            if d.get("mode") != "dann":
                raise RuntimeError(f"{p}: mode={d.get('mode')!r}, expected 'dann'")
            out.append(d["test_balanced_accuracy"])
    return out


def load_srconly_33():
    """Source-Only (33-class, probability-averaged) test balanced accuracy per
    (target, fold) from the `20260715fix_source_only_eval_*` directories
    (Source D). Bypasses master_results.csv on purpose: the CSV's source_only
    rows are the April 2026 evaluations, which the 2026-07-15 fix superseded;
    only the fixed runs reproduce Table 4's printed 0.249 / 0.240 / 0.266 and
    test 1's printed Delta=+0.039 (verified 2026-09-16)."""
    out = []
    for t in TARGETS:
        for f in FOLDS:
            patt = os.path.join(RES, f"20260715fix_source_only_eval_{t}_fold{f}", "source_only_results.json")
            paths = sorted(glob.glob(patt))
            p = _exactly_one(paths, f"load_srconly_33 target={t} fold{f}")
            with open(p) as fh:
                d = json.load(fh)
            out.append(d[f"{t}_test"]["balanced_accuracy"])
    return out


def analyze(name, a, b, rng):
    """Paired analysis. Returns dict with t, p, Cohen's d_z, bootstrap CIs."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    av = np.array([p[0] for p in pairs], float)
    bv = np.array([p[1] for p in pairs], float)
    diff = av - bv
    n = len(diff)
    t, p = ttest_rel(av, bv)
    sd = diff.std(ddof=1)
    d_z = diff.mean() / sd if sd > 0 else float("inf")
    # bootstrap CIs (resample paired differences)
    idx = rng.integers(0, n, size=(B, n))
    boot = diff[idx]                                  # (B, n)
    boot_mean = boot.mean(axis=1)
    boot_sd = boot.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_dz = np.where(boot_sd > 0, boot_mean / boot_sd, np.nan)
    ci_delta = np.percentile(boot_mean, [2.5, 97.5])
    ci_dz = np.nanpercentile(boot_dz, [2.5, 97.5])
    return dict(name=name, n=n, m1=av.mean(), m2=bv.mean(), delta=diff.mean(),
                t=float(t), p=float(p), dz=d_z, ci_delta=ci_delta, ci_dz=ci_dz)


def holm(results, alpha=0.05):
    """Holm-Bonferroni across the family. Returns name->(p_adj, reject, rank, mult, floored).
    `floored` is True when monotonicity raised this test's value above its own
    rank*p -- expected for adjacent raw p's (three tests share 0.1052 here)."""
    m = len(results)
    order = sorted(range(m), key=lambda i: results[i]["p"])
    out = {}
    running = 0.0
    for rank, i in enumerate(order):
        mult = m - rank
        adj = min(1.0, mult * results[i]["p"])
        floored = adj < running
        running = max(running, adj)               # enforce monotonicity
        out[results[i]["name"]] = (running, running < alpha, rank + 1, mult, floored)
    return out


def run_family():
    """Compute all seven tests in the fixed order (the bootstrap RNG stream depends
    on it). Returns (results, holm_map)."""
    lookup = load_master()
    rng = np.random.default_rng(SEED)
    uni = load_uni_dann()
    srconly = load_srconly_33()
    tests = [
        ("UDA vs SO",          collect(lookup, "dann", "0.00"),        srconly),
        ("LE vs SSDA @0.5%",   collect(lookup, "target_only", "0.05"), collect(lookup, "dann", "0.05")),
        ("LE vs SSDA @1%",     collect(lookup, "target_only", "0.10"), collect(lookup, "dann", "0.10")),
        ("LE vs SSDA @2.5%",   collect(lookup, "target_only", "0.25"), collect(lookup, "dann", "0.25")),
        ("LE@0.5% vs UDA",     collect(lookup, "target_only", "0.05"), uni),
        ("LP-K400 vs LP-DANN", load_lp("linear_probe"),                load_lp("linear_probe_uda_dann")),
        ("TO(full) vs UDA",    collect(lookup, "target_only", ""),     uni),
    ]
    res = [analyze(nm, a, b, rng) for nm, a, b in tests]
    return res, holm(res)


def main():
    res, holm_map = run_family()
    print(f"# Extended stats  (bootstrap B={B}, seed={SEED}, Cohen's d_z = paired)\n")
    hdr = (f"{'Test':22s} {'n':>2s} {'Δ':>8s} {'95% CI(Δ)':>18s} {'t':>7s} {'p':>10s} "
           f"{'rank':>4s} {'x':>2s} {'p_Holm':>9s} {'d_z':>6s} {'95% CI(d_z)':>16s}")
    print(hdr)
    print("-" * len(hdr))
    for r in res:
        pa, rej, rank, mult, floored = holm_map[r["name"]]
        star = "*" if rej else " "
        fl = "^" if floored else " "
        print(f"{r['name']:22s} {r['n']:>2d} {r['delta']:+8.4f} "
              f"[{r['ci_delta'][0]:+.3f},{r['ci_delta'][1]:+.3f}] "
              f"{r['t']:7.3f} {r['p']:10.3g} {rank:>4d} {mult:>2d} {pa:9.3g}{star}{fl}"
              f"{r['dz']:6.2f} [{r['ci_dz'][0]:.2f},{r['ci_dz'][1]:.2f}]")
    print(f"\n* = significant at alpha=0.05 after Holm-Bonferroni (family size = {len(res)}).")
    print("^ = corrected p raised to the running maximum by Holm's monotonicity step.")
    print("Cohen's d_z interpretation: 0.2 small, 0.5 medium, 0.8 large.")
    print("\nManuscript sites:")
    for r in res:
        print(f"  {r['name']:22s} {TEST_SITES[r['name']]}")


if __name__ == "__main__":
    main()

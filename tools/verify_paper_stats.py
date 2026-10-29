#!/usr/bin/env python3
"""Verify the manuscript's paired statistical claims against the released data, line by line.

    python3 tools/verify_paper_stats.py          # exit 0 = every printed statistic reproduces
    (needs numpy + scipy; on this cluster: `module load anaconda3/2024.06`)

WHAT THIS CHECKS
  The family of seven preregistered paired tests named in the manuscript's footnote in
  Sec. V-E ("family of seven confirmatory paired tests ... computed by our released script
  code/cross_dataset/stat_tests_extended.py"): it runs that script's `run_family()` on the
  released per-run summaries under results/ and compares every statistic the manuscript
  PRINTS (Delta, t, raw p, Cohen's d_z, bootstrap CI, Holm-corrected p, n) at the
  manuscript's own printed precision. Two family members (LE-vs-SSDA at 25%, LE@5%-vs-UDA)
  have no printed statistic; they are still computed and shown, because they take part
  in the Holm ordering that determines the other five corrected p-values.

WHAT THIS DOES NOT CHECK  (do not mistake it for a full reproduction of the paper)
  - Any table or figure value. Those come from the campaign CSVs and generator scripts
    listed in RELEASE_NOTES.md / PACKAGING_PLAN.md.
  - The 107-run campaign's selection-rule test (Mann-Whitney) and the noise floor:
    tools/selection_rule_test.py and tools/noise_floor_fold_spread.py.
  - Training or evaluation itself: every number here is read from summaries that the
    released code already wrote (results/*/results.json, source_only_results.json,
    linear_probe_results.json, results/analysis/master_results.csv).

WHY THE FAMILY READS FROM FOUR SOURCES
  See the population-to-test binding table at the top of stat_tests_extended.py. In
  short: results/analysis/master_results.csv is a dated (2026-04-13) record, and two of
  the populations the manuscript reports were produced after it (the zero-label DANN
  reruns of 2026-06-07 and the fixed Source-Only evaluations of 2026-07-15), so three
  tests read one side directly from run directories. The CSV is deliberately not
  regenerated.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARR = os.environ.get("ARR_ROOT", os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ARR, "code", "cross_dataset"))
os.environ.setdefault("ARR_ROOT", ARR)

import stat_tests_extended as ste  # noqa: E402

# Every statistic the manuscript prints for this family, at its printed precision.
# (test name, statistic, manuscript location, printed string)
PRINTED = [
    ("UDA vs SO",          "delta",  "Sec. V-D prose after Table 5",        "+0.039"),
    ("UDA vs SO",          "holm",   "Sec. V-D prose after Table 5",        "0.105"),
    ("LE vs SSDA @0.5%",   "n",      "Sec. V-E label-ratio paragraph",      "6"),
    ("LE vs SSDA @0.5%",   "delta",  "Sec. V-E label-ratio paragraph",      "+0.090"),
    ("LE vs SSDA @0.5%",   "t",      "Sec. V-E label-ratio paragraph",      "5.51"),
    ("LE vs SSDA @0.5%",   "p",      "Sec. V-E label-ratio paragraph",      "2.7e-3"),
    ("LE vs SSDA @0.5%",   "dz",     "Sec. V-E label-ratio paragraph",      "2.2"),
    ("LE vs SSDA @0.5%",   "ci_lo",  "Sec. V-E label-ratio paragraph",      "+0.062"),
    ("LE vs SSDA @0.5%",   "ci_hi",  "Sec. V-E label-ratio paragraph",      "+0.120"),
    ("LE vs SSDA @0.5%",   "holm",   "Sec. V-E label-ratio paragraph",      "1.1e-2"),
    ("LE vs SSDA @1%",     "n",      "Sec. V-E label-ratio paragraph",      "9"),
    ("LE vs SSDA @1%",     "delta",  "Sec. V-E label-ratio paragraph",      "+0.038"),
    ("LE vs SSDA @1%",     "t",      "Sec. V-E label-ratio paragraph",      "2.53"),
    ("LE vs SSDA @1%",     "p",      "Sec. V-E label-ratio paragraph",      "0.035"),
    ("LE vs SSDA @1%",     "dz",     "Sec. V-E label-ratio paragraph",      "0.84"),
    ("LE vs SSDA @1%",     "holm",   "Sec. V-E label-ratio paragraph",      "0.105"),
    ("LP-K400 vs LP-DANN", "delta",  "Sec. VI-B linear-probing paragraph",  "+0.104"),
    ("LP-K400 vs LP-DANN", "t",      "Sec. VI-B linear-probing paragraph",  "4.84"),
    ("LP-K400 vs LP-DANN", "p",      "Sec. VI-B linear-probing paragraph",  "1.3e-3"),
    ("LP-K400 vs LP-DANN", "dz",     "Sec. VI-B linear-probing paragraph",  "1.61"),
    ("LP-K400 vs LP-DANN", "holm",   "Sec. VI-B paragraph + Intro bullet",  "6.5e-3"),
    ("TO(full) vs UDA",    "delta",  "Sec. V-D prose after Table 5",        "+0.670"),
    ("TO(full) vs UDA",    "t",      "Sec. V-D prose after Table 5",        "33.4"),
    ("TO(full) vs UDA",    "p",      "Sec. V-D prose after Table 5",        "7.1e-10"),
    ("TO(full) vs UDA",    "dz",     "Sec. V-D prose after Table 5",        "11.1"),
    ("TO(full) vs UDA",    "holm",   "Sec. V-D prose after Table 5",        "5.0e-9"),
]


def render(value, printed):
    """Format `value` the way the manuscript printed `printed` (same decimals, or 2
    significant figures for scientific notation), so the comparison is at printed
    precision rather than on the raw float."""
    if "e" in printed:
        return f"{float(value):.1e}".replace("e-0", "e-")
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    sign = "+" if printed.startswith("+") else ""
    return f"{float(value):{sign}.{decimals}f}"


def main():
    res, holm_map = ste.run_family()
    by = {r["name"]: r for r in res}

    def stat(r, key):
        if key == "holm":
            return holm_map[r["name"]][0]
        if key == "ci_lo":
            return r["ci_delta"][0]
        if key == "ci_hi":
            return r["ci_delta"][1]
        return r[key]

    width = max(len(x[0]) for x in PRINTED)
    print("Paper statistic verification  (stat_tests_extended.run_family, "
          f"B={ste.B}, seed={ste.SEED})\n")
    print(f"{'test':<{width}}  {'stat':<6} {'manuscript location':<36} {'printed':>9} {'computed':>9}  ok")
    print("-" * (width + 78))
    bad = 0
    for name, key, where, printed in PRINTED:
        computed = render(stat(by[name], key), printed)
        # normalise the printed side through the same renderer so "2.7e-3" == "2.7e-3"
        expect = render(float(printed), printed)
        ok = computed == expect
        bad += 0 if ok else 1
        print(f"{name:<{width}}  {key:<6} {where:<36} {expect:>9} {computed:>9}  {'yes' if ok else 'NO'}")

    print("\nFull family (including the two members the manuscript does not print):")
    print(f"{'rank':>4} {'test':<{width}} {'n':>2} {'delta':>8} {'t':>8} {'raw p':>10} {'x':>2} {'Holm p':>9}  floored")
    for name, (pa, rej, rank, mult, floored) in sorted(holm_map.items(), key=lambda kv: kv[1][2]):
        r = by[name]
        print(f"{rank:>4} {name:<{width}} {r['n']:>2} {r['delta']:+8.4f} {r['t']:8.3f} "
              f"{r['p']:10.3g} {mult:>2} {pa:9.3g}  {'yes' if floored else ''}")
    print("\nfloored = Holm's monotonicity step raised this value to the running maximum; "
          "three tests legitimately share 0.105 for that reason.")
    if bad:
        print(f"\nFAIL: {bad} printed statistic(s) do not reproduce.")
        sys.exit(1)
    print(f"\nPASS: all {len(PRINTED)} printed statistics reproduce at printed precision.")


if __name__ == "__main__":
    main()

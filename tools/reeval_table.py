#!/usr/bin/env python3
"""[ARR 2026-09-05] TASK 1 step 4 -- the tidy selection-rule table.

Consumes analysis/reeval/{3mdad,dmd}_f{0,1,2}.json (written by tools/reeval_rules.py) and
emits:
  analysis/reeval_rules_table.csv   one row per run: run_id, group, method, source, target,
                                    fold, test BA under each of the four rules, the epoch each
                                    rule picked, the control outcome.
  analysis/reeval_cells.csv         per (group, method, source, target) cell: mean and fold
                                    spread (max-min) of test BA under each rule, n folds.
  analysis/reeval_report.md         the same, plus every control failure and missing
                                    checkpoint, listed explicitly.

FAILS LOUD. Exits nonzero if any expected shard is missing, if any run is absent, if any
control did not reproduce results.json exactly, or if any rule cell is empty. A 0-row table
is an error, not an empty success.
"""
import csv, json, os, sys, glob, statistics, collections

ARR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
RULES = ["oracle_target_val", "source_val", "last_epoch", "min_target_entropy"]
SHARDS = ["3mdad_f0", "3mdad_f1", "3mdad_f2", "dmd_f0", "dmd_f1", "dmd_f2"]


def main():
    appr = {r["run_id"].replace("-", ""): r
            for r in csv.DictReader(open(os.path.join(ARR, "MATRIX_APPROVED.csv")))}
    expected = {r["run_tag"] for r in
                json.load(open(os.path.join(ARR, "analysis", "reeval_inventory.json")))
                if r["status"] == "OK"}
    problems = []
    runs = []
    for sh in SHARDS:
        p = os.path.join(ARR, "analysis", "reeval", sh + ".json")
        if not os.path.isfile(p):
            problems.append("MISSING SHARD %s.json" % sh); continue
        d = json.load(open(p))
        if d.get("RED_TEST_ARTEFACT"):
            problems.append("%s.json is a RED TEST artefact - refusing to use it" % sh); continue
        if d.get("stopped_early"):
            problems.append("%s stopped early on budget at %s (%d/%d runs)"
                            % (sh, d["stopped_early"]["at_run"], d["stopped_early"]["done"],
                               d["stopped_early"]["total"]))
        runs.extend(d["runs"])

    seen = {r["run_tag"] for r in runs}
    for t in sorted(expected - seen):
        problems.append("NO RESULT for run %s" % t)

    rows = []
    for r in runs:
        a = appr[r["run_tag"]]
        row = dict(run_id=a["run_id"], group=a["group"], method=a["mode"], source=a["source"],
                   target=a["target"], fold=int(a["fold"]))
        for rule in RULES:
            c = r["by_rule"][rule]
            row["%s_epoch" % rule] = c["epoch"]
            row["%s_test_ba" % rule] = c["test_ba"]
            if c["test_ba"] is None:
                problems.append("%s: rule %s has NO test BA (epoch %s)"
                                % (a["run_id"], rule, c["epoch"]))
        row["reported_test_ba"] = r["test_ba_reported"]
        row["control_best_model_test_ba"] = (r.get("control_best_model") or {}).get("test_ba")
        row["control_match_exact"] = r.get("control_match_exact")
        row["oracle_ckpt_matches_results_json"] = r.get("oracle_ckpt_matches_results_json")
        rows.append(row)
        if r.get("control_match_exact") is not True:
            problems.append("CONTROL FAILED for %s: %s" % (a["run_id"], "; ".join(r["notes"]) or
                                                           "; ".join(r["load_errors"])))
        if r.get("oracle_ckpt_matches_results_json") is not True:
            problems.append("ORACLE CHECKPOINT does not reproduce results.json for %s"
                            % a["run_id"])
        for le in r["load_errors"]:
            problems.append("LOAD ERROR %s: %s" % (a["run_id"], le))

    rows.sort(key=lambda x: (x["group"], x["method"], x["source"], x["target"], x["fold"]))
    if not rows:
        print("NO ROWS - refusing to write an empty table."); sys.exit(3)
    with open(os.path.join(ARR, "analysis", "reeval_rules_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["group"], r["method"], r["source"], r["target"])].append(r)
    crows = []
    for k, v in sorted(cells.items()):
        c = dict(group=k[0], method=k[1], source=k[2], target=k[3], n_folds=len(v),
                 folds=",".join(str(x["fold"]) for x in sorted(v, key=lambda y: y["fold"])))
        for rule in RULES:
            vals = [x["%s_test_ba" % rule] for x in v if x["%s_test_ba" % rule] is not None]
            c["%s_mean" % rule] = statistics.mean(vals) if vals else None
            c["%s_spread" % rule] = (max(vals) - min(vals)) if len(vals) > 1 else None
            c["%s_n" % rule] = len(vals)
        crows.append(c)
    with open(os.path.join(ARR, "analysis", "reeval_cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crows[0].keys())); w.writeheader(); w.writerows(crows)

    with open(os.path.join(ARR, "analysis", "reeval_report.md"), "w") as f:
        f.write("# Selection-rule re-evaluation (ISAAC, %d runs)\n\n" % len(rows))
        f.write("Checkpoints scored through the trainer's INTERNAL eval path; per-run control "
                "= best_model.pt re-scored and compared to results.json exactly.\n\n")
        f.write("## Problems (%d)\n\n" % len(problems))
        f.write("\n".join("- " + p for p in problems) if problems
                else "- none: every control reproduced results.json exactly and every rule "
                     "checkpoint loaded.\n")
        f.write("\n\n## Per-cell aggregate (mean / fold spread of test BA)\n\n")
        f.write("| group | method | source | target | n | " +
                " | ".join("%s mean | %s spread" % (r, r) for r in RULES) + " |\n")
        f.write("|" + "---|" * (5 + 2 * len(RULES)) + "\n")
        for c in crows:
            f.write("| %s | %s | %s | %s | %d | " % (c["group"], c["method"], c["source"],
                                                      c["target"], c["n_folds"]))
            f.write(" | ".join(
                ("%.4f | %s" % (c["%s_mean" % r],
                                "-" if c["%s_spread" % r] is None else "%.4f" % c["%s_spread" % r]))
                if c["%s_mean" % r] is not None else "- | -" for r in RULES))
            f.write(" |\n")

    print("rows %d, cells %d, problems %d" % (len(rows), len(crows), len(problems)))
    for p in problems:
        print("  !! " + p)
    print("-> analysis/reeval_rules_table.csv, reeval_cells.csv, reeval_report.md")
    sys.exit(0 if not problems else 5)


if __name__ == "__main__":
    main()

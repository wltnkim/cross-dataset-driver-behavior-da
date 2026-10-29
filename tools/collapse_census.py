#!/usr/bin/env python3
"""[ARR 2026-09-05] TASK 2 -- collapse census, computed on ISAAC's own per-epoch curves.

Reads only results/*/metrics.jsonl (written by the trainer's --instrument block) plus
config.yaml and MATRIX_APPROVED.csv for the run's group/method/source labels. It does NOT
read, quote or check any Swan summary: the point of the task is an independent reproduction.

Definitions, stated once so two readers cannot apply them differently:

  chance_target   1/(number of unified classes actually PRESENT in that target's val split).
                  MEASURED from the memmap metadata, not assumed: it is 0.2000 for all six
                  (target, fold) combinations here, because unified class 5 (seatbelt) has
                  zero clips in DMD and 3MDAD.
  peak_epoch      argmax over epochs 1..N of target_val_bal_acc; ties resolved to the
                  EARLIEST epoch.
  last10_mean     mean of target_val_bal_acc over the final 10 recorded epochs.
  collapse_epoch  the smallest e such that target_val_bal_acc < 1.25 x chance for EVERY
                  epoch >= e ("first drops below and STAYS"). None if the final epoch is
                  at or above the threshold -- i.e. it never entered a terminal collapse.
  source-val bucket, from the last-10 mean of source_val_bal_acc:
                  RETAINED     >= 0.50   (far above either plausible chance: 1/6 or 1/5)
                  COLLAPSED    <  0.25   (= 1.25 x 0.20)
                  INTERMEDIATE otherwise. The middle bucket exists on purpose: forcing a
                  binary would hide any run that sits between the two regimes.
  adversarial     alpha_dann > 0. source_only runs are DANN with alpha_dann == 0.0, so the
                  trainer mode alone does NOT separate them; alpha does.

Writes analysis/collapse_census.json and analysis/collapse_census.csv.
"""
import csv, json, os, statistics, collections

ARR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
LAST_N = 10
COLLAPSE_MULT = 1.25
SRC_RETAINED, SRC_COLLAPSED = 0.50, 0.25


def target_chance(memmap_metadata, fold):
    meta = json.load(open(memmap_metadata))
    labs = {m["unified_label"] for m in meta
            if m.get("fold_assignment", {}).get(str(fold)) == "val"}
    return 1.0 / len(labs), len(labs)


def main():
    inv = [r for r in json.load(open(os.path.join(ARR, "analysis", "reeval_inventory.json")))
           if r["status"] == "OK"]
    appr = {r["run_id"].replace("-", ""): r
            for r in csv.DictReader(open(os.path.join(ARR, "MATRIX_APPROVED.csv")))}
    chance_cache = {}
    rows = []

    for r in inv:
        a = appr[r["run_tag"]]
        d = os.path.join(ARR, "results", r["run_dir"])
        recs = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl")) if l.strip()]
        recs.sort(key=lambda x: x["epoch"])
        tv = [x["target_val_bal_acc"] for x in recs]
        sv = [x["source_val_bal_acc"] for x in recs]
        cls = [x["train_loss_cls"] for x in recs]
        tot = [x["train_loss_total"] for x in recs]
        dom = [x["train_loss_domain"] for x in recs]
        ent = [x["target_mean_entropy"] for x in recs]

        ck = (a["target"], a["fold"])
        if ck not in chance_cache:
            chance_cache[ck] = target_chance(r["memmap_metadata"], a["fold"])
        chance, n_cls = chance_cache[ck]
        thr = COLLAPSE_MULT * chance

        peak_i = max(range(len(tv)), key=lambda i: (tv[i], -i))
        # first epoch from which it is below threshold for the rest of the run
        collapse_ep = None
        for i in range(len(tv)):
            if all(v < thr for v in tv[i:]):
                collapse_ep = recs[i]["epoch"]
                break

        last_tv = tv[-LAST_N:]
        last_sv = [v for v in sv[-LAST_N:] if v is not None]
        sv_last = statistics.mean(last_sv) if last_sv else None
        bucket = ("NO_SOURCE_VAL" if sv_last is None else
                  "RETAINED" if sv_last >= SRC_RETAINED else
                  "COLLAPSED" if sv_last < SRC_COLLAPSED else "INTERMEDIATE")

        adversarial = float(a["mode"] != "source_only" and float(r["alpha_dann"]) > 0)
        rows.append(dict(
            run_id=a["run_id"], group=a["group"], method=a["mode"],
            adversarial=bool(adversarial), alpha_dann=r["alpha_dann"],
            source=a["source"], target=a["target"], fold=int(a["fold"]),
            n_epochs=len(recs), n_val_classes=n_cls, chance=chance, collapse_threshold=thr,
            peak_epoch=recs[peak_i]["epoch"], peak_value=tv[peak_i],
            last10_mean=statistics.mean(last_tv),
            last10_min=min(last_tv), last10_max=max(last_tv),
            collapse_epoch=collapse_ep,
            never_collapsed=(collapse_ep is None),
            final_tv=tv[-1],
            src_val_last10=sv_last, src_bucket=bucket,
            src_val_peak=(max(v for v in sv if v is not None) if last_sv else None),
            train_loss_cls_last10=statistics.mean(cls[-LAST_N:]),
            train_loss_total_last10=statistics.mean(tot[-LAST_N:]),
            train_loss_total_max=max(tot),
            train_loss_domain_last10=(statistics.mean([v for v in dom[-LAST_N:] if v is not None])
                                      if any(v is not None for v in dom[-LAST_N:]) else None),
            entropy_last10=statistics.mean(ent[-LAST_N:]),
            entropy_ceiling=recs[-1]["entropy_ceiling_ln_C"],
            peak_is_epoch1=(recs[peak_i]["epoch"] == 1),
            peak_in_first3=(recs[peak_i]["epoch"] <= 3),
        ))

    rows.sort(key=lambda x: (x["group"], x["method"], x["source"], x["target"], x["fold"]))
    json.dump(rows, open(os.path.join(ARR, "analysis", "collapse_census.json"), "w"), indent=1)
    with open(os.path.join(ARR, "analysis", "collapse_census.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote analysis/collapse_census.{json,csv}  (%d runs)" % len(rows))
    return rows


if __name__ == "__main__":
    main()

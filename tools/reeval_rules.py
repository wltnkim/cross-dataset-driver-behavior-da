#!/usr/bin/env python3
"""[ARR 2026-09-05] Re-evaluate every completed run's test split under each of the four
selection rules retained by tools/prune_run.py.

USES THE TRAINER'S OWN INTERNAL EVAL PATH, not eval_only():
  * datasets are built with the same classes and the same arguments as train() builds
    dna_test (memmap256 pipeline, augment=False, crop 224);
  * the model is built with train_cross.build_model(), the same call train() makes;
  * the scoring loop is character-for-character the [Test] block of train_cross.train().
eval_only() is NOT imported: it rebuilds the model from --backbone-blind build_model and
constructs its test set through decord, a different input pipeline (analysis/K_eval_path.md).

CONTROL (mandatory, never skipped): for every run we additionally score best_model.pt --
the exact artefact results.json was written from -- and compare to results.json's
test_balanced_accuracy. A mismatch is recorded as CONTROL_MISMATCH and the run is marked
so downstream. The oracle-rule checkpoint epoch_<oracle>.pt must give the same number:
that is a second, independent leg of the same control.

Read-only w.r.t. results/: opens no file there for writing.

BUDGET: --wall-seconds is the granted wall. The script projects its own cost from the
calibration constant (analysis/c_calibration.json: s per evaluated clip) before starting,
refuses to start if the projection exceeds the grant, and re-checks remaining wall before
every checkpoint. It writes partial output rather than being SIGKILLed mid-write.
"""
import argparse, json, os, sys, time, hashlib

ARR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, os.path.join(ARR, "code"))

RULES = ["oracle_target_val", "source_val", "last_epoch", "min_target_entropy"]
SAFETY = 1.6          # projection multiplier over the pure forward-pass constant
LOAD_S = 25.0         # per-checkpoint state_dict load allowance, s (349 MiB off Lustre)
SETUP_S = 900.0       # container start + HF model build + memmap open allowance, s


def sha256_head(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["dmd", "3mdad"])
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wall-seconds", type=float, required=True)
    ap.add_argument("--limit-runs", type=int, default=0, help="canary: stop after N runs")
    ap.add_argument("--redtest-control", action="store_true",
                    help="RED TEST: perturb the expected results.json values by +1e-6 so the "
                         "control MUST report a mismatch. Proves the control can fail. Never "
                         "use for real numbers -- the output file is a red-test artefact.")
    a = ap.parse_args()

    t_start = time.time()
    inv = [r for r in json.load(open(os.path.join(ARR, "analysis", "reeval_inventory.json")))
           if r["status"] == "OK"
           and r["target_dataset"] == a.target and r["fold"] == a.fold]
    inv.sort(key=lambda r: r["run_dir"])
    if a.limit_runs:
        inv = inv[:a.limit_runs]
    if a.redtest_control:
        for r in inv:
            r["test_ba_reported"] = r["test_ba_reported"] + 1e-6
            r["test_acc_reported"] = r["test_acc_reported"] + 1e-6
        print("[RED TEST] expected values perturbed by +1e-6; every control MUST mismatch.")
    if not inv:
        print("no runs for %s fold %d" % (a.target, a.fold)); sys.exit(2)

    # ---- budget declaration, BEFORE any heavy import ----------------------------------
    calib = json.load(open(os.path.join(ARR, "analysis", "c_calibration.json")))
    s_per_clip = float(calib["sec_per_eval_clip"])   # measured, job 6155473

    n_eval = 0
    for r in inv:
        n_eval += len({v["epoch"] for v in r["rules"].values()}) + 1   # +1 = best_model.pt control
    # test size is read after the dataset is built; use the metadata count now
    meta_p = inv[0]["memmap_metadata"]
    meta = json.load(open(meta_p))
    n_test = sum(1 for m in meta
                 if m.get("fold_assignment", {}).get(str(a.fold)) == "test")
    proj = SETUP_S + n_eval * (LOAD_S + n_test * s_per_clip * SAFETY)
    print("[BUDGET] %s fold %d : %d runs, %d checkpoint evaluations, test split %d clips"
          % (a.target, a.fold, len(inv), n_eval, n_test))
    print("[BUDGET] per-clip constant %.4f s (measured, c_calibration.json) x safety %.1f"
          % (s_per_clip, SAFETY))
    print("[BUDGET] projected %.0f s (%.2f h) against granted %.0f s (%.2f h)"
          % (proj, proj / 3600, a.wall_seconds, a.wall_seconds / 3600))
    if proj > a.wall_seconds:
        print("[BUDGET] REFUSING TO START: projection exceeds the granted wall.")
        sys.exit(9)

    # ---- heavy imports ----------------------------------------------------------------
    import numpy as np, torch, yaml
    from torch.amp import autocast
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from cross_dataset import train_cross as T
    from cross_dataset.memmap_dataset import MemmapVideoDataset

    print("[env] torch %s  cuda %s  device %s"
          % (torch.__version__, torch.version.cuda,
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("REFUSING: no GPU. The frozen numbers were produced on a GPU; a CPU recompute "
              "is not comparable."); sys.exit(10)

    da_modes = ("dann", "cbdann", "cdan", "mcc", "bnm")

    # ---- test dataset: same construction train() uses for dna_test --------------------
    cfg0 = inv[0]
    _mm_crop = cfg0["memmap_crop"] if cfg0["memmap_crop"] and cfg0["memmap_crop"] > 0 else None
    dna_test = MemmapVideoDataset(
        cfg0["memmap_path"], cfg0["memmap_metadata"],
        fold=a.fold, split="test",
        num_frames=cfg0["num_frames"], augment=False,
        crop_size=_mm_crop,
    )
    assert len(dna_test) == n_test, "test split size %d != metadata count %d" % (len(dna_test), n_test)
    num_classes = T.get_num_unified_classes()
    class_names = T.get_unified_class_names()
    print("[data] test clips %d, unified classes %d %s" % (len(dna_test), num_classes, class_names))

    nw = 4
    test_loader = DataLoader(dna_test, batch_size=cfg0["batch_size"], shuffle=False,
                             num_workers=nw, pin_memory=True,
                             persistent_workers=True, prefetch_factor=4)

    models = {}

    def get_model(mode):
        if mode not in models:
            m = T.build_model(num_classes, mode, None, 0.1,
                              model_name="MCG-NJU/videomae-base-finetuned-kinetics",
                              no_pretrain=False).to(device)
            models[mode] = m
        return models[mode]

    def score(model, mode_now, fp16):
        """Character-for-character the [Test] block of train_cross.train()."""
        model.eval()
        test_preds, test_labels_list = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                with autocast("cuda", enabled=fp16):
                    if mode_now in da_modes:
                        logits, _, _ = model(imgs, 0.0)
                    else:
                        logits = model(pixel_values=imgs).logits
                test_preds.extend(logits.argmax(-1).cpu().numpy())
                test_labels_list.extend(labels.numpy())
        return (float(accuracy_score(test_labels_list, test_preds)),
                float(balanced_accuracy_score(test_labels_list, test_preds)),
                len(test_labels_list))

    out = {"target": a.target, "fold": a.fold, "n_test": n_test,
           "RED_TEST_ARTEFACT": bool(a.redtest_control),
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "s_per_clip_constant": s_per_clip, "runs": []}

    def flush():
        json.dump(out, open(a.out, "w"), indent=1)

    for ri, r in enumerate(inv):
        remaining = a.wall_seconds - (time.time() - t_start)
        eps = sorted({v["epoch"] for v in r["rules"].values()})
        need = (len(eps) + 1) * (LOAD_S + n_test * s_per_clip * SAFETY)
        if need > remaining:
            print("[BUDGET] STOPPING before %s: needs %.0f s, %.0f s left. %d/%d runs done."
                  % (r["run_dir"], need, remaining, ri, len(inv)))
            out["stopped_early"] = {"at_run": r["run_dir"], "done": ri, "total": len(inv)}
            break

        d = os.path.join(ARR, "results", r["run_dir"])
        mode_now = r["mode"]
        model = get_model(mode_now)
        rec = {"run_dir": r["run_dir"], "run_tag": r["run_tag"], "mode": mode_now,
               "alpha_dann": r["alpha_dann"], "target": a.target, "fold": a.fold,
               "test_ba_reported": r["test_ba_reported"],
               "test_acc_reported": r["test_acc_reported"],
               "by_epoch": {}, "by_rule": {}, "load_errors": [], "notes": []}

        # ---- CONTROL: best_model.pt, the artefact results.json came from --------------
        try:
            sd = torch.load(os.path.join(d, "best_model.pt"), weights_only=False)
            model.load_state_dict(sd)
            del sd
            t0 = time.time()
            acc, ba, n = score(model, mode_now, r["fp16"])
            rec["control_best_model"] = {"test_acc": acc, "test_ba": ba, "n": n,
                                         "seconds": round(time.time() - t0, 1)}
            match = (ba == r["test_ba_reported"]) and (acc == r["test_acc_reported"])
            rec["control_match_exact"] = bool(match)
            if not match:
                rec["notes"].append(
                    "CONTROL_MISMATCH best_model.pt: recomputed BA %.10f acc %.10f vs "
                    "results.json BA %.10f acc %.10f"
                    % (ba, acc, r["test_ba_reported"], r["test_acc_reported"]))
                print("  !! " + rec["notes"][-1])
        except Exception as e:
            rec["load_errors"].append("best_model.pt: %r" % (e,))
            rec["control_match_exact"] = None
            print("  !! best_model.pt load/score FAILED: %r" % (e,))

        # ---- the four rules ----------------------------------------------------------
        for e in eps:
            f = "epoch_%03d.pt" % e
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                rec["load_errors"].append("%s: MISSING" % f)
                print("  !! %s MISSING" % f)
                continue
            try:
                sd = torch.load(p, weights_only=False)
                model.load_state_dict(sd)
                del sd
            except Exception as ex:
                rec["load_errors"].append("%s: %r" % (f, ex))
                print("  !! %s load FAILED: %r" % (f, ex))
                continue
            t0 = time.time()
            acc, ba, n = score(model, mode_now, r["fp16"])
            rec["by_epoch"][str(e)] = {"file": f, "test_acc": acc, "test_ba": ba, "n": n,
                                       "seconds": round(time.time() - t0, 1)}
            print("  %s e%02d  testBA=%.6f acc=%.6f  (%.0fs)"
                  % (r["run_dir"][-34:], e, ba, acc, time.time() - t0))

        for rule in RULES:
            e = r["rules"][rule]["epoch"]
            cell = rec["by_epoch"].get(str(e))
            rec["by_rule"][rule] = {
                "epoch": e,
                "val_criterion_value": r["rules"][rule]["value"],
                "test_ba": (cell or {}).get("test_ba"),
                "test_acc": (cell or {}).get("test_acc"),
            }
        # second leg of the control: oracle-rule checkpoint vs results.json
        orc = rec["by_rule"]["oracle_target_val"]["test_ba"]
        rec["oracle_ckpt_matches_results_json"] = (
            None if orc is None else bool(orc == r["test_ba_reported"]))
        if orc is not None and orc != r["test_ba_reported"]:
            rec["notes"].append(
                "ORACLE_CKPT_MISMATCH epoch_%03d.pt gives BA %.10f, results.json says %.10f"
                % (rec["by_rule"]["oracle_target_val"]["epoch"], orc, r["test_ba_reported"]))
            print("  !! " + rec["notes"][-1])

        out["runs"].append(rec)
        flush()
        print("[%d/%d] %s done  (elapsed %.0f s)"
              % (ri + 1, len(inv), r["run_dir"], time.time() - t_start))

    out["elapsed_seconds"] = round(time.time() - t_start, 1)
    flush()
    n_ctrl_bad = sum(1 for r in out["runs"] if r.get("control_match_exact") is not True)
    print("\n[SUMMARY] %s fold %d: %d runs scored, %d control mismatches/failures, elapsed %.0f s"
          % (a.target, a.fold, len(out["runs"]), n_ctrl_bad, out["elapsed_seconds"]))
    print("-> %s" % a.out)
    # A control failure is a REPORTABLE condition, not a crash: the file is written either way.
    sys.exit(0 if n_ctrl_bad == 0 else 7)


if __name__ == "__main__":
    main()

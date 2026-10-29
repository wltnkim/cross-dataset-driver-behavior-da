# Protocol change: fixed 50-epoch schedule for every matrix run

**Decided 2026-08-31, before any matrix run was submitted.**

## What the frozen trainer actually did — and why it matters

`train_cross.py` (frozen) drives early stopping from the **selection metric**, and that metric
defaults to **target-validation balanced accuracy**:

```
:750   sel_bal_acc = val_bal_acc                      # val_loader = TARGET val
:751   use_src_sel = getattr(args, "select_by_source_val", False) and args.mode in da_modes
:768   if sel_bal_acc > best_val_bal_acc: ... patience_counter = 0
:774   else: patience_counter += 1
:787   if patience_counter >= args.patience: break
```

So in every submitted run except the one `srconlysv` build, **the number of epochs a run trained
was itself a function of target labels.** This is an oracle dependency *beyond* checkpoint
selection, and it is not disclosed anywhere in the submitted paper:

- The paper discloses that **which checkpoint** is chosen uses target-val (oracle selection).
- It does **not** disclose that **how long the run trained** also used target-val.

A genuinely label-free deployment could not have known when to stop. Any "fixed schedule"
baseline computed from the submitted runs would therefore be contaminated — the trajectory was
truncated by an oracle.

**Measured consequence, from the calibration (job 6155473):** target-val peaked at epoch 1, so
`patience=15` stopped the run at **epoch 16 of 50**. Epochs 17–50 never existed. A last-epoch or
fixed-schedule criterion computed on that run would be reporting epoch 16, chosen by the oracle.

## The change

Every matrix run uses **`--patience 999`**, giving a **fixed 50-epoch schedule**. Every selection
criterion then sees the same full trajectory, and `metrics.jsonl` records all 50 epochs.

**No trainer edit was required**, so `DIFF.md` is unchanged and the label-discipline harness did
not need re-running. Verified by simulating the frozen loop: with `epochs=50`, the maximum
reachable `patience_counter` is 49 (best at epoch 1), so `>= 999` can never fire. The same
simulation reproduces the calibration's observed 16 epochs exactly from `patience=15`, which is
what validates the model of the loop.

| `--patience` | epochs run (best at epoch 1) | early stop |
|---|---:|---|
| 15 (frozen default) | 16 | yes — **matches the calibration exactly** |
| 49 | 50 | yes (on the last epoch, harmless) |
| 50 | 50 | no |
| **999 (chosen)** | **50** | **no** |

`best_model.pt` still tracks target-val, so the oracle criterion remains recorded — it is now one
criterion among four rather than also controlling run length.

## Disclosure consequences

Two documents must gain a line when the results land. **Both are deferred to the results round**,
because the sentence that states the size of the effect needs the numbers:

1. **`analysis/R2-1_protocol_table.tex`** — the model-selection footnote currently says
   "target-validation balanced accuracy for all trained methods (oracle protocol)". It must also
   say that **early stopping used the same target-val signal**, so the oracle governed both the
   checkpoint and the schedule.
2. **R2-4 text in the manuscript** — the `%% TODO-RESULTS` marker at the §V-C oracle paragraph
   should note that the submitted runs' epoch counts were themselves oracle-determined, and that
   the revision's reruns use a fixed schedule so the label-free criteria are computed on an
   uncontaminated trajectory.

This file is the record that the change was made **before** any matrix result existed.

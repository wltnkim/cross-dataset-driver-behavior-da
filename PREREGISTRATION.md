# Preregistration — E1 source-breadth experiment

**Written 2026-08-29 21:24 UTC. No E1 training result exists at this timestamp**: the only GPU
job submitted is the calibration (6155473, `PENDING`, a different source/target pairing used
solely to measure cost), and no E1 run has been created, queued, or scored.

**This document is frozen once E1 runs begin.** No hypothesis, ordering, statistic, arm
designation or threshold in it may be edited after the first E1 result exists. If something here
turns out to be badly specified, the correction is an **amendment appended below with its own
timestamp**, never an edit in place.

---

## 1. What is being manipulated

Four labelled source sets, **all at n = 573** (matched to OBC's training split), each used as the
DANN source against unlabelled targets DMD and 3MDAD, three folds each.

| arm | construction | role |
|---|---|---|
| `OBC` | the paper's existing source | reference |
| `da573_random` | uniform random draw from Drive&Act fold-0 train, fixed seed | **unbiased broad source at OBC size** |
| `broad573` | farthest-point sampling in embedding space, per class | mechanism pair, wide end |
| `narrow573` | tightest neighbourhood of each class centroid | mechanism pair, tight end |

Manifests are frozen and will not be regenerated:

```
b3f22b31025d58cb...  broad573.csv
83c0f39d0d4dfe18...  narrow573.csv
1f7247ab5f92f3e4...  da573_random.csv
```

`broad573` and `narrow573` carry an **identical class distribution** by construction.
`da573_random` does **not** — it is an unbiased draw and follows Drive&Act's own proportions up to
sampling noise. That asymmetry is deliberate (an unbiased arm must not be re-weighted) and is
disclosed here rather than discovered later.

Everything else is held fixed: backbone, optimizer, schedule, batch size, epochs, target splits,
label space, and the pre-decoded pipeline.

## 2. What the A2 containment result exposes

A2 measured, before this document, how well each candidate source covers each target in
Kinetics-400 embedding space (`analysis/a2_containment.json`, frozen). Two accounts survive that
table, and E1 is designed to separate them. **No predicted values are recorded here — only
orderings.** Numeric predictions are exactly what would be quietly moved later.

**H1 — containment-dominated.** What matters is whether the source covers the target's region of
feature space, not how internally broad the source is. *Prediction:* on DMD and on 3MDAD, all
three Drive&Act arms land together — `da573_random` ≈ `broad573` ≈ `narrow573` — and all three
sit above `OBC`.

**H2 — self-breadth contributes.** Internal spread of the source adds something beyond
containment. *Prediction:* `broad573` > `narrow573`, by more than the across-fold spread of the
arms themselves.

H1 and H2 are not mutually exclusive: H1 concerns the gap to OBC, H2 concerns the gap within the
Drive&Act arms. Both are pre-registered, and both are judged separately.

## 3. Committed primary comparison (R1-1)

> **A Drive&Act-573 source versus OBC, at matched n = 573.**
> Designated instance: **`da573_random` vs `OBC`**.
> Statistic: three-fold mean target-**test** balanced accuracy, reported separately for DMD and
> for 3MDAD. Direction of interest: Drive&Act-573 > OBC.
> Model selection: target-validation balanced accuracy — the same oracle protocol the submitted
> paper already uses and discloses, chosen so E1 is comparable with the published table rather
> than better than it.

`da573_random` is the designated instance **because it is the only unbiased one**. `broad573` and
`narrow573` are constructed sets; using whichever of the three scores highest would be selection
on the benchmark, which this project has an explicit rule against. Therefore:

- **The primary comparison may not be re-pointed at another arm after results exist.**
- A "best of the three Drive&Act arms" reading is **not** pre-registered and must not be reported
  as the primary result. If a max-over-arms statistic is wanted, it needs its own null (the
  expected maximum of three draws is not the expected value of one) and must be registered
  *before* results, in an amendment.
- `broad573` and `narrow573` are reported alongside as the mechanism pair, secondary.

Balanced accuracy averages over the classes **present in ground truth**; DMD and 3MDAD each have
five (no seatbelt), so a single-class collapse scores 1/5 on those targets, not 1/6. This is a
property of the metric as implemented and is stated here so it is not mistaken for signal.

## 4. Outcome grid — what each ordering implies for the paper's claim

The paper attributes the UDA failure to the *breadth* of the labelled source distribution. One
sentence per ordering, committed in advance.

| # | outcome | implication for the paper's claim |
|---|---|---|
| 1 | All three D&A arms above OBC; `broad573` ≈ `narrow573` ≈ `da573_random` | The mechanism is source–target **containment**, not source breadth, and §5's attribution must be recast in those terms. |
| 2 | All three above OBC **and** `broad573` > `narrow573` | Containment sets the level and self-breadth modulates it; the claim survives as a two-factor account rather than the single-factor one now written. |
| 3 | All three D&A arms indistinguishable from OBC | Containment does not explain the failure, the breadth account loses its main support, and the cause must be sought elsewhere (label space, optimisation, or the target side). |
| 4 | `narrow573` > `broad573` | The breadth mechanism runs **opposite** to the stated direction and the §5 attribution must be withdrawn, not softened. |
| 5 | `da573_random` above both constructed arms | The FPS / tightest-cluster construction is itself harmful, so the mechanism pair is not a clean manipulation and must be rebuilt before any breadth claim is made. |
| 6 | The ordering holds on one target but not the other | The effect is target-specific and no general claim about source breadth is supportable from this benchmark. |
| 7 | Any arm collapses to a single class (BA at the 1/5 floor) | That arm is uninformative about breadth and is reported as a collapse, not entered into the ordering. |

Outcome 3 and outcome 4 are both adverse to the paper as currently written. They are listed with
the same weight as the favourable ones **on purpose**.

## 5. Analysis rules fixed in advance

1. **Folds.** Three folds per (source, target). The source manifest is a fold-0 draw and is held
   **fixed** across folds; the fold index rotates the target split only. Reported as a three-fold
   mean with the across-fold spread shown.
2. **No arm may be added, dropped, re-drawn or re-seeded** after results exist. The manifest
   hashes above are the record.
3. **No re-running an arm that produced an unwelcome number.** A re-run is permitted only for a
   named infrastructure failure (OOM, node failure, timeout), must reuse the identical manifest
   and seed, and must be recorded with both job ids.
4. **"Better" means the primary statistic in §3**, not any secondary metric that happens to
   separate the arms.
5. **Ordering claims require the gap to exceed the across-fold spread.** An ordering inside the
   fold spread is reported as "not separated", not as a direction.
6. Every arm is scored with the same code, the same selection criterion and the same evaluation
   class count per target.

## 6. Deliberately not pre-registered

The backbone ladder (Part H) has no GPU timing yet and no committed hypothesis; it is
implementation only this round. The reverse-direction 0-label group (E2) is blocked by an
assertion in the trainer and is not registered until it can actually run.

---

*Amendments must be appended below this line with their own UTC timestamp and a statement of what
was known at the time. Nothing above this line changes.*

## Amendment 1 — 2026-08-29 (UTC timestamp below), no change to anything above the line

**Status: the preregistration stands exactly as registered. This amendment adds nothing to it and
removes nothing from it.**

What was known at the time of writing: the calibration job (6155473) was still `PENDING`, no E1
run had been created or queued, and no E1 result existed.

The author considered the alternative wording "any DA-573 arm versus OBC" for the R1-1 primary
comparison and has **withdrawn it**. The registered primary therefore remains, unchanged:

> `da573_random` vs `OBC`, three-fold mean target-test balanced accuracy, per target.

Specifically:

1. **No max-statistic amendment is being made.** A best-of-three-arms reading remains
   un-registered and must not be reported as the primary result.
2. `broad573` and `narrow573` remain the **mechanism pair**, secondary, exactly as in §3.
3. §4's outcome grid and §5's analysis rules are untouched.

This amendment exists so that the withdrawal is on the record rather than inferred from silence.
Recorded: 2026-08-29T21:47:49Z

## Amendment 2 — E2 protocol note (0-label reverse direction)

**Appended, not edited. Nothing above the Amendment 1 line changes.** Written before any E2 run
exists: the E2 code path was unblocked and CPU-verified on the date below, and no E2 job has been
created, queued or scored.

E2 is the reverse direction with **zero target labels**: a full target-dataset train split is the
labelled source, OBC is the unlabelled target. It is registered here as mirroring the forward
protocol, so that E2 and the forward tables are comparable by construction rather than by
argument.

1. **Instrumentation.** E2 runs carry the full B2 instrumentation — per-epoch model checkpoints
   and a `metrics.jsonl` line per epoch with separated classification and domain loss,
   source-validation balanced accuracy, target-validation balanced accuracy, and mean target
   prediction entropy. Identical to every other group.

2. **Primary comparison and selection rule.** The primary E2 statistic uses **the same selection
   rule as the forward tables**: checkpoint chosen by target-validation balanced accuracy (the
   oracle protocol the submitted paper already uses and discloses), three-fold mean, reported on
   the target **test** split. Any deviation from that rule would make E2 incomparable with the
   forward numbers it exists to contrast.

3. **Non-oracle criteria are secondary and free.** Source-validation selection, last-epoch, and
   minimum-target-entropy selection are all recoverable from `metrics.jsonl` without re-running.
   They are **reported as secondary**, and none of them replaces the primary rule in (2). This is
   pre-registered so that a non-oracle criterion cannot later be promoted because it happens to
   look better.

4. **Selection and reporting use different splits.** E2 selects on OBC **val** (176 clips) and
   reports on OBC **test** (174 clips). This is a deliberate departure from the frozen
   `--reverse_direction` block, which set `dna_test = obc_target_val` and therefore selected and
   reported on the same split. The frozen behaviour is left untouched in the original tree; the
   split separation applies to E2 only, and is recorded here because it changes what an E2 number
   means relative to the published Table 10.

5. **Label discipline.** In the 0-label path the target-train labels are structurally replaced by
   an out-of-range sentinel before any loader wraps them, so they cannot reach a training loss or
   an adaptation objective. Target labels are read in exactly two places, both reporting-only: the
   evaluation block and the metrics logger. This is verified by a red test with a positive
   control, not asserted.

No numbers are recorded here, and no outcome grid is added: E2 tests whether the reverse-direction
advantage survives the removal of target labels, and the orderings in §4 already cover what each
result would mean for the claim.
Recorded: 2026-08-29T22:10:06Z

## Amendment 3 — E2 replicate structure (seeds, not folds), no change to anything above the line

Recorded: 2026-09-05T09:30:36Z

**What changed and why.** E2's nine runs are three sources x three runs. Reading the frozen
trainer line by line established that under the approved E2 route (`--target_dataset obc` with
`--source_manifest`, no `--reverse_direction`) **nothing in an E2 run depends on `--fold`**:

- `OBCCrossDataset` takes no fold argument. It splits 42 subjects 26/8/8 at its own fixed
  internal `seed=42` (`code/cross_dataset/obc_dataset.py:62, 95-108`), giving OBC
  train/val/test = 573/176/174 for every value of `--fold`.
- `build_manifest_source` hardcodes `fold=0` (`train_cross.py:167`), so the labelled source
  is the frozen manifest regardless of `--fold`.
- the source-validation loader hardcodes `fold=0` (`train_cross.py:662`).
- `--fold` therefore reaches only the result-directory name and the `fold` field written into
  `results.json`.

Left as it was, the three runs per source would have been **the same run three times**, separated
only by the GPU nondeterminism `set_seed()` deliberately does not suppress
(`train_cross.py:447-448`), and their spread would have been reported as fold variance when it
is nothing of the kind.

**What is now registered.**

1. The three E2 runs per source are **seed replicates**, not folds. The generator emits
   `--seed 42`, `--seed 43`, `--seed 44` for `--fold` 0/1/2 respectively
   (`tools/gen_run_sbatch.py`). Seed 42 is the trainer default, so the fold-0 run keeps exactly
   the behaviour every other group in the matrix has.

2. They run on **OBC's single fixed 26/8/8 subject-disjoint split**. There is no 3-fold split of
   OBC anywhere in this codebase, and E2 does not create one.

3. Their spread therefore estimates **run-to-run nondeterminism at a fixed data split**, not
   data-split variance. It is not a cross-validation standard deviation and must not be compared
   with one, in either direction, without saying so.

4. **E2 results are reported as "mean (std) over 3 seeds."** Not "3-fold mean (std)".
   This supersedes the phrase *"three-fold mean"* in Amendment 2 §2 for E2 and for E2 only;
   everything else in Amendment 2 — the selection rule, the val/test split separation, the
   secondary criteria, the label discipline — stands unchanged.

5. Nothing here changes the forward groups. E1, E4, R2-4 and E3 target datasets do have real
   folds and continue to be reported as 3-fold means.

**Not registered, deliberately:** no claim that three seeds are enough to characterise the noise,
and no threshold on the spread. The identical-configuration noise floor measured elsewhere in this
project is large relative to the effects at stake; three draws bound it loosely and are reported
as such.

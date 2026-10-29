# Preregistration adjudication — 2026-09-07

Written 2026-09-07T19:06:13Z, after the 107-run Swan campaign completed and its selection-rule
re-evaluation, collapse census, cell aggregation, cross-site comparison, and adaptation-gain
tables were produced (`swan/analysis/CAMPAIGN_REPORT.md` and its CSVs). The user adjudicated the
preregistration from those numbers; this document verifies that arithmetic against the named
CSVs, records the verdicts, and states plainly where a number did not check out (§2 below).

**This file is append-only from the moment it is written**, the same discipline
`PREREGISTRATION.md` itself follows: a correction is a dated section appended below, never an
edit in place. Its sha256 is recorded in `analysis/PREREG_FREEZE.sha256` alongside
`PREREGISTRATION.md`'s own hash chain (§9 below).

**Source of record.** Per `STATE.md`'s standing rule, Swan's `results/` is the paper's source and
ISAAC's `results/` is the independent replication set. Every verdict below is adjudicated on the
**Swan campaign** (107 runs, `swan/analysis/reeval_cells.csv`). ISAAC's 65-run set is cited only
as corroboration, never as the basis for a verdict — stated explicitly in §2 because that section
turned up a place where the two sites disagree.

All test-BA figures below are the **oracle rule** (`oracle_target_val`) unless a rule is named.
Noise floor = **0.077445**, the mean pairwise |Δ| among the five identical-seed Swan parity
repeats under the oracle rule on test data (`swan/NOISE_FLOOR.md`, "Test-side confirmation"
table, oracle row). Chance = 0.2 for 3MDAD/DMD targets (five classes with nonzero test support,
`PREREGISTRATION.md` §3) and 1/6 = 0.166667 for the OBC target (E2).

---

## 1. PRIMARY (R1-1): `da573_random` vs `OBC`, oracle rule, reported test BA

Cites `PREREGISTRATION.md` §3, "Committed primary comparison (R1-1)": *"A Drive&Act-573 source
versus OBC, at matched n = 573. Designated instance: `da573_random` vs `OBC`."*

| target | method | da573_random (spread) | OBC (spread) | diff | exceeds noise floor (0.077445)? |
|---|---|---|---|---|---|
| 3MDAD | source_only | 0.518137 (0.163235) | 0.307353 (0.104412) | **+0.210784** | yes |
| 3MDAD | dann | 0.386887 (0.151103) | 0.309926 (0.023529) | **+0.076961** | **no** — 0.000484 below the floor |
| 3MDAD | cdan | 0.213971 (0.017647) | 0.200000 (0.000000) | **+0.013971** | no |
| DMD | source_only | 0.547955 (0.118363) | 0.324151 (0.070981) | **+0.223805** | yes |
| DMD | dann | 0.498733 (0.057010) | 0.304979 (0.056742) | **+0.193754** | yes |
| DMD | cdan | 0.500613 (0.140770) | 0.280122 (0.027877) | **+0.220491** | yes |

Direction of interest (Drive&Act-573 > OBC, §3) holds in all six comparisons. Four of six clear
the noise floor outright (DMD under all three methods, 3MDAD under source_only). 3MDAD/dann sits
0.000484 below the floor — inside the region NOISE_FLOOR.md calls unresolvable by re-running, not
a clear separation. 3MDAD/cdan does not approach the floor at all.

**Note on the user's original figures, corrected per instruction:** the 3MDAD source_only diff
rounds to **0.2108**, not 0.2107 as first written, and the DMD dann diff rounds to **0.1938**,
not 0.1937. Both are ~0.0001 rounding-convention differences (round-half-up vs. truncation) and
change no exceeds/does-not-exceed call above.

**Outcome-grid determination.** `PREREGISTRATION.md` §4's rows 1, 2, 3 and 5 are all stated in
terms of *"all three D&A arms"* (`da573_random`, `broad573`, `narrow573`) against OBC — none of
them is decidable from `da573_random` alone. This section therefore establishes only that the
R1-1 primary comparison's direction holds, with the target-asymmetric strength shown above; the
outcome-grid row that actually fires for §4 is determined in §2, from the arms whose ordering the
grid rows are written about.

---

## 2. MECHANISM PAIR: `broad573` vs `narrow573`, oracle rule, both targets, all three methods

| target | method | broad573 (spread) | narrow573 (spread) | broad − narrow | \|gap\| exceeds noise floor? |
|---|---|---|---|---|---|
| 3MDAD | cdan | 0.426348 (0.043750) | 0.264216 (0.035294) | **+0.162132** | yes |
| 3MDAD | dann | 0.381250 (0.062132) | 0.301471 (0.048529) | **+0.079779** | yes (barely — 0.002334 over) |
| 3MDAD | source_only | 0.515931 (0.023529) | 0.409191 (0.129412) | **+0.106740** | yes |
| DMD | cdan | 0.485589 (0.081036) | 0.539819 (0.090499) | **−0.054230** | no |
| DMD | dann | 0.514040 (0.146356) | 0.527326 (0.078009) | **−0.013286** | no |
| DMD | source_only | 0.551457 (0.061236) | 0.575833 (0.093745) | **−0.024375** | no |

**Sign reverses between targets for all three methods on Swan**: `broad573 > narrow573` on 3MDAD,
`narrow573 > broad573` on DMD, in every method. On 3MDAD all three gaps clear the noise floor
(dann only barely, by 0.002334). On DMD none of the three gaps clear it — the reversed ordering
there sits entirely inside execution noise.

This is `PREREGISTRATION.md` §4 row 6: *"The ordering holds on one target but not the other |
The effect is target-specific and no general claim about source breadth is supportable from this
benchmark."* **Row 6 fires.** Per §4's closing note, this is recorded with the same weight as a
favourable outcome, not softened: **the breadth attribution (paper §5) is withdrawn as a general
claim**, per the row's own wording — withdrawn, not recast or qualified.

### ISAAC cross-check — per method, disagreement stated, not hidden

The user's original sentence claimed ISAAC's independent 65-run set showed "the same reversal."
That checked out for one of three methods and not for another; corrected here rather than written
as first stated.

| method | Swan gap, DMD (broad − narrow) | ISAAC gap, DMD (broad − narrow) | agree? |
|---|---|---|---|
| dann | −0.013286 | −0.036685 | yes — both reverse (narrow > broad) |
| cdan | −0.054230 | **+0.020743** | **no** — Swan reverses, ISAAC does not (broad > narrow on both ISAAC targets) |
| source_only | −0.024375 | **no ISAAC data** | unverifiable |

- **DANN**: the reversal is confirmed on ISAAC (narrow573 > broad573 on DMD there too).
- **CDAN**: ISAAC shows `broad573 > narrow573` on **both** targets (3MDAD +0.163113, DMD
  +0.020743) — no sign change at all. This disagrees with Swan's CDAN result on DMD.
- **source_only**: ISAAC has no data to check. `source_only`/`narrow573` on 3MDAD and DMD are 2 of
  the 7 rows ISAAC deferred and never ran (`STATE.md`, `SITE_ASSIGNMENT.csv` history) — stated
  explicitly rather than left as a silent gap.

**Recorded as measured, not argued:** the broad-vs-narrow contrast is not stable across targets on
either site, and for CDAN it is not stable across sites either, while every gap involved sits at
or below the measured noise floor (CDAN/DMD: Swan −0.054230 and ISAAC +0.020743 are both smaller
in magnitude than 0.077445). **The §4 row-6 verdict stands unchanged and is adjudicated on the
Swan campaign, the paper's source of record; the ISAAC comparison above is corroboration, and
here it corroborates only partially — that partial corroboration is itself part of the record,
not evidence against row 6, since row 6's own text is about target-specificity within a single
campaign, which Swan alone already establishes.**

---

## 3. ADAPTATION GAIN (`swan/analysis/adaptation_gain_swan.csv`, 20 cells × 4 rules)

| rule | negative | positive | zero |
|---|---|---|---|
| oracle_target_val | 18 | 2 | 0 |
| source_val | 20 | 0 | 0 |
| last_epoch | 20 | 0 | 0 |
| min_target_entropy | 18 | 0 | 2 |

**Overall: 80 (cell × rule) values, 76 negative, 2 positive, 2 zero.**

The two positive `oracle_target_val` cells:

| group | method | source → target | gain | exceeds noise floor (0.077445)? |
|---|---|---|---|---|
| E1 | dann | obc → 3mdad | +0.002574 | no |
| E4 | dann | obc → 3mdad | +0.023775 | no |

Both positive cells are the same (method, source, target) triple under two different label
spaces (E1 unified6, E4 nosb5); neither clears the noise floor.

---

## 4. SELECTION-RULE FINDING

`swan/analysis/reeval_cells.csv`, adversarial cells (dann+cdan, n=27) vs source_only cells
(n=10), label-free rules against each cell's target chance (0.2 for 3MDAD/DMD/driveandact,
0.166667 for OBC):

| population | rule | mean | mean − chance | n above chance |
|---|---|---|---|---|
| adversarial (n=27) | source_val | 0.191184 | −0.005113 | 8/27 |
| adversarial (n=27) | last_epoch | 0.191755 | −0.004541 | 9/27 |
| adversarial (n=27) | min_target_entropy | 0.194177 | −0.002119 | 6/27 |
| source_only (n=10) | source_val | 0.367714 | +0.167714 | 10/10 |
| source_only (n=10) | last_epoch | 0.326982 | +0.126982 | 10/10 |
| source_only (n=10) | min_target_entropy | 0.293366 | +0.093366 | 9/10 |

**Description, no interpretation beyond the numbers:** every adversarial cell's mean under every
label-free rule sits at or below its target's chance value; every source_only cell's mean under
every label-free rule sits above its target's chance value, by a margin exceeding the noise floor
(0.077445) in all three rules (source_val +0.167714, last_epoch +0.126982, min_target_entropy
+0.093366).

---

## 5. E2 (R2-2)

`swan/analysis/reeval_cells.csv`, group E2, all four rules, against OBC chance 1/6 = 0.166667:

| source → OBC | oracle_target_val | source_val | last_epoch | min_target_entropy |
|---|---|---|---|---|
| 3mdad_full153 | 0.190082 (0.023771) | 0.170139 (0.010417) | 0.168403 (0.015625) | 0.177083 (0.031250) |
| dmd_full2659 | 0.227742 (0.016827) | 0.166667 (0.000000) | 0.165954 (0.002137) | 0.166667 (0.000000) |
| driveandact_full2368 | 0.215812 (0.088675) | 0.167379 (0.002137) | 0.166667 (0.000000) | 0.166667 (0.000000) |

All three E2 cells' oracle test BA sit between 0.190 and 0.228 — near the 0.166667 chance floor,
not the 0.897/0.898/0.896 the submitted paper reports in Table 10 for the DANN reverse-direction
result (`paper_rev/main.tex:509`, caption confirms `target_label_ratio = 1.0`).

**This closes `EDIT_SITES.md` §8.1 (`paper_rev/main.tex:693` = `paper_overleaf/main.tex:772`)
with evidence, not argument.** §8.1's own text already established, from `config.yaml` and the
trainer's code path, that the submitted reverse-direction result used a fully-labelled
(`target_label_ratio = 1.0`) OBC target, not the zero-label target the Conclusion's sentence
claims. E2 is the label-matched (0-label) run that sentence describes, and its measured result —
0.19–0.23 balanced accuracy, near chance — is the direct empirical answer to what happens when
the label-matched version is actually run: the 0.897–0.898 result does not survive removing the
target labels.

---

## 6. E4 (R1-2): native 5-class (`nosb5`) vs E1's unified6 (33→6 mapped) equivalent

`swan/analysis/reeval_cells.csv`, group E4 (`obc` source, `nosb5` label space) vs group E1
(`obc` source, `unified6` label space), oracle rule:

| target | method | E4 (nosb5, native 5-class) | E1 (unified6) | equal? |
|---|---|---|---|---|
| 3MDAD | cdan | **0.200000** | **0.200000** | **yes, exact** |
| 3MDAD | dann | 0.342402 | 0.309926 | no |
| 3MDAD | source_only | 0.318627 | 0.307353 | no |
| DMD | cdan | 0.298236 | 0.280122 | no |
| DMD | dann | 0.304470 | 0.304979 | no |
| DMD | source_only | 0.336710 | 0.324151 | no |

**cdan → 3MDAD: the 0.200 value is reproduced exactly under a native 5-class label space**
(E4's `nosb5`, which drops the seatbelt class from the model's own output layer) and under
E1's mapped 6-class space (which never has a seatbelt example in 3MDAD ground truth, so also
scores against 5 effective classes). The two routes to a 5-way classification problem — dropping
the class from the head versus never seeing it in the data — give the identical chance-level
result for cdan on this cell. The other five (target, method) pairs are close but not identical
between E4 and E1, as expected since the training runs differ.

---

## 7. CROSS-SITE (replication evidence)

`swan/analysis/crosssite_summary_swan.csv`, 65 run_ids common to both sites:

| rule | n | mean diff | mean \|diff\| | exceeds noise floor (0.077445)? |
|---|---|---|---|---|
| reported (oracle_target_val) | 65 | −0.001450 | 0.034265 | no |
| oracle_target_val | 65 | −0.001450 | 0.034265 | no |
| source_val | 65 | −0.023689 | 0.034801 | no |
| last_epoch | 65 | +0.005013 | 0.025426 | no |
| min_target_entropy | 65 | −0.001624 | 0.021390 | no |

**None of the four rules' mean absolute cross-site difference exceeds the 0.077445 noise floor.**
The Swan-vs-ISAAC difference on the same 65 run_ids is, in every rule, smaller than the
difference this pipeline already produces between two identical-seed runs on the same site. This
is the replication evidence: nothing measured here separates the two sites from execution noise.

---

## 8. OPEN

- **E3 ladder expansion**: in flight on Swan as of this writing (3/72 complete,
  `swan/E3_LADDER_ADDENDUM.csv`, submitted under `swan/tools/pump_swan.sbatch` job 17891215).
  Not part of this adjudication; MATRIX_APPROVED.csv and this campaign's 107 rows are unaffected
  by it.
- **`PREREGISTRATION.md` §4 rows 1, 2, 3, 5**: not independently adjudicated here. They are all
  stated in terms of the ordering among *all three* D&A arms and OBC; row 6 (§2 above) already
  fires and, per §4's own framing (one outcome per ordering, not a ranked preference among them),
  supersedes the need to separately test rows 1/2/3/5 once the ordering itself is shown to be
  target-dependent.
- **Only the oracle rule was checked for the §2 reversal.** Whether the broad/narrow sign
  reversal also holds under the three label-free rules was not part of this instruction and is
  not reported here.
- **H2 (self-breadth) is not separately adjudicated** beyond what §2 already shows: the
  registered H2 prediction (`broad573 > narrow573`, PREREGISTRATION §2) holds on 3MDAD and fails
  on DMD, which is the same target-dependence row 6 already covers.

---

## 9. Verification — every number above, reproducible from a named CSV

```bash
# Item 1 (PRIMARY) and item 2 (MECHANISM PAIR) exact values:
python3 -c "
import csv
cells = list(csv.DictReader(open('swan/analysis/reeval_cells.csv')))
icells = list(csv.DictReader(open('analysis/reeval_cells.csv')))
def cell(rows, method, source, target, group='E1'):
    for r in rows:
        if r['group']==group and r['method']==method and r['source']==source and r['target']==target:
            return r
for target in ('3mdad','dmd'):
    for method in ('source_only','dann','cdan'):
        da = cell(cells, method,'da573_random',target); ob = cell(cells, method,'obc',target)
        print(target, method, float(da['oracle_target_val_mean']) - float(ob['oracle_target_val_mean']))
for method in ('cdan','dann','source_only'):
    for target in ('3mdad','dmd'):
        b = cell(cells, method,'broad573',target); n = cell(cells, method,'narrow573',target)
        print('swan', method, target, float(b['oracle_target_val_mean']) - float(n['oracle_target_val_mean']))
        ib = cell(icells, method,'broad573',target); inn = cell(icells, method,'narrow573',target)
        print('isaac', method, target, (float(ib['oracle_target_val_mean']) - float(inn['oracle_target_val_mean'])) if (ib and inn) else 'MISSING')
"

# Item 3 (ADAPTATION GAIN):
python3 -c "
import csv
rows = list(csv.DictReader(open('swan/analysis/adaptation_gain_swan.csv')))
for rule in ('oracle_target_val','source_val','last_epoch','min_target_entropy'):
    vals = [float(r['%s_gain'%rule]) for r in rows]
    print(rule, 'neg', sum(1 for v in vals if v<0), 'pos', sum(1 for v in vals if v>0), 'zero', sum(1 for v in vals if v==0))
"

# Item 4 (SELECTION-RULE FINDING):
python3 -c "
import csv, statistics
cells = list(csv.DictReader(open('swan/analysis/reeval_cells.csv')))
chance = lambda t: 1/6 if t=='obc' else 0.2
for label, pred in (('adversarial', lambda r: r['method'] in ('dann','cdan')), ('source_only', lambda r: r['method']=='source_only')):
    grp = [r for r in cells if pred(r)]
    for rule in ('source_val','last_epoch','min_target_entropy'):
        vals = [float(r['%s_mean'%rule]) for r in grp]
        print(label, rule, statistics.mean(vals), statistics.mean(vals) - statistics.mean(chance(r['target']) for r in grp))
"

# Item 5 (E2) and item 6 (E4 vs E1): both read directly from swan/analysis/reeval_cells.csv,
# group column E2 / E4 / E1 respectively — see the filters in the item-1/2 command above.

# Item 7 (CROSS-SITE): cat swan/analysis/crosssite_summary_swan.csv

# Noise floor 0.077445: swan/NOISE_FLOOR.md, "Test-side confirmation" table, oracle row,
# "mean pairwise |Δ|" column.

# Table 10 values (0.897/0.898/0.896) and target_label_ratio=1.0: paper_rev/main.tex:509-521.
# EDIT_SITES 8.1 citation: EDIT_SITES.md, section "8.1 The reverse-direction target was
# LABELLED, not unlabelled".
```

---

*Recorded 2026-09-07T19:06:13Z. Append-only from this point: any correction is a new dated
section below this line, never an edit above it.*

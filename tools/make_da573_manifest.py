#!/usr/bin/env python3
"""F1 - freeze the DA573-random manifest as a full E1 source arm.

CRITICAL: this must be the SAME 573 clips A2 measured, or the prereg's predictions would be
about a different set than the table. The draw is reproduced with the identical construction
used in tools/a2_containment.py -- fresh RandomState(20260830), one choice(2368, 573,
replace=False) -- and then VERIFIED by recomputing A2's containment numbers from the frozen
manifest and comparing to the stored values.
"""
import csv, json, hashlib, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stat_lib import nn_cos_dist

PREP = "<HOME_ROOT>/access_rebuttal_prep"
ARR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
MD   = "<DATA_ROOT>/memmap256/driveandact_metadata.json"
OUT  = os.path.join(ARR, "manifests")
SEED = 20260830
NAMES = ["drinking","phone_call","phone_texting","reaching","adjusting_controls","seatbelt"]

z = np.load(os.path.join(PREP, "embeddings", "driveandact_train_kinetics.npz"))
Xda, ROW = z["feat"].astype(np.float64), z["row"]
idn = json.load(open(os.path.join(PREP, "embeddings", "driveandact_train_identity.json")))
meta = json.load(open(MD))
by_row = {int(d["row"]): d for d in idn}

rng = np.random.RandomState(SEED)
pos = np.sort(rng.choice(len(Xda), 573, replace=False))
rows = [int(ROW[p]) for p in pos]
assert len(set(rows)) == 573

# ---- VERIFY this is the arm A2 measured: recompute its containment cells ----
A2 = json.load(open(os.path.join(ARR, "analysis/a2_containment.json")))
E = {d: np.load(os.path.join(PREP, "embeddings", "%s_train_kinetics.npz" % d))["feat"].astype(np.float64)
     for d in ("dmd", "3mdad")}
ok = True
for t in ("dmd", "3mdad"):
    d = nn_cos_dist(E[t], Xda[pos])
    got, want = float(np.median(d)), A2["cells"]["DA573 random (control) -> %s" % t]["median"]
    same = abs(got - want) < 1e-12
    print("  reproduce A2 median NN (%s): got %.6f stored %.6f  %s"
          % (t, got, want, "MATCH" if same else "**MISMATCH**"))
    ok = ok and same
if not ok:
    print("GATE FAIL: the frozen manifest is not the set A2 measured."); sys.exit(3)

COLS = ["row","unified_label","class_name","subject","session","view",
        "frame_start","frame_end","video_path"]
path = os.path.join(OUT, "da573_random.csv")
cls = collections.Counter(); subj=set(); sess=set(); view=set()
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
    for r in sorted(rows):
        d = by_row[r]; m = meta[r]
        assert int(m["unified_label"]) == int(d["label"])
        w.writerow({"row": r, "unified_label": d["label"], "class_name": NAMES[d["label"]],
                    "subject": d["subject"], "session": d["session"], "view": d["view"],
                    "frame_start": m["frame_start"], "frame_end": m["frame_end"],
                    "video_path": m["video_path"]})
        cls[d["label"]] += 1; subj.add(d["subject"]); sess.add(d["session"]); view.add(d["view"])
h = hashlib.sha256(open(path, "rb").read()).hexdigest()
S = json.load(open(os.path.join(OUT, "manifest_summary.json")))
S["da573_random"] = {"path": path, "sha256": h, "n": 573, "seed": SEED,
                     "draw": "RandomState(20260830).choice(2368, 573, replace=False) over the "
                             "fold-0 D&A train embedding order; verified against A2's cells",
                     "class_counts": {NAMES[k]: v for k, v in sorted(cls.items())},
                     "subjects": len(subj), "sessions": len(sess), "views": sorted(view)}
json.dump(S, open(os.path.join(OUT, "manifest_summary.json"), "w"), indent=1)
with open(os.path.join(OUT, "MANIFESTS.sha256"), "w") as f:
    for t in ("broad573", "narrow573", "da573_random"):
        f.write("%s  %s.csv\n" % (S[t]["sha256"], t))
print("\nda573_random n=573 subj=%d sess=%d sha256=%s" % (len(subj), len(sess), h[:16]))
print("class counts:", {NAMES[k]: v for k, v in sorted(cls.items())})
print("broad/narrow class counts:", S["broad573"]["class_counts"])
print("\nNOTE: da573_random is an UNBIASED draw, so its class distribution is NOT forced to match")
print("broad573/narrow573 - it follows D&A's own proportions up to sampling noise. That is the")
print("point of the arm; it is reported, not corrected.")

#!/usr/bin/env python3
"""B4 — frozen source manifests for broad573 / narrow573, in the format the patched
dataloader consumes (`row` = index into driveandact_metadata.json), carrying subject and
session. Also emits a per-manifest summary so the two are comparable at a glance.
"""
import csv, json, hashlib, os, sys, collections
PREP = "<HOME_ROOT>/access_rebuttal_prep"
ARR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
MD   = "<DATA_ROOT>/memmap256/driveandact_metadata.json"
OUT  = os.path.join(ARR, "manifests"); os.makedirs(OUT, exist_ok=True)
NAMES = ["drinking","phone_call","phone_texting","reaching","adjusting_controls","seatbelt"]

subs = json.load(open(os.path.join(PREP, "partD_subsets.json")))
idn  = json.load(open(os.path.join(PREP, "embeddings", "driveandact_train_identity.json")))
meta = json.load(open(MD))
by_row = {int(d["row"]): d for d in idn}

COLS = ["row","unified_label","class_name","subject","session","view",
        "frame_start","frame_end","video_path"]
summary = {}
for tag in ("broad573", "narrow573"):
    rows = subs["%s_rows" % tag]
    assert len(rows) == len(set(rows)) == 573, "%s: %d rows, %d unique" % (tag, len(rows), len(set(rows)))
    path = os.path.join(OUT, "%s.csv" % tag)
    cls = collections.Counter(); subj = set(); sess = set(); view = set()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in sorted(rows):
            d = by_row[r]; m = meta[r]
            assert int(m["unified_label"]) == int(d["label"]), "label mismatch at row %d" % r
            assert m["video_path"] == d["video_path"], "path mismatch at row %d" % r
            w.writerow({"row": r, "unified_label": d["label"], "class_name": NAMES[d["label"]],
                        "subject": d["subject"], "session": d["session"], "view": d["view"],
                        "frame_start": m["frame_start"], "frame_end": m["frame_end"],
                        "video_path": m["video_path"]})
            cls[d["label"]] += 1; subj.add(d["subject"]); sess.add(d["session"]); view.add(d["view"])
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    summary[tag] = {"path": path, "sha256": h, "n": len(rows),
                    "class_counts": {NAMES[k]: v for k, v in sorted(cls.items())},
                    "subjects": len(subj), "sessions": len(sess), "views": sorted(view)}
    print("%-10s n=%d  subj=%d sess=%d views=%s  sha256=%s"
          % (tag, len(rows), len(subj), len(sess), sorted(view), h[:16]))

# the two must have identical class distributions - that was the construction constraint
a, b = summary["broad573"]["class_counts"], summary["narrow573"]["class_counts"]
if a != b:
    print("GATE FAIL: class distributions differ\n  broad =%s\n  narrow=%s" % (a, b)); sys.exit(3)
print("class distribution (identical in both):", a)
ov = len(set(subs["broad573_rows"]) & set(subs["narrow573_rows"]))
summary["overlap_clips"] = ov
print("overlap:", ov, "clips")

with open(os.path.join(OUT, "MANIFESTS.sha256"), "w") as f:
    for tag in ("broad573", "narrow573"):
        f.write("%s  %s.csv\n" % (summary[tag]["sha256"], tag))
json.dump(summary, open(os.path.join(OUT, "manifest_summary.json"), "w"), indent=1)
print("frozen ->", os.path.join(OUT, "MANIFESTS.sha256"))

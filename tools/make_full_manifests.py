#!/usr/bin/env python3
"""I1 - freeze the FULL fold-0 train manifests, in the same format and with the same guarantees
as the 573-clip ones. These are the E2 reverse-direction source specifications, and the
Drive&Act one is also the 'DriveAndAct full (2368)' source arm.

Same columns, same freeze discipline. Nothing is sampled here: these are the complete fold-0
train splits, so the only thing to get wrong is the row set, which is asserted against the
audited counts.
"""
import csv, json, hashlib, os, sys, collections
CODE = "<DATA_ROOT>/UMTRI/Occupant_Behavior/code"
ARR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
OUT  = os.path.join(ARR, "manifests")
NAMES = ["drinking","phone_call","phone_texting","reaching","adjusting_controls","seatbelt"]
EXPECT = {"driveandact": 2368, "dmd": 2659, "3mdad": 153}

def ident(ds, p):
    import re
    if ds == "driveandact":
        m = re.search(r"/kinect_color/(vp\d+)/(.+)\.(\w+)\.mp4$", p)
        return m.group(1), m.group(2), m.group(3)
    if ds == "dmd":
        m = re.search(r"/dmd/(g[A-Z])/(\d+)/(s\d+)/([^/]+)_((?:rgb|ir|depth)_\w+)\.mp4$", p)
        return "%s/%s" % (m.group(1), m.group(2)), "%s/%s/%s" % (m.group(1), m.group(2), m.group(3)), m.group(5)
    m = re.search(r"/videos/S(\d+)_AC(\d+)\.mp4$", p)
    return "S" + m.group(1), "Day", "RGB1"

COLS = ["row","unified_label","class_name","subject","session","view",
        "frame_start","frame_end","video_path"]
S = json.load(open(os.path.join(OUT, "manifest_summary.json")))
made = []
for ds in ("driveandact", "dmd", "3mdad"):
    meta = json.load(open(os.path.join(CODE, "memmap256", "%s_metadata.json" % ds)))
    rows = [i for i, m in enumerate(meta) if m.get("fold_assignment", {}).get("0") == "train"]
    if len(rows) != EXPECT[ds]:
        print("GATE FAIL: %s fold-0 train is %d rows, audited value is %d"
              % (ds, len(rows), EXPECT[ds])); sys.exit(3)
    tag = "%s_full%d" % (ds, len(rows))
    path = os.path.join(OUT, "%s.csv" % tag)
    cls = collections.Counter(); subj=set(); sess=set(); view=set()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows:
            m = meta[r]; sj, ss, vw = ident(ds, m["video_path"])
            w.writerow({"row": r, "unified_label": m["unified_label"],
                        "class_name": NAMES[m["unified_label"]], "subject": sj, "session": ss,
                        "view": vw, "frame_start": m["frame_start"], "frame_end": m["frame_end"],
                        "video_path": m["video_path"]})
            cls[m["unified_label"]] += 1; subj.add(sj); sess.add(ss); view.add(vw)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    S[tag] = {"path": path, "sha256": h, "n": len(rows), "dataset": ds,
              "construction": "complete fold-0 train split of %s (no sampling)" % ds,
              "memmap": os.path.join(CODE, "memmap256", "%s_frames.npy" % ds),
              "metadata": os.path.join(CODE, "memmap256", "%s_metadata.json" % ds),
              "class_counts": {NAMES[k]: v for k, v in sorted(cls.items())},
              "subjects": len(subj), "sessions": len(sess), "views": sorted(view)}
    made.append(tag)
    print("%-22s n=%-5d subj=%-3d sess=%-4d views=%s sha256=%s"
          % (tag, len(rows), len(subj), len(sess), sorted(view), h[:16]))
json.dump(S, open(os.path.join(OUT, "manifest_summary.json"), "w"), indent=1)
with open(os.path.join(OUT, "MANIFESTS.sha256"), "w") as f:
    for t in sorted(k for k in S if k != "overlap_clips" and isinstance(S[k], dict) and "sha256" in S[k]):
        f.write("%s  %s.csv\n" % (S[t]["sha256"], t))
print("\nfrozen ->", os.path.join(OUT, "MANIFESTS.sha256"))

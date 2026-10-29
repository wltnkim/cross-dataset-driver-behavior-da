"""Check every matrix in cm_data.py against its run's classification_report.txt
(per-class support, recall, precision at 4 decimals, and overall accuracy).
Run from anywhere: python3 validate_cm.py   (exits 1 on any mismatch)."""
import os, re, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cm_data import CM, R
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
C = ["drinking", "phone_call", "phone_texting", "reaching", "adjusting_controls", "seatbelt"]
bad = 0
for k, (rep, m) in CM.items():
    m = np.array(m); txt = open(os.path.join(ROOT, R, rep)).read()
    for i in range(m.shape[0]):
        r = re.search(re.escape(C[i]) + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", txt)
        sup, colsum = m[i].sum(), m[:, i].sum()
        if r is None:
            if sup == 0 and colsum == 0: continue
            bad += 1; print("MISSING", k, C[i]); continue
        P, Rc, S = float(r[1]), float(r[2]), int(r[4])
        rec = m[i, i] / sup if sup else 0; prec = m[i, i] / colsum if colsum else 0
        if not (S == sup and abs(rec - Rc) < 6e-5 and abs(prec - P) < 6e-5):
            bad += 1; print("MISMATCH", k, C[i])
    acc = re.search(r"Test Accuracy: ([\d.]+)", txt) or re.search(r"accuracy\s+([\d.]+)", txt)
    if acc and abs(np.trace(m) / m.sum() - float(acc[1])) > 6e-5:
        bad += 1; print("ACC MISMATCH", k)
print("matrices checked:", len(CM), "mismatches:", bad)
sys.exit(1 if bad else 0)

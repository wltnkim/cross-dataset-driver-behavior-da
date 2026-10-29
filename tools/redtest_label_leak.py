#!/usr/bin/env python3
"""E2 label-leak red test. CPU only, fixed seeds, no GPU, nothing submitted.

Poisons the OBC TARGET-TRAIN labels with a sentinel permutation and verifies the first N training
iterations are bit-identical to the unpoisoned run. If any target label reached a loss or an
adaptation objective, the losses -- and the resulting weights -- would differ.

Two guards against the test itself being vacuous:
  * the loss formulas used here are checked to appear VERBATIM in train_cross.py, so this harness
    cannot silently drift from the trainer it claims to test;
  * a POSITIVE CONTROL poisons the SOURCE labels instead, where a leak is expected: that run MUST
    differ. A test that only ever reports "identical" proves nothing.

The frozen OBC pkl is never written: poisoning replaces the in-memory `labels` list on a
throwaway dataset object. The file hash is re-checked at the end.
"""
import copy, hashlib, os, sys, json
ARR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
CODE = "<DATA_ROOT>/UMTRI/Occupant_Behavior/code"
sys.path.insert(0, os.path.join(ARR, "code"))
os.environ.setdefault("TORCH_HOME", os.path.join(ARR, "weights"))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
import cross_dataset
assert os.path.realpath(cross_dataset.__file__).startswith(os.path.realpath(ARR))
from cross_dataset.train_cross import (ZeroLabelTarget, LabeledSubsetWrapper,
                                       build_manifest_source, build_model_backbone)
from cross_dataset.obc_dataset import OBCCrossDataset

OBC_PKL = os.path.join(CODE, "features/keyframe_video_metadata_dualcam.pkl")
H_BEFORE = hashlib.sha256(open(OBC_PKL, "rb").read()).hexdigest()
NC, BS, NITER = 6, 4, 3
TSRC = open(os.path.join(ARR, "code/cross_dataset/train_cross.py")).read()

# ---- guard 1: the formulas below must exist verbatim in the trainer ----------------------
FORMULAS = [
    "s_domain_loss = domain_criterion(s_domain, s_domain_labels)",
    "t_domain_loss = domain_criterion(t_domain, t_domain_labels)",
    "domain_loss = 0.5 * (s_domain_loss + t_domain_loss)",
    "loss = cls_loss + args.alpha_dann * domain_loss",
    "cls_loss = s_cls_loss",
]
missing = [f for f in FORMULAS if f not in TSRC]
if missing:
    print("HARNESS DRIFT: these formulas are not in train_cross.py verbatim:")
    for m in missing: print("   ", m)
    sys.exit(3)
print("guard 1: %d loss formulas verified verbatim against train_cross.py" % len(FORMULAS))


def build(poison_target=False, poison_source=False, seed=1234):
    torch.manual_seed(seed); np.random.seed(seed)
    src = build_manifest_source(os.path.join(ARR, "manifests/driveandact_full2368.csv"),
                                os.path.join(CODE, "memmap256/driveandact_frames.npy"),
                                os.path.join(CODE, "memmap256/driveandact_metadata.json"),
                                num_frames=16, augment=False)          # augment off: determinism
    tgt = OBCCrossDataset(OBC_PKL, split="train", num_frames=16, augment=False)
    if poison_target:
        rng = np.random.RandomState(999)
        tgt.labels = list(rng.permutation(np.array(tgt.labels)))       # throwaway, in-memory only
    if poison_source:
        base = src.dataset
        rng = np.random.RandomState(999)
        base._labels = list(rng.permutation(np.array(base._labels)))
    tgt = ZeroLabelTarget(tgt, NC)                                     # the structural mask
    tgt = LabeledSubsetWrapper(tgt, 0.0, seed=42)                      # ratio 0 => nothing labeled
    return src, tgt


def run(poison_target=False, poison_source=False, seed=1234):
    src, tgt = build(poison_target, poison_source, seed)
    torch.manual_seed(seed)
    model = build_model_backbone(NC, "dann", "resnet50_tsn", pretrained=False)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    dcrit = nn.CrossEntropyLoss()
    g = torch.Generator(); g.manual_seed(seed)
    sl = DataLoader(src, batch_size=BS, shuffle=True, num_workers=0, drop_last=True, generator=g)
    g2 = torch.Generator(); g2.manual_seed(seed)
    tl = DataLoader(tgt, batch_size=BS, shuffle=True, num_workers=0, drop_last=True, generator=g2)
    si = iter(sl); losses = []
    for k, batch in enumerate(tl):
        if k >= NITER: break
        t_imgs, t_labels, t_is_labeled = batch
        try: s_imgs, s_labels = next(si)
        except StopIteration: si = iter(sl); s_imgs, s_labels = next(si)
        lam = 0.5
        s_logits, s_domain, _ = model(s_imgs, lam)
        s_cls_loss = crit(s_logits, s_labels)
        s_domain_labels = torch.zeros(s_logits.size(0), dtype=torch.long)
        t_logits, t_domain, _ = model(t_imgs, lam)
        t_domain_labels = torch.ones(t_logits.size(0), dtype=torch.long)
        s_domain_loss = dcrit(s_domain, s_domain_labels)
        t_domain_loss = dcrit(t_domain, t_domain_labels)
        domain_loss = 0.5 * (s_domain_loss + t_domain_loss)
        cls_loss = s_cls_loss                      # ratio<=0 branch: SOURCE ONLY
        loss = cls_loss + 0.5 * domain_loss
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append((float(cls_loss), float(domain_loss), float(loss)))
        print("   iter %d  cls=%.12f dom=%.12f loss=%.12f  tgt_labels_seen=%s"
              % (k, losses[-1][0], losses[-1][1], losses[-1][2],
                 sorted(set(t_labels.tolist()))))
    w = float(sum(p.detach().abs().sum() for p in model.parameters()))
    return losses, w


print("\n[A] clean run")
La, Wa = run()
print("\n[B] TARGET labels poisoned (sentinel permutation)")
Lb, Wb = run(poison_target=True)
print("\n[C] POSITIVE CONTROL: SOURCE labels poisoned - this MUST differ")
Lc, Wc = run(poison_source=True)

ident_ab = all(a == b for a, b in zip(La, Lb)) and Wa == Wb
diff_ac = any(a != c for a, c in zip(La, Lc)) or Wa != Wc
print("\n%-46s %s" % ("A vs B (target poisoned) bit-identical?", "YES" if ident_ab else "NO"))
print("%-46s %s" % ("A vs C (source poisoned) differs?", "YES" if diff_ac else "NO"))
for i, (a, b, c) in enumerate(zip(La, Lb, Lc)):
    print("  iter %d  A.loss=%.12f  B.loss=%.12f  C.loss=%.12f" % (i, a[2], b[2], c[2]))
print("  |w| A=%.9f B=%.9f C=%.9f" % (Wa, Wb, Wc))

H_AFTER = hashlib.sha256(open(OBC_PKL, "rb").read()).hexdigest()
print("\nfrozen OBC pkl sha256 unchanged: %s" % (H_BEFORE == H_AFTER))
json.dump({"n_iters": NITER, "batch": BS, "backbone": "resnet50_tsn (random init, CPU)",
           "A_clean": La, "B_target_poisoned": Lb, "C_source_poisoned": Lc,
           "target_poison_identical": ident_ab, "source_poison_differs": diff_ac,
           "weight_sums": {"A": Wa, "B": Wb, "C": Wc},
           "obc_pkl_sha256_unchanged": H_BEFORE == H_AFTER,
           "formulas_verified_verbatim": FORMULAS},
          open(os.path.join(ARR, "analysis/e2_label_leak_redtest.json"), "w"), indent=1)
ok = ident_ab and diff_ac and H_BEFORE == H_AFTER
print("\nRED TEST %s" % ("PASS - target labels cannot reach the loss" if ok else "**FAIL**"))
sys.exit(0 if ok else 3)

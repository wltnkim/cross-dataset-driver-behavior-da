"""Cross-dataset VideoMAE training for driver behavior recognition.

Supports multiple experiment modes:
  - target_only:  Train on Drive&Act only (target upper bound)
  - finetune:     OBC pretrained → Drive&Act fine-tune (simple transfer)
  - dann:         DANN domain adaptation (OBC source + D&A target)
  - cbdann:       Class-Balanced DANN (class-aware domain loss weighting)
  - cdan:         Conditional Domain Adversarial Network (DANN + class conditioning)
  - mcc:          Minimum Class Confusion (target-side regularization)
  - bnm:          Batch Nuclear-norm Maximization (target-side regularization)

Usage:
  python -m cross_dataset.train_cross \
      --mode target_only \
      --dna_root <DATA_ROOT>/DriveAndAct \
      --fold 0 --epochs 50 --batch_size 8

  python -m cross_dataset.train_cross \
      --mode finetune \
      --pretrained results/20260305_191532_videomae_da_dann_none/best_model.pt \
      --dna_root <DATA_ROOT>/DriveAndAct \
      --fold 0

  python -m cross_dataset.train_cross \
      --mode dann \
      --obc_metadata features/keyframe_video_metadata_dualcam.pkl \
      --dna_root <DATA_ROOT>/DriveAndAct \
      --fold 0 --alpha_dann 0.5

  python -m cross_dataset.train_cross \
      --mode cdan \
      --obc_metadata features/keyframe_video_metadata_dualcam.pkl \
      --dna_root <DATA_ROOT>/DriveAndAct \
      --fold 0 --alpha_dann 0.5

  python -m cross_dataset.train_cross \
      --mode mcc \
      --obc_metadata features/keyframe_video_metadata_dualcam.pkl \
      --dna_root <DATA_ROOT>/DriveAndAct \
      --fold 0 --alpha_dann 0.5 --alpha_mcc 1.0

  python -m cross_dataset.train_cross \
      --mode bnm \
      --obc_metadata features/keyframe_video_metadata_dualcam.pkl \
      --dna_root <DATA_ROOT>/DriveAndAct \
      --fold 0 --alpha_dann 0.5 --alpha_bnm 0.1
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

from cross_dataset.class_mapping_utils import (
    get_unified_class_names,
    get_num_unified_classes,
    get_unified_label_maps,
    load_class_mapping,
)
from cross_dataset.driveandact_dataset import DriveAndActDataset
from cross_dataset.dmd_dataset import DMDDataset
from cross_dataset.mdad_dataset import MDADDataset
from cross_dataset.obc_dataset import OBCCrossDataset


# ========== [ARR] helpers ==========

class ZeroLabelTarget(torch.utils.data.Dataset):
    """[ARR] E2 label discipline: the 0-label target carries NO usable label.

    Returns (imgs, SENTINEL) with SENTINEL = num_classes, which is OUT OF RANGE for a
    num_classes-way CrossEntropyLoss. Two deliberate consequences:
      * the true target-train label never enters the returned tensor, so it cannot reach any
        loss or adaptation objective even by accident;
      * if a future edit did compute a classification loss on the target here, PyTorch raises
        "Target N is out of bounds" instead of silently training on labels.
    Target labels are still read in exactly two places, both reporting-only: the eval block
    (held-out target TEST split) and the metrics logger (target-val curve).
    """

    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.sentinel = int(num_classes)
        print("[ARR-0label] target train labels replaced by sentinel %d (out of range for a "
              "%d-way loss); %d samples" % (self.sentinel, num_classes, len(dataset)))

    def __len__(self):
        return len(self.dataset)

    def get_labels(self):
        return [self.sentinel] * len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        return item[0], torch.tensor(self.sentinel, dtype=torch.long)


class DropClassDataset(torch.utils.data.Dataset):
    """[ARR] Keep only samples whose label != drop_label. Labels 0..4 stay contiguous when
    drop_label == 5, so no remap is needed; asserted below."""

    def __init__(self, dataset, drop_label):
        self.dataset = dataset
        self.drop_label = int(drop_label)
        labels = _dataset_labels(dataset)
        self.indices = [i for i, y in enumerate(labels) if int(y) != self.drop_label]
        kept = sorted({int(y) for y in labels if int(y) != self.drop_label})
        assert not kept or max(kept) < self.drop_label, (
            "[ARR] dropping label %d would leave a gap: kept=%s" % (self.drop_label, kept))
        print("[ARR-nosb5] %d -> %d samples (dropped label %d)"
              % (len(labels), len(self.indices), self.drop_label))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


def _dataset_labels(ds):
    """[ARR] Labels without decoding pixels, for every dataset class used here."""
    if hasattr(ds, "get_labels"):
        return list(ds.get_labels())
    for attr in ("labels", "_labels"):
        if hasattr(ds, attr):
            return [int(x) for x in getattr(ds, attr)]
    if hasattr(ds, "samples"):
        return [int(s["label"]) for s in ds.samples]
    if isinstance(ds, torch.utils.data.Subset):
        base = _dataset_labels(ds.dataset)
        return [base[i] for i in ds.indices]
    raise TypeError("[ARR] cannot read labels from %s without decoding" % type(ds).__name__)


def build_manifest_source(manifest_csv, memmap_path, metadata_path, num_frames,
                          augment, temporal_jitter=0, color_jitter=0.0):
    """[ARR] Build a SOURCE dataset from a frozen manifest of memmap metadata rows.

    The manifest carries `row` = index into metadata.json. MemmapVideoDataset exposes
    `_indices` (dataset position -> metadata row), so the mapping is row -> position.
    Fails loudly if any manifest row is absent from the fold-0 train split.
    """
    import csv as _csv
    from cross_dataset.memmap_dataset import MemmapVideoDataset
    base = MemmapVideoDataset(memmap_path, metadata_path, fold=0, split="train",
                              num_frames=num_frames, augment=augment,
                              color_jitter=color_jitter, crop_size=224)
    row_to_pos = {int(r): p for p, r in enumerate(base._indices)}
    rows, missing = [], []
    with open(manifest_csv) as f:
        for rec in _csv.DictReader(f):
            r = int(rec["row"])
            if r in row_to_pos:
                rows.append(row_to_pos[r])
            else:
                missing.append(r)
    if missing:
        raise SystemExit("[ARR] %d manifest rows are not in the fold-0 train split, e.g. %s"
                         % (len(missing), missing[:5]))
    if not rows:
        raise SystemExit("[ARR] manifest %s produced ZERO usable rows" % manifest_csv)
    print("[ARR-source] %s -> %d clips from %s" % (manifest_csv, len(rows), memmap_path))
    return torch.utils.data.Subset(base, rows)


@torch.no_grad()
def eval_split(model, loader, device, mode, da_modes, fp16, want_entropy=False):
    """[ARR] One forward pass -> (balanced accuracy, mean prediction entropy).

    Entropy uses only the logits, never the labels, so it is a legitimate label-free
    selection signal. Natural log; the ceiling is log(num_classes).
    """
    model.eval()
    preds, labs, ents = [], [], []
    for batch in loader:
        imgs, labels = batch[0], batch[1]
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda", enabled=fp16):
            if mode in da_modes:
                logits, _, _ = model(imgs, 0.0)
            else:
                logits = model(pixel_values=imgs).logits
        logits = logits.float()
        preds.extend(logits.argmax(-1).cpu().numpy())
        labs.extend(labels.numpy())
        if want_entropy:
            p = torch.softmax(logits, dim=1)
            ents.append((-(p * torch.log(p + 1e-12)).sum(1)).cpu().numpy())
    ba = balanced_accuracy_score(labs, preds)
    ent = float(np.concatenate(ents).mean()) if ents else None
    return ba, ent


# ========== SSDA: Labeled Subset Wrapper ==========

class LabeledSubsetWrapper(torch.utils.data.Dataset):
    """Wraps a dataset, marking only a fraction of samples as 'labeled'.

    For SSDA experiments: labeled samples contribute classification loss,
    all samples contribute domain alignment loss.

    Returns (imgs, label, is_labeled) where is_labeled is a bool.
    """

    def __init__(self, dataset, label_ratio=1.0, seed=42):
        self.dataset = dataset
        n = len(dataset)
        if label_ratio >= 1.0:
            self.labeled_mask = np.ones(n, dtype=bool)
        elif label_ratio <= 0.0:
            self.labeled_mask = np.zeros(n, dtype=bool)
        else:
            rng = np.random.RandomState(seed)
            n_labeled = max(1, int(n * label_ratio))
            indices = rng.choice(n, n_labeled, replace=False)
            self.labeled_mask = np.zeros(n, dtype=bool)
            self.labeled_mask[indices] = True
        self.n_labeled = int(self.labeled_mask.sum())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        imgs, label = self.dataset[idx]
        return imgs, label, self.labeled_mask[idx]


# ========== Gradient Reversal Layer (DANN) ==========

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)


# ========== DANN Model Wrapper ==========

class VideoMAEDANN(nn.Module):
    """VideoMAE + Domain Classifier for DANN/CDAN."""

    def __init__(self, num_classes, domain_hidden=256, drop_path_rate=0.1,
                 use_cdan=False, cdan_random_dim=1024,
                 model_name="MCG-NJU/videomae-base-finetuned-kinetics",
                 no_pretrain=False):
        super().__init__()
        config = VideoMAEConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        config.drop_path_rate = drop_path_rate
        if no_pretrain:
            self.videomae = VideoMAEForVideoClassification(config)
            print(f"[Model] RANDOM-INIT {model_name} (no_pretrain)")
        else:
            self.videomae = VideoMAEForVideoClassification.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
        hidden_size = config.hidden_size  # 768
        self.use_cdan = use_cdan
        self.num_classes = num_classes

        self.grl = GradientReversalLayer()

        if use_cdan:
            # CDAN: randomized multilinear conditioning (Long et al., NeurIPS 2018)
            # Random projection: (features ⊗ softmax) → cdan_random_dim
            self.cdan_random_dim = cdan_random_dim
            self.random_matrix_feat = nn.Parameter(
                torch.randn(hidden_size, cdan_random_dim), requires_grad=False
            )
            self.random_matrix_cls = nn.Parameter(
                torch.randn(num_classes, cdan_random_dim), requires_grad=False
            )
            domain_input_dim = cdan_random_dim
        else:
            domain_input_dim = hidden_size

        self.domain_classifier = nn.Sequential(
            nn.Linear(domain_input_dim, domain_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(domain_hidden, 2),
        )

    def _cdan_conditioning(self, features, logits):
        """Randomized multilinear conditioning for CDAN."""
        softmax_output = torch.softmax(logits, dim=1)
        # Random projection: element-wise product of two projected vectors
        feat_proj = torch.mm(features, self.random_matrix_feat)  # (B, R)
        cls_proj = torch.mm(softmax_output, self.random_matrix_cls)  # (B, R)
        return feat_proj * cls_proj  # (B, R)

    def forward(self, pixel_values, lambda_dann=1.0):
        outputs = self.videomae.videomae(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state

        # Use CLS token or mean pooling
        if self.videomae.fc_norm is not None:
            features = self.videomae.fc_norm(sequence_output.mean(1))
        else:
            features = sequence_output[:, 0]

        # Action classification
        logits = self.videomae.classifier(features)

        # Domain classification (with GRL)
        if self.use_cdan:
            cdan_features = self._cdan_conditioning(features, logits)
            reversed_features = self.grl(cdan_features, lambda_dann)
        else:
            reversed_features = self.grl(features, lambda_dann)
        domain_logits = self.domain_classifier(reversed_features)

        return logits, domain_logits, features


# ========== Auxiliary DA Losses ==========

def mcc_loss(logits, temperature=2.5):
    """Minimum Class Confusion loss (Jin et al., ECCV 2020).
    Minimizes off-diagonal entries of the class correlation matrix.
    """
    softmax_out = torch.softmax(logits / temperature, dim=1)
    # Class correlation matrix: C = Y^T Y / batch_size
    batch_size = softmax_out.size(0)
    correlation = torch.mm(softmax_out.t(), softmax_out) / batch_size
    # Minimize off-diagonal (exclude diagonal)
    mask = 1.0 - torch.eye(correlation.size(0), device=correlation.device)
    loss = (correlation * mask).sum() / mask.sum()
    return loss


def bnm_loss(logits):
    """Batch Nuclear-norm Maximization loss (Cui et al., CVPR 2020).
    Maximizes nuclear norm of the batch softmax matrix.
    """
    softmax_out = torch.softmax(logits, dim=1)
    # Nuclear norm = sum of singular values; maximize → minimize negative
    _, s, _ = torch.svd(softmax_out)
    return -s.sum()


def build_model_backbone(num_classes, mode, backbone, da_modes=("dann", "cbdann", "cdan", "mcc", "bnm"),
                         pretrained=True):
    """[ARR] CNN backbones for the E3 ladder. videomae never reaches here."""
    from cross_dataset.backbones import CNNDANN, CNNPlain
    if mode in da_modes:
        return CNNDANN(backbone, num_classes, GradientReversalLayer, pretrained=pretrained)
    return CNNPlain(backbone, num_classes, pretrained=pretrained)


def build_model(num_classes, mode, pretrained_path=None, drop_path_rate=0.1,
                model_name="MCG-NJU/videomae-base-finetuned-kinetics",
                no_pretrain=False):
    """Build model based on experiment mode."""
    if mode in ("dann", "cbdann", "mcc", "bnm"):
        model = VideoMAEDANN(num_classes, drop_path_rate=drop_path_rate,
                             use_cdan=False, model_name=model_name,
                             no_pretrain=no_pretrain)
        print(f"[Model] VideoMAE-DANN ({mode}) [{model_name}{', random-init' if no_pretrain else ''}], {num_classes} classes")
        return model
    if mode == "cdan":
        model = VideoMAEDANN(num_classes, drop_path_rate=drop_path_rate,
                             use_cdan=True, model_name=model_name,
                             no_pretrain=no_pretrain)
        print(f"[Model] VideoMAE-CDAN [{model_name}{', random-init' if no_pretrain else ''}], {num_classes} classes")
        return model

    # target_only or finetune: standard VideoMAE
    config = VideoMAEConfig.from_pretrained(model_name)
    config.num_labels = num_classes
    config.drop_path_rate = drop_path_rate
    if no_pretrain:
        model = VideoMAEForVideoClassification(config)
        print(f"[Model] RANDOM-INIT {model_name} (no_pretrain), {num_classes} classes")
    else:
        model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

    if mode == "finetune" and pretrained_path:
        # Load OBC pretrained weights, replace head for unified classes
        print(f"[Model] Loading OBC pretrained: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Load backbone weights (skip classifier and domain_classifier)
        filtered = {}
        for k, v in state_dict.items():
            if "domain_classifier" in k:
                continue
            if "classifier" in k:
                continue  # skip old 33-class head
            filtered[k] = v

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[Model] Loaded backbone. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        print(f"[Model] Fine-tuning with new {num_classes}-class head")
    else:
        init_desc = "random-init" if no_pretrain else "Kinetics pretrained"
        print(f"[Model] Target-only training [{model_name}, {init_desc}], {num_classes} classes")

    return model


def get_dann_lambda(epoch, max_epochs):
    """Progressive lambda schedule for DANN (0 -> 1)."""
    p = epoch / max_epochs
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


def set_seed(seed):
    """Seed RNGs for reproducibility. Does NOT force deterministic cuDNN algorithms
    (avoids crashes on VideoMAE ops); residual GPU nondeterminism is accepted."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    start_time = time.time()
    if getattr(args, "seed", None) is not None:
        set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = get_num_unified_classes()
    class_names = get_unified_class_names()
    # [ARR] label-space control. unified6 leaves this untouched.
    _arr_drop_label = None
    if getattr(args, "label_space", "unified6") == "nosb5":
        _arr_drop_label = 5
        assert class_names[_arr_drop_label] == "seatbelt", \
            "[ARR] expected unified label 5 == seatbelt, got %s" % class_names[_arr_drop_label]
        class_names = class_names[:_arr_drop_label]
        num_classes = len(class_names)
        print("[ARR-nosb5] head resized to %d classes: %s" % (num_classes, class_names))
    print(f"[Info] Unified classes ({num_classes}): {class_names}")
    print(f"[Info] Mode: {args.mode}, Fold: {args.fold}, Device: {device}")

    # Result directory
    if args.resume_dir:
        result_dir = args.resume_dir
        assert os.path.isdir(result_dir), f"Resume dir not found: {result_dir}"
        print(f"[Resume] Resuming from {result_dir}")
    else:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tgt_suffix = "" if args.target_dataset == "driveandact" else f"_{args.target_dataset}"
        ratio_suffix = ""
        if args.target_label_ratio < 1.0:
            ratio_suffix = f"_r{args.target_label_ratio:.2f}".replace(".", "")
        bb_suffix = ""
        if args.no_pretrain:
            bb_suffix = "_bbRandom"
        elif "videomae-small" in args.model_name:
            bb_suffix = "_bbSmall"
        elif "videomae-large" in args.model_name:
            bb_suffix = "_bbLarge"
        rev_suffix = "_reverse" if args.reverse_direction else ""
        tag_suffix = f"_{args.run_tag}" if getattr(args, "run_tag", "") else ""
        result_dir = os.path.join("results", f"{now}_cross_{args.mode}{tgt_suffix}{ratio_suffix}{bb_suffix}{rev_suffix}{tag_suffix}_fold{args.fold}")
        os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # ===== Datasets =====
    # Target dataset
    if args.memmap_path and args.memmap_metadata:
        # Option D: pre-decoded uint8 memmap (no decord I/O at train time).
        from cross_dataset.memmap_dataset import MemmapVideoDataset
        print(f"[memmap] using {args.memmap_path} (metadata={args.memmap_metadata})")
        _mm_crop = args.memmap_crop if args.memmap_crop > 0 else None
        dna_train = MemmapVideoDataset(
            args.memmap_path, args.memmap_metadata,
            fold=args.fold, split="train",
            num_frames=args.num_frames, augment=True, color_jitter=0.3,
            crop_size=_mm_crop,
        )
        # val split tagged in metadata (2026-06-05) — model selection on val,
        # final eval on test, matching the decord pipeline (no test leakage).
        dna_val = MemmapVideoDataset(
            args.memmap_path, args.memmap_metadata,
            fold=args.fold, split="val",
            num_frames=args.num_frames, augment=False,
            crop_size=_mm_crop,
        )
        dna_test = MemmapVideoDataset(
            args.memmap_path, args.memmap_metadata,
            fold=args.fold, split="test",
            num_frames=args.num_frames, augment=False,
            crop_size=_mm_crop,
        )
    elif args.target_dataset == "driveandact":
        dna_train = DriveAndActDataset(
            args.dna_root, split="train", fold=args.fold,
            num_frames=args.num_frames, augment=True,
            temporal_jitter=2, color_jitter=0.3,
        )
        dna_val = DriveAndActDataset(
            args.dna_root, split="val", fold=args.fold,
            num_frames=args.num_frames, augment=False,
        )
        dna_test = DriveAndActDataset(
            args.dna_root, split="test", fold=args.fold,
            num_frames=args.num_frames, augment=False,
        )
    elif args.target_dataset == "dmd":
        dna_train = DMDDataset(
            args.dmd_root, split="train", fold=args.fold,
            num_frames=args.num_frames, augment=True,
            temporal_jitter=2, color_jitter=0.3,
        )
        dna_val = DMDDataset(
            args.dmd_root, split="val", fold=args.fold,
            num_frames=args.num_frames, augment=False,
        )
        dna_test = DMDDataset(
            args.dmd_root, split="test", fold=args.fold,
            num_frames=args.num_frames, augment=False,
        )
    elif args.target_dataset == "obc":
        # [ARR] E2. Same data path as the OBC SOURCE arm (the 51-frame memmap cache via the
        # frozen OBCCrossDataset), so source-side and target-side OBC are identical inputs.
        assert not args.memmap_path, \
            "[ARR] --target_dataset obc is incompatible with --memmap_path (OBC has its own cache)"
        dna_train = OBCCrossDataset(
            args.obc_metadata, split="train", num_frames=args.num_frames,
            augment=True, temporal_jitter=2, color_jitter=0.3)
        dna_val = OBCCrossDataset(
            args.obc_metadata, split="val", num_frames=args.num_frames, augment=False)
        dna_test = OBCCrossDataset(
            args.obc_metadata, split="test", num_frames=args.num_frames, augment=False)
        print("[ARR-E2] target = OBC (train/val/test = %d/%d/%d). Selection on OBC val, "
              "reporting on OBC TEST. The frozen --reverse_direction block instead set "
              "dna_test = obc val, i.e. it selected and reported on one split."
              % (len(dna_train), len(dna_val), len(dna_test)))
    elif args.target_dataset == "3mdad":
        dna_train = MDADDataset(
            args.mdad_root, split="train", fold=args.fold,
            num_frames=args.num_frames, augment=True,
            temporal_jitter=2, color_jitter=0.3,
        )
        dna_val = MDADDataset(
            args.mdad_root, split="val", fold=args.fold,
            num_frames=args.num_frames, augment=False,
        )
        dna_test = MDADDataset(
            args.mdad_root, split="test", fold=args.fold,
            num_frames=args.num_frames, augment=False,
        )

    # [ARR] nosb5: filter seatbelt out of every target split before loaders are built.
    if _arr_drop_label is not None:
        dna_train = DropClassDataset(dna_train, _arr_drop_label)
        dna_val = DropClassDataset(dna_val, _arr_drop_label)
        dna_test = DropClassDataset(dna_test, _arr_drop_label)

    da_modes = ("dann", "cbdann", "cdan", "mcc", "bnm")

    # Reverse direction: source = current target_dataset, target = OBC.
    # For mode=target_only ("source-only reverse"): train on dna_train (D&A/DMD/3MDAD), eval on OBC.
    # For mode=dann: swap so dna_* = OBC (target) and source becomes D&A/DMD/3MDAD.
    _reverse_src_train = None
    if args.reverse_direction:
        # [ARR 2026-09-05] The two routes into a 0-label OBC target are ALTERNATIVES, never a
        # pair. With --target_dataset obc the branch at :558-573 has already set
        # dna_train/val/test to OBC train/val/test; falling through into this block would then
        #   (a) overwrite dna_test with obc_target_val (:630) -- selection and reporting on ONE
        #       split, which PREREGISTRATION Amendment 2 forbids for E2; and
        #   (b) set obc_train = _reverse_src_train (:651), which is the OBC train set the
        #       :558 branch built, making the run OBC -> OBC and silently discarding
        #       --source_manifest (read only in the else at :655).
        # Neither is caught by the label-leak harness: target labels are still removed on both
        # paths (_arr_zero_label is True either way), so the defect is WHICH DATASETS sit on
        # each side, not label hygiene. Refuse the combination here, loudly.
        assert args.target_dataset != "obc", (
            "[ARR] --reverse_direction with --target_dataset obc is REFUSED. These are two "
            "alternative routes into a 0-label OBC target, not a pair: combining them "
            "overwrites dna_test with OBC val and replaces the manifest source with OBC "
            "train. Use --target_dataset obc + --source_manifest (route A, E2 as approved "
            "2026-09-05), or --reverse_direction with --target_dataset <source> (the frozen "
            "route that produced Table 10). Not both.")
        # [ARR] E2: the 0-label reverse is now permitted. Anything strictly between 0 and 1
        # is still refused - this code does not define that regime.
        assert args.target_label_ratio == 1.0 or (
            args.target_label_ratio == 0.0 and args.mode == "dann"), \
            ("[ARR] --reverse_direction allows target_label_ratio 1.0 (frozen behaviour) or "
             "0.0 with --mode dann (E2). Got ratio=%s mode=%s."
             % (args.target_label_ratio, args.mode))
        assert args.mode in ("target_only", "dann"), \
            f"--reverse_direction supports only target_only or dann, got {args.mode}"
        obc_target_train = OBCCrossDataset(
            args.obc_metadata, split="train",
            num_frames=args.num_frames, augment=True,
            temporal_jitter=2, color_jitter=0.3,
        )
        obc_target_val = OBCCrossDataset(
            args.obc_metadata, split="val",
            num_frames=args.num_frames, augment=False,
        )
        if args.mode == "target_only":
            # Source-only reverse: train on D&A/DMD/3MDAD train (existing dna_train), eval on OBC.
            dna_val = obc_target_val
            dna_test = obc_target_val
            print(f"[Reverse target_only] Train on {args.target_dataset}, Eval on OBC val")
        else:  # dann
            # DANN reverse: swap. D&A/DMD/3MDAD becomes source, OBC becomes target (unlabeled).
            _reverse_src_train = dna_train
            dna_train = obc_target_train
            dna_val = obc_target_val
            dna_test = obc_target_val
            print(f"[Reverse dann] Source: {args.target_dataset}, Target: OBC")

    # SSDA / Label efficiency: wrap or subset target training data
    if args.target_label_ratio < 1.0:
        if args.mode in ("target_only", "finetune"):
            # Label efficiency: just use a subset of training data
            n_total = len(dna_train)
            n_labeled = max(1, int(n_total * args.target_label_ratio))
            rng = np.random.RandomState(42)
            subset_indices = rng.choice(n_total, n_labeled, replace=False).tolist()
            dna_train = Subset(dna_train, subset_indices)
            print(f"[LabelEff] Using {n_labeled}/{n_total} target samples "
                  f"({args.target_label_ratio*100:.0f}%)")
        # For DA modes, wrapping happens below

    ssda_mode = False  # will be set True if DA mode + target_label_ratio < 1.0

    if args.mode in da_modes:
        if args.reverse_direction:
            # Source = D&A/DMD/3MDAD (already built above)
            obc_train = _reverse_src_train
            obc_val = _reverse_src_train  # unused; harmless placeholder
        else:
            # Source: OBC, or [ARR] a frozen manifest over the Drive&Act memmap
            if args.source_manifest:
                from cross_dataset.memmap_dataset import MemmapVideoDataset as _MMV
                obc_train = build_manifest_source(
                    args.source_manifest, args.source_memmap,
                    args.source_memmap_metadata, args.num_frames,
                    augment=True, color_jitter=0.3)
                obc_val = _MMV(args.source_memmap, args.source_memmap_metadata,
                               fold=0, split=args.source_val_split,
                               num_frames=args.num_frames, augment=False, crop_size=224)
                print("[ARR-source] source-val = %s split of %s (%d clips)"
                      % (args.source_val_split, args.source_memmap_metadata, len(obc_val)))
            else:
                obc_train = OBCCrossDataset(
                    args.obc_metadata, split="train",
                    num_frames=args.num_frames, augment=True,
                    temporal_jitter=2, color_jitter=0.3,
                )
                obc_val = OBCCrossDataset(
                    args.obc_metadata, split="val",
                    num_frames=args.num_frames, augment=False,
                )

            # [ARR] nosb5 on the source side as well
        if _arr_drop_label is not None:
            obc_train = DropClassDataset(obc_train, _arr_drop_label)
            obc_val = DropClassDataset(obc_val, _arr_drop_label)

        # [ARR] E2 label discipline. Covers BOTH routes into a 0-label target - the new
        # --target_dataset obc branch and the now-permitted --reverse_direction path - so
        # neither can bypass it. Must run BEFORE LabeledSubsetWrapper, which wraps this.
        _arr_zero_label = (args.target_label_ratio <= 0.0
                           and (args.target_dataset == "obc" or args.reverse_direction))
        if _arr_zero_label:
            dna_train = ZeroLabelTarget(dna_train, num_classes)
        args._arr_zero_label_active = bool(_arr_zero_label)

        # SSDA: wrap target train to track labeled/unlabeled
        if args.target_label_ratio < 1.0:
            dna_train = LabeledSubsetWrapper(dna_train, args.target_label_ratio, seed=42)
            print(f"[SSDA] {dna_train.n_labeled}/{len(dna_train)} target samples labeled "
                  f"({args.target_label_ratio*100:.0f}%)")
            ssda_mode = True
        else:
            ssda_mode = False

    # DataLoaders
    # Loader-efficiency knobs (perf only, no effect on results): keep workers
    # alive across epochs and deepen the prefetch buffer to avoid GPU starving
    # between batches. Guarded so num_workers=0 stays valid.
    _persist = args.num_workers > 0
    _prefetch = 4 if args.num_workers > 0 else None
    train_loader = DataLoader(
        dna_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=_persist, prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        dna_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=_persist, prefetch_factor=_prefetch,
    )
    test_loader = DataLoader(
        dna_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=_persist, prefetch_factor=_prefetch,
    )

    if args.mode in da_modes:
        source_loader = DataLoader(
            obc_train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=_persist, prefetch_factor=_prefetch,
        )
        source_val_loader = DataLoader(
            obc_val, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=_persist, prefetch_factor=_prefetch,
        )

    # CB-DANN: precompute class weights from source label frequencies
    if args.mode == "cbdann" and args.cbdann_class_weight == "frequency":
        # Gather all source labels
        src_labels = np.array([obc_train[i][1] for i in range(len(obc_train))])
        class_counts = np.bincount(src_labels, minlength=num_classes).astype(np.float64)
        class_counts = np.maximum(class_counts, 1.0)  # avoid div-by-zero
        inv_freq = 1.0 / class_counts
        inv_freq = inv_freq / inv_freq.sum() * num_classes  # normalize so mean=1
        cbdann_class_weights = torch.tensor(inv_freq, dtype=torch.float32, device=device)
        print(f"[CB-DANN] frequency class weights: {cbdann_class_weights.cpu().numpy()}")

    # ===== Model =====
    if getattr(args, "backbone", "videomae") != "videomae":
        model = build_model_backbone(num_classes, args.mode, args.backbone,
                                     pretrained=not args.no_pretrain)
        print("[ARR-backbone] %s, %d classes" % (args.backbone, num_classes))
    else:
        model = build_model(num_classes, args.mode, args.pretrained, args.drop_path_rate,
                            model_name=args.model_name, no_pretrain=args.no_pretrain)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Params: {total_params/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")

    # ===== Optimizer =====
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=1
    )
    scaler = GradScaler("cuda", enabled=args.fp16)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    domain_criterion = nn.CrossEntropyLoss()

    best_val_bal_acc = 0.0
    patience_counter = 0
    start_epoch = 0

    # Resume from checkpoint
    if args.resume_dir:
        ckpt_path = os.path.join(result_dir, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_val_bal_acc = ckpt["best_val_bal_acc"]
            patience_counter = ckpt["patience_counter"]
            print(f"[Resume] Loaded checkpoint: epoch {ckpt['epoch']}, "
                  f"bestValBA={best_val_bal_acc:.4f}, patience={patience_counter}/{args.patience}")
            del ckpt
        elif os.path.exists(os.path.join(result_dir, "best_model.pt")):
            # Fallback: no checkpoint.pt but best_model.pt exists (from timed-out run)
            # Load best model weights, reset optimizer (no state saved), use --resume_epoch
            model.load_state_dict(torch.load(
                os.path.join(result_dir, "best_model.pt"), map_location=device, weights_only=False))
            if args.resume_epoch is not None:
                start_epoch = args.resume_epoch
            print(f"[Resume] Loaded best_model.pt (no checkpoint). "
                  f"Starting from epoch {start_epoch+1}, optimizer reset.")

    print(f"[Train] Starting from epoch {start_epoch+1} for {args.epochs} epochs, patience={args.patience}")

    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()   # [ARR] wall-clock per epoch
        model.train()
        train_loss = 0.0
        _arr_cls_loss = 0.0      # [ARR] classification component
        _arr_dom_loss = 0.0      # [ARR] domain component (unweighted, pre-alpha)
        _arr_nsteps = 0
        all_preds, all_labels = [], []

        if args.mode in da_modes:
            lambda_dann = get_dann_lambda(epoch, args.epochs)
            source_iter = iter(source_loader)

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"E{epoch+1}/{args.epochs}")
        ):
            # Unpack batch: SSDA wrapper returns 3 values, normal dataset returns 2
            if args.mode in da_modes and ssda_mode:
                target_imgs, target_labels, target_is_labeled = batch
                target_is_labeled = target_is_labeled.bool().to(device, non_blocking=True)
            else:
                target_imgs, target_labels = batch[:2]
                target_is_labeled = None
            target_imgs = target_imgs.to(device, non_blocking=True)
            target_labels = target_labels.to(device, non_blocking=True)

            if args.mode in da_modes:
                # Get source batch (cycle if exhausted)
                try:
                    source_imgs, source_labels = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    source_imgs, source_labels = next(source_iter)
                source_imgs = source_imgs.to(device, non_blocking=True)
                source_labels = source_labels.to(device, non_blocking=True)

                with autocast("cuda", enabled=args.fp16):
                    # Source forward
                    s_logits, s_domain, _ = model(source_imgs, lambda_dann)
                    s_cls_loss = criterion(s_logits, source_labels)
                    s_domain_labels = torch.zeros(s_logits.size(0), dtype=torch.long, device=device)

                    # Target forward (domain alignment uses ALL target samples)
                    t_logits, t_domain, _ = model(target_imgs, lambda_dann)
                    t_domain_labels = torch.ones(t_logits.size(0), dtype=torch.long, device=device)

                    # Target classification loss: depends on label availability
                    if target_is_labeled is not None:
                        # SSDA: only labeled target samples contribute to cls loss
                        if target_is_labeled.any():
                            t_cls_loss = criterion(
                                t_logits[target_is_labeled],
                                target_labels[target_is_labeled],
                            )
                        else:
                            t_cls_loss = torch.tensor(0.0, device=device)
                    else:
                        # Supervised DA (ratio=1.0): all target samples labeled
                        t_cls_loss = criterion(t_logits, target_labels)

                    if args.mode == "cbdann":
                        # Class-Balanced domain loss: weight each sample by its class weight
                        if args.cbdann_class_weight == "frequency":
                            # Source: use ground-truth labels for weight lookup
                            s_weights = cbdann_class_weights[source_labels]
                            # Target: use predicted class for weight lookup
                            t_pred_classes = t_logits.argmax(dim=-1).detach()
                            t_weights = cbdann_class_weights[t_pred_classes]
                        elif args.cbdann_class_weight == "entropy":
                            # Entropy-based: higher prediction entropy = harder = more weight
                            with torch.no_grad():
                                s_probs = torch.softmax(s_logits, dim=1)
                                s_entropy = -(s_probs * torch.log(s_probs + 1e-8)).sum(dim=1)
                                t_probs = torch.softmax(t_logits, dim=1)
                                t_entropy = -(t_probs * torch.log(t_probs + 1e-8)).sum(dim=1)
                                max_entropy = np.log(num_classes)
                                # Normalize to [0.5, 1.5] range so weights are moderate
                                s_weights = 0.5 + s_entropy / max_entropy
                                t_weights = 0.5 + t_entropy / max_entropy
                        else:  # uniform (same as standard DANN)
                            s_weights = torch.ones(s_logits.size(0), device=device)
                            t_weights = torch.ones(t_logits.size(0), device=device)

                        # Per-sample domain loss with class-based weighting
                        s_domain_loss_per = F.cross_entropy(s_domain, s_domain_labels, reduction='none')
                        t_domain_loss_per = F.cross_entropy(t_domain, t_domain_labels, reduction='none')
                        s_domain_loss = (s_domain_loss_per * s_weights).mean()
                        t_domain_loss = (t_domain_loss_per * t_weights).mean()
                    else:
                        s_domain_loss = domain_criterion(s_domain, s_domain_labels)
                        t_domain_loss = domain_criterion(t_domain, t_domain_labels)

                    # Combined loss
                    if args.target_label_ratio <= 0.0:
                        # UDA: source classification only
                        cls_loss = s_cls_loss
                    elif target_is_labeled is not None and not target_is_labeled.any():
                        # SSDA batch with no labeled target samples
                        cls_loss = s_cls_loss
                    else:
                        cls_loss = 0.5 * s_cls_loss + 0.5 * t_cls_loss
                    domain_loss = 0.5 * (s_domain_loss + t_domain_loss)
                    loss = cls_loss + args.alpha_dann * domain_loss

                    # Auxiliary losses
                    if args.mode == "mcc":
                        loss = loss + args.alpha_mcc * mcc_loss(t_logits, temperature=args.mcc_temperature)
                    elif args.mode == "bnm":
                        loss = loss + args.alpha_bnm * bnm_loss(t_logits)

                preds = t_logits.argmax(dim=-1).cpu().numpy()
            else:
                # target_only or finetune
                with autocast("cuda", enabled=args.fp16):
                    outputs = model(pixel_values=target_imgs)
                    loss = criterion(outputs.logits, target_labels)

                preds = outputs.logits.argmax(dim=-1).cpu().numpy()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            # [ARR] component capture. cls_loss/domain_loss are DA-mode locals; in
            # target_only/finetune only the single supervised loss exists.
            _arr_nsteps += 1
            if args.mode in da_modes:
                _arr_cls_loss += float(cls_loss.detach())
                _arr_dom_loss += float(domain_loss.detach())
            else:
                _arr_cls_loss += float(loss.detach())
            all_preds.extend(preds)
            all_labels.extend(target_labels.cpu().numpy())

        scheduler.step(epoch)
        # [ARR] with a 0-label target the collected labels are sentinels, so a target-train
        # balanced accuracy is undefined. Report it absent rather than compute nonsense.
        train_bal_acc = (None if getattr(args, "_arr_zero_label_active", False)
                         else balanced_accuracy_score(all_labels, all_preds))
        avg_loss = train_loss / len(train_loader)

        # ===== Validation (on D&A val) =====
        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                with autocast("cuda", enabled=args.fp16):
                    if args.mode in da_modes:
                        logits, _, _ = model(imgs, 0.0)
                    else:
                        logits = model(pixel_values=imgs).logits
                val_preds.extend(logits.argmax(-1).cpu().numpy())
                val_labels_list.extend(labels.numpy())

        val_acc = accuracy_score(val_labels_list, val_preds)
        val_bal_acc = balanced_accuracy_score(val_labels_list, val_preds)

        # Model-selection metric. Default = TARGET val BA (paper protocol, kept
        # for DANN-UDA / CDAN / MCC / BNM so table comparisons stay consistent).
        # --select_by_source_val selects by OBC SOURCE val instead: required for
        # the SFDA cold-start source-only build so the model never touches the
        # target (clean source-free).
        sel_bal_acc = val_bal_acc
        use_src_sel = getattr(args, "select_by_source_val", False) and args.mode in da_modes
        if use_src_sel:
            sv_preds, sv_labels = [], []
            with torch.no_grad():
                for imgs, labels in source_val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    with autocast("cuda", enabled=args.fp16):
                        logits, _, _ = model(imgs, 0.0)
                    sv_preds.extend(logits.argmax(-1).cpu().numpy())
                    sv_labels.extend(labels.numpy())
            sel_bal_acc = balanced_accuracy_score(sv_labels, sv_preds)

        _tba = "n/a" if train_bal_acc is None else ("%.4f" % train_bal_acc)
        print(f"  Loss={avg_loss:.4f} TrainBA={_tba} | "
              f"ValAcc={val_acc:.4f} ValBA={val_bal_acc:.4f}"
              + (f" SrcValBA={sel_bal_acc:.4f}" if use_src_sel else "")
              + (f" λ={lambda_dann:.3f}" if args.mode in da_modes else ""))

        if sel_bal_acc > best_val_bal_acc:
            best_val_bal_acc = sel_bal_acc
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pt"))
            patience_counter = 0
            print(f"  -> Best model saved ({'SrcValBA' if use_src_sel else 'ValBA'}={sel_bal_acc:.4f})")
        else:
            patience_counter += 1

        # Save checkpoint for resume
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_bal_acc": best_val_bal_acc,
            "patience_counter": patience_counter,
        }, os.path.join(result_dir, "checkpoint.pt"))

        # ===== [ARR] instrumentation. Runs AFTER best_model/checkpoint so their
        # behaviour is bit-identical to the frozen trainer. =====
        if args.instrument:
            _t0 = time.time()
            torch.save(model.state_dict(),
                       os.path.join(result_dir, "epoch_%03d.pt" % (epoch + 1)))
            _src_ba, _ = (eval_split(model, source_val_loader, device, args.mode,
                                     da_modes, args.fp16)
                          if args.mode in da_modes else (None, None))
            _tgt_ba, _tgt_ent = eval_split(model, val_loader, device, args.mode,
                                           da_modes, args.fp16, want_entropy=True)
            _rec = {
                "epoch": epoch + 1,
                "train_loss_total": avg_loss,
                "train_loss_cls": _arr_cls_loss / max(_arr_nsteps, 1),
                "train_loss_domain": (_arr_dom_loss / max(_arr_nsteps, 1)
                                      if args.mode in da_modes else None),
                "alpha_dann": args.alpha_dann if args.mode in da_modes else None,
                "lambda_dann": float(lambda_dann) if args.mode in da_modes else None,
                "train_bal_acc": (None if train_bal_acc is None else float(train_bal_acc)),
                "zero_label_target": bool(getattr(args, "_arr_zero_label_active", False)),
                "source_val_bal_acc": (float(_src_ba) if _src_ba is not None else None),
                "target_val_bal_acc": float(_tgt_ba),
                "target_val_acc": float(val_acc),
                "target_mean_entropy": _tgt_ent,
                "entropy_ceiling_ln_C": float(np.log(num_classes)),
                "selection_metric": float(sel_bal_acc),
                "best_so_far": float(best_val_bal_acc),
                "patience_counter": int(patience_counter),
                "iters": int(_arr_nsteps),
                "epoch_seconds": float(time.time() - epoch_t0),
                "instrument_seconds": float(time.time() - _t0),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "checkpoint": "epoch_%03d.pt" % (epoch + 1),
            }
            with open(os.path.join(result_dir, "metrics.jsonl"), "a") as _f:
                _f.write(json.dumps(_rec) + "\n")
            print("  [ARR] e%d cls=%.4f dom=%s srcValBA=%s tgtValBA=%.4f ent=%.4f %.1fs"
                  % (epoch + 1, _rec["train_loss_cls"],
                     ("%.4f" % _rec["train_loss_domain"]) if _rec["train_loss_domain"] is not None else "-",
                     ("%.4f" % _rec["source_val_bal_acc"]) if _rec["source_val_bal_acc"] is not None else "-",
                     _rec["target_val_bal_acc"], _rec["target_mean_entropy"],
                     _rec["epoch_seconds"]))

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # ===== Test =====
    print("\n[Test] Loading best model...")
    model.load_state_dict(
        torch.load(os.path.join(result_dir, "best_model.pt"), weights_only=False)
    )
    model.eval()

    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda", enabled=args.fp16):
                if args.mode in da_modes:
                    logits, _, _ = model(imgs, 0.0)
                else:
                    logits = model(pixel_values=imgs).logits
            test_preds.extend(logits.argmax(-1).cpu().numpy())
            test_labels_list.extend(labels.numpy())

    test_acc = accuracy_score(test_labels_list, test_preds)
    test_bal_acc = balanced_accuracy_score(test_labels_list, test_preds)

    print(f"\n{'='*60}")
    print(f"[RESULT] Mode={args.mode} Fold={args.fold}")
    print(f"  Test Accuracy:          {test_acc:.4f}")
    print(f"  Test Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"{'='*60}")

    # Save results FIRST (before classification_report which can crash on missing classes)
    results = {
        "mode": args.mode,
        "fold": args.fold,
        "target_dataset": args.target_dataset,
        "test_accuracy": test_acc,
        "test_balanced_accuracy": test_bal_acc,
        "best_val_balanced_accuracy": best_val_bal_acc,
        "epochs_trained": epoch + 1,
    }
    with open(os.path.join(result_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    report = classification_report(
        test_labels_list, test_preds,
        labels=list(range(len(class_names))),
        target_names=class_names, digits=4, zero_division=0,
    )
    print(report)

    with open(os.path.join(result_dir, "classification_report.txt"), "w") as f:
        f.write(f"Mode: {args.mode}, Fold: {args.fold}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Balanced Accuracy: {test_bal_acc:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(test_labels_list, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.title(f"Confusion Matrix — {args.mode} (fold {args.fold})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    elapsed = (time.time() - start_time) / 60
    print(f"\n[Done] Total time: {elapsed:.1f} min. Results: {result_dir}")


def eval_only(args):
    """Load best_model.pt from an existing result_dir and run test only."""
    import time as _time
    start_time = _time.time()
    if getattr(args, "seed", None) is not None:
        set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert args.result_dir, "--result_dir is required for --eval_only"
    result_dir = args.result_dir
    model_path = os.path.join(result_dir, "best_model.pt")
    assert os.path.exists(model_path), f"best_model.pt not found in {result_dir}"

    num_classes = get_num_unified_classes()
    class_names = get_unified_class_names()
    da_modes = ("dann", "cbdann", "cdan", "mcc", "bnm")
    print(f"[Eval-Only] Mode={args.mode}, Fold={args.fold}, Dir={result_dir}")

    # Dataset
    if args.target_dataset == "dmd":
        dna_test = DMDDataset(
            args.dmd_root, split="test", fold=args.fold,
            num_frames=args.num_frames, augment=False,
        )
    elif args.target_dataset == "3mdad":
        dna_test = MDADDataset(
            args.mdad_root, split="test", fold=args.fold,
            num_frames=args.num_frames, augment=False,
        )
    else:
        dna_test = DriveAndActDataset(
            args.dna_root, split="test", fold=args.fold,
            num_frames=args.num_frames, augment=False,
        )
    test_loader = DataLoader(
        dna_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = build_model(num_classes, args.mode, args.pretrained, args.drop_path_rate,
                        model_name=args.model_name, no_pretrain=args.no_pretrain)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model = model.to(device)
    model.eval()

    # Test
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda", enabled=args.fp16):
                if args.mode in da_modes:
                    logits, _, _ = model(imgs, 0.0)
                else:
                    logits = model(pixel_values=imgs).logits
            test_preds.extend(logits.argmax(-1).cpu().numpy())
            test_labels_list.extend(labels.numpy())

    test_acc = accuracy_score(test_labels_list, test_preds)
    test_bal_acc = balanced_accuracy_score(test_labels_list, test_preds)

    print(f"\n{'='*60}")
    print(f"[RESULT] Mode={args.mode} Fold={args.fold}")
    print(f"  Test Accuracy:          {test_acc:.4f}")
    print(f"  Test Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"{'='*60}")

    # Save results FIRST (before classification_report which can crash on missing classes)
    # Determine if this is a cross-target eval (trained on X, eval on Y)
    # by comparing the eval target with the target encoded in the dir name
    dir_name = os.path.basename(result_dir)
    trained_target = "driveandact"
    if "_dmd_" in dir_name:
        trained_target = "dmd"
    elif "_3mdad_" in dir_name:
        trained_target = "3mdad"
    is_cross_target = (args.target_dataset != trained_target)
    # Always include eval target AND fold so repeated eval_only runs into a shared
    # result_dir (e.g. one source-only model evaluated on 3 targets x 3 folds) never
    # collide and never overwrite the dir's own training results.json.
    eval_suffix = f"_eval_{args.target_dataset}_fold{args.fold}"

    results = {
        "mode": args.mode,
        "fold": args.fold,
        "target_dataset": args.target_dataset,
        "eval_target": args.target_dataset,
        "test_accuracy": test_acc,
        "test_balanced_accuracy": test_bal_acc,
    }
    with open(os.path.join(result_dir, f"results{eval_suffix}.json"), "w") as f:
        json.dump(results, f, indent=2)

    report = classification_report(
        test_labels_list, test_preds,
        labels=list(range(len(class_names))),
        target_names=class_names, digits=4, zero_division=0,
    )
    print(report)

    with open(os.path.join(result_dir, f"classification_report{eval_suffix}.txt"), "w") as f:
        f.write(f"Mode: {args.mode}, Fold: {args.fold}, EvalTarget: {args.target_dataset}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Balanced Accuracy: {test_bal_acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(test_labels_list, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.title(f"Confusion Matrix — {args.mode} (fold {args.fold}) → {args.target_dataset}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"confusion_matrix{eval_suffix}.png"), dpi=150)
    plt.close()

    elapsed = (_time.time() - start_time) / 60
    print(f"\n[Done] Eval time: {elapsed:.1f} min. Results: {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["target_only", "finetune", "dann", "cbdann", "cdan", "mcc", "bnm"])
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed (init + data order). cuDNN nondeterminism not forced.")
    parser.add_argument("--select_by_source_val", action="store_true", default=False,
                        help="Select best_model by OBC SOURCE val BA instead of target val "
                             "(clean source-free; for the SFDA cold-start source-only build). "
                             "Default off = target-val (paper protocol).")
    # ===== [ARR] access_rebuttal_runs additions. All defaults = frozen behaviour. =====
    parser.add_argument("--instrument", action="store_true", default=False,
                        help="[ARR] Per-epoch epoch_###.pt (model only) + metrics.jsonl. "
                             "Does NOT change best_model.pt / checkpoint.pt behaviour.")
    parser.add_argument("--source_manifest", type=str, default=None,
                        help="[ARR] CSV of Drive&Act memmap rows to use as the SOURCE set "
                             "(e.g. broad573/narrow573). Replaces the OBC source.")
    parser.add_argument("--source_memmap", type=str, default=None,
                        help="[ARR] frames.npy backing --source_manifest.")
    parser.add_argument("--source_memmap_metadata", type=str, default=None,
                        help="[ARR] metadata.json backing --source_manifest.")
    parser.add_argument("--source_val_split", type=str, default="val",
                        help="[ARR] fold-0 split of the source memmap used as source-val "
                             "when --source_manifest is set.")
    parser.add_argument("--backbone", type=str, default="videomae",
                        choices=["videomae", "r2plus1d_18", "resnet50_tsn"],
                        help="[ARR] E3 backbone ladder. videomae = the existing path, "
                             "byte-identical behaviour. The two CNNs consume the SAME frames "
                             "the existing 51->16 rule produces (obc_dataset.py:138-142); "
                             "nothing re-samples. Per-backbone normalization and spatial size "
                             "are applied inside the model (cross_dataset/backbones.py).")
    parser.add_argument("--label_space", type=str, default="unified6",
                        choices=["unified6", "nosb5"],
                        help="[ARR] unified6 = frozen 6-class taxonomy. nosb5 = drop seatbelt "
                             "(unified label 5) from every split and build a 5-way head.")
    parser.add_argument("--run_tag", type=str, default="",
                        help="Optional tag appended to the result-dir name for provenance (e.g. srconly).")

    # Data
    parser.add_argument("--target_dataset", type=str, default="driveandact",
                        choices=["driveandact", "dmd", "3mdad", "obc"],
                        help="Target dataset: driveandact, dmd, or 3mdad")
    parser.add_argument("--dna_root", type=str,
                        default="<DATA_ROOT>/DriveAndAct")
    parser.add_argument("--dmd_root", type=str,
                        default="<DATA_ROOT>/DMD/distraction/dmd")
    parser.add_argument("--mdad_root", type=str,
                        default="<DATA_ROOT>/MDAD/MDAD")
    parser.add_argument("--obc_metadata", type=str,
                        default="features/keyframe_video_metadata_dualcam.pkl")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)

    # Model
    parser.add_argument("--pretrained", type=str, default=None,
                        help="OBC pretrained model path (for finetune mode)")
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--model_name", type=str,
                        default="MCG-NJU/videomae-base-finetuned-kinetics",
                        help="HF backbone (e.g. MCG-NJU/videomae-small-finetuned-kinetics, "
                             "MCG-NJU/videomae-large-finetuned-kinetics)")
    parser.add_argument("--no_pretrain", action="store_true", default=False,
                        help="Random-init backbone (skip from_pretrained). For ablation.")
    parser.add_argument("--reverse_direction", action="store_true", default=False,
                        help="Reverse: source = --target_dataset (D&A/DMD/3MDAD), target = OBC. "
                             "Supports target_only (= source-only reverse) and dann.")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--T_0", type=int, default=10)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--memmap_path", type=str, default=None,
                        help="Pre-decoded uint8 memmap (.npy) for target dataset. "
                             "If set, bypasses decord on-the-fly decoding. "
                             "Also requires --memmap_metadata.")
    parser.add_argument("--memmap_metadata", type=str, default=None,
                        help="Metadata json companion to --memmap_path.")
    parser.add_argument("--memmap_crop", type=int, default=0,
                        help="Tier-2: crop memmap frames to this size at load "
                             "(RandomCrop train / CenterCrop val-test). 0 = no crop. "
                             "Use 224 with 256-resolution memmaps to restore spatial-crop aug.")

    # DANN / CDAN / CB-DANN
    parser.add_argument("--alpha_dann", type=float, default=0.5,
                        help="DANN/CDAN domain loss weight")

    # CB-DANN
    parser.add_argument("--cbdann_class_weight", type=str, default="frequency",
                        choices=["frequency", "entropy", "uniform"],
                        help="CB-DANN weighting strategy: frequency (inverse class freq), "
                             "entropy (prediction entropy), uniform (same as standard DANN)")

    # MCC
    parser.add_argument("--alpha_mcc", type=float, default=1.0,
                        help="MCC loss weight (used with --mode mcc)")
    parser.add_argument("--mcc_temperature", type=float, default=2.5,
                        help="MCC softmax temperature")

    # BNM
    parser.add_argument("--alpha_bnm", type=float, default=0.1,
                        help="BNM loss weight (used with --mode bnm)")

    # SSDA / Label efficiency
    parser.add_argument("--target_label_ratio", type=float, default=1.0,
                        help="Fraction of target labels to use. "
                             "1.0=all (default, supervised DA), 0.0=none (UDA), "
                             "0.05/0.1/0.25=SSDA. For target_only/finetune, "
                             "subsets training data.")

    # Resume: load checkpoint.pt from existing result_dir and continue training
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Existing result directory to resume training from checkpoint.pt")
    parser.add_argument("--resume_epoch", type=int, default=None,
                        help="Override start epoch when resuming without checkpoint.pt (fallback mode)")

    # Eval-only: load best_model.pt from existing result_dir, skip training
    parser.add_argument("--eval_only", action="store_true", default=False,
                        help="Skip training; load best_model.pt from --result_dir and run test only")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="Existing result directory (required for --eval_only)")

    args = parser.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train(args)

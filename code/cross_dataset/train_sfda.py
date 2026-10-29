"""Source-Free Domain Adaptation for cross-dataset driver behavior recognition.

Implements:
  - shot:        SHOT (Liang et al., ICML 2020) — pseudo-label + info max + self-training
  - cleanadapt:  CleanAdapt (Dasgupta et al., Pattern Recognition 2025) — noisy label filtering
  - csfda:       C-SFDA (CVPR 2023) — curriculum-based pseudo-label selection extending SHOT

All methods use a source-trained model and adapt on target data only (no source access).

Stage 1: Train source model (use existing DANN/FT checkpoint, or train source-only)
Stage 2: Adapt on target with self-training

Usage:
  python -m cross_dataset.train_sfda \
      --method shot \
      --source_model results/XXXX_cross_dann_fold0/best_model.pt \
      --source_model_type dann \
      --dna_root <DATA_ROOT>/DriveAndAct \
      --fold 0

  python -m cross_dataset.train_sfda \
      --method cleanadapt \
      --source_model results/XXXX_cross_dann_fold0/best_model.pt \
      --source_model_type dann \
      --dna_root <DATA_ROOT>/DriveAndAct \
      --fold 0
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
from torch.utils.data import DataLoader
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
)
from cross_dataset.driveandact_dataset import DriveAndActDataset
from cross_dataset.dmd_dataset import DMDDataset
from cross_dataset.mdad_dataset import MDADDataset
from cross_dataset.train_cross import VideoMAEDANN


def load_source_model(model_path, model_type, num_classes, device):
    """Load a source-trained model checkpoint."""
    if model_type in ("dann", "cdan", "mcc", "bnm"):
        model = VideoMAEDANN(num_classes, use_cdan=(model_type == "cdan"))
    else:
        config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        config.num_labels = num_classes
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            config=config,
            ignore_mismatched_sizes=True,
        )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"[Model] Loaded source model ({model_type}) from {model_path}")
    return model


def get_features_and_logits(model, loader, device, model_type, fp16=True):
    """Extract features and logits from a model on a dataset."""
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda", enabled=fp16):
                if model_type in ("dann", "cdan", "mcc", "bnm"):
                    logits, _, _ = model(imgs, 0.0)
                else:
                    logits = model(pixel_values=imgs).logits
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def generate_pseudo_labels(logits, threshold=0.0):
    """Generate pseudo labels from model predictions."""
    probs = F.softmax(logits, dim=1)
    confidence, pseudo_labels = probs.max(dim=1)
    if threshold > 0:
        mask = confidence >= threshold
        return pseudo_labels, confidence, mask
    return pseudo_labels, confidence, torch.ones(len(logits), dtype=torch.bool)


# ========== Resume helpers ==========

def save_checkpoint(result_dir, epoch, model, optimizer, scheduler, scaler,
                    best_val_bal_acc, patience_counter, **extra):
    """Save checkpoint for resume."""
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_bal_acc": best_val_bal_acc,
        "patience_counter": patience_counter,
    }
    ckpt.update(extra)
    torch.save(ckpt, os.path.join(result_dir, "checkpoint.pt"))


def load_checkpoint(result_dir, device, model, optimizer, scheduler, scaler,
                    resume_epoch=None):
    """Load checkpoint if available. Returns (start_epoch, best_val_bal_acc, patience_counter, extra)."""
    ckpt_path = os.path.join(result_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val_bal_acc"]
        patience = ckpt["patience_counter"]
        print(f"[Resume] Loaded checkpoint: epoch {ckpt['epoch']}, "
              f"bestValBA={best_val:.4f}, patience={patience}")
        return start_epoch, best_val, patience, ckpt
    elif os.path.exists(os.path.join(result_dir, "best_model.pt")):
        model.load_state_dict(torch.load(
            os.path.join(result_dir, "best_model.pt"), map_location=device, weights_only=False))
        start_epoch = resume_epoch if resume_epoch is not None else 0
        print(f"[Resume] Loaded best_model.pt (no checkpoint). Starting from epoch {start_epoch+1}.")
        return start_epoch, 0.0, 0, {}
    return 0, 0.0, 0, {}


# ========== SHOT ==========

def shot_entropy_loss(logits):
    """Information maximization loss for SHOT.
    Combines: (1) per-sample entropy minimization + (2) batch diversity maximization.
    """
    softmax_out = F.softmax(logits, dim=1)

    # Per-sample entropy minimization (certainty)
    entropy = -(softmax_out * torch.log(softmax_out + 1e-8)).sum(dim=1).mean()

    # Batch-level diversity (maximize entropy of mean prediction)
    mean_softmax = softmax_out.mean(dim=0)
    diversity = (mean_softmax * torch.log(mean_softmax + 1e-8)).sum()

    return entropy + diversity


def train_shot(model, train_loader, val_loader, test_loader, args, device,
               result_dir, class_names, num_classes):
    """SHOT adaptation: freeze classifier, adapt backbone with entropy + pseudo-labels."""

    # Freeze the classifier head, only adapt backbone + fc_norm
    if hasattr(model, 'videomae'):
        # DANN model
        backbone = model.videomae
        classifier = backbone.classifier
    else:
        backbone = model
        classifier = model.classifier

    for param in classifier.parameters():
        param.requires_grad = False

    # Also freeze domain classifier if DANN
    if hasattr(model, 'domain_classifier'):
        for param in model.domain_classifier.parameters():
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=args.fp16)

    is_dann = hasattr(model, 'domain_classifier')
    best_val_bal_acc = 0.0
    patience_counter = 0
    start_epoch = 0

    if args.resume_dir:
        start_epoch, best_val_bal_acc, patience_counter, _ = load_checkpoint(
            result_dir, device, model, optimizer, scheduler, scaler, args.resume_epoch)

    print(f"[SHOT] Adapting from epoch {start_epoch+1} for {args.epochs} epochs (classifier frozen)")
    print(f"[SHOT] Trainable params: {sum(p.numel() for p in trainable_params)/1e6:.2f}M")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Keep classifier in eval mode (frozen)
        classifier.eval()
        train_loss = 0.0
        all_preds, all_labels = [], []

        for imgs, labels in tqdm(train_loader, desc=f"E{epoch+1}/{args.epochs}"):
            imgs = imgs.to(device, non_blocking=True)

            with autocast("cuda", enabled=args.fp16):
                if is_dann:
                    logits, _, _ = model(imgs, 0.0)
                else:
                    logits = model(pixel_values=imgs).logits

                # SHOT loss: entropy minimization + diversity
                loss = args.alpha_ent * shot_entropy_loss(logits)

                # Pseudo-label cross-entropy (optional, with confidence threshold)
                if args.shot_pseudo_weight > 0:
                    pseudo_labels, confidence, mask = generate_pseudo_labels(
                        logits.detach(), threshold=args.shot_threshold
                    )
                    if mask.sum() > 0:
                        pl_loss = F.cross_entropy(logits[mask], pseudo_labels[mask].to(device))
                        loss = loss + args.shot_pseudo_weight * pl_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        scheduler.step()

        train_bal_acc = balanced_accuracy_score(all_labels, all_preds)
        avg_loss = train_loss / len(train_loader)

        # Validation
        val_bal_acc = evaluate(model, val_loader, device, is_dann, args.fp16)
        print(f"  Loss={avg_loss:.4f} TrainBA={train_bal_acc:.4f} | ValBA={val_bal_acc:.4f}")

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pt"))
            patience_counter = 0
            print(f"  -> Best model saved (ValBA={val_bal_acc:.4f})")
        else:
            patience_counter += 1

        save_checkpoint(result_dir, epoch, model, optimizer, scheduler, scaler,
                        best_val_bal_acc, patience_counter)

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_val_bal_acc


# ========== CleanAdapt ==========

def train_cleanadapt(model, train_loader, val_loader, test_loader, args, device,
                     result_dir, class_names, num_classes):
    """CleanAdapt: filter noisy pseudo-labels using small-loss trick + EMA teacher."""

    is_dann = hasattr(model, 'domain_classifier')

    # Create EMA teacher
    import copy
    teacher = copy.deepcopy(model)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=args.fp16)

    best_val_bal_acc = 0.0
    patience_counter = 0
    start_epoch = 0

    if args.resume_dir:
        start_epoch, best_val_bal_acc, patience_counter, ckpt = load_checkpoint(
            result_dir, device, model, optimizer, scheduler, scaler, args.resume_epoch)
        # Refresh teacher from current model state
        import copy as _copy
        teacher = _copy.deepcopy(model)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    print(f"[CleanAdapt] Adapting from epoch {start_epoch+1} for {args.epochs} epochs")
    print(f"[CleanAdapt] Clean ratio: {args.clean_ratio}, EMA decay: {args.ema_decay}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        # Curriculum: increase clean ratio over epochs
        curr_ratio = min(args.clean_ratio + 0.1 * epoch / args.epochs, 0.9)

        for imgs, labels in tqdm(train_loader, desc=f"E{epoch+1}/{args.epochs}"):
            imgs = imgs.to(device, non_blocking=True)

            with autocast("cuda", enabled=args.fp16):
                # Teacher generates pseudo-labels
                with torch.no_grad():
                    if is_dann:
                        t_logits, _, _ = teacher(imgs, 0.0)
                    else:
                        t_logits = teacher(pixel_values=imgs).logits
                    pseudo_labels = t_logits.argmax(dim=1)

                # Student forward
                if is_dann:
                    s_logits, _, _ = model(imgs, 0.0)
                else:
                    s_logits = model(pixel_values=imgs).logits

                # Per-sample CE loss
                per_sample_loss = F.cross_entropy(s_logits, pseudo_labels, reduction='none')

                # Small-loss trick: select clean samples
                num_clean = max(int(curr_ratio * len(per_sample_loss)), 1)
                _, clean_indices = per_sample_loss.topk(num_clean, largest=False)

                loss = per_sample_loss[clean_indices].mean()

                # Entropy regularization
                loss = loss + args.alpha_ent * shot_entropy_loss(s_logits)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA teacher
            with torch.no_grad():
                for t_param, s_param in zip(teacher.parameters(), model.parameters()):
                    t_param.data.mul_(args.ema_decay).add_(s_param.data, alpha=1.0 - args.ema_decay)

            train_loss += loss.item()
            preds = s_logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        scheduler.step()

        train_bal_acc = balanced_accuracy_score(all_labels, all_preds)
        avg_loss = train_loss / len(train_loader)

        # Validation
        val_bal_acc = evaluate(model, val_loader, device, is_dann, args.fp16)
        print(f"  Loss={avg_loss:.4f} TrainBA={train_bal_acc:.4f} | ValBA={val_bal_acc:.4f}")

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pt"))
            patience_counter = 0
            print(f"  -> Best model saved (ValBA={val_bal_acc:.4f})")
        else:
            patience_counter += 1

        save_checkpoint(result_dir, epoch, model, optimizer, scheduler, scaler,
                        best_val_bal_acc, patience_counter)

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_val_bal_acc


# ========== C-SFDA ==========

def compute_reliability_scores(logits):
    """Compute confidence (max softmax) and uncertainty (entropy) for each sample."""
    probs = F.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return confidence, entropy


def get_curriculum_stage(epoch, total_epochs):
    """Return curriculum stage (1 or 2) and confidence percentile threshold.

    Stage 1 (first half): train on high-confidence samples only.
    Stage 2 (second half): gradually include lower-confidence samples.
    """
    midpoint = total_epochs // 2
    if epoch < midpoint:
        stage = 1
    else:
        stage = 2
    return stage


def get_sample_masks(confidence, conf_percentile):
    """Split batch into reliable/unreliable based on confidence percentile threshold."""
    if len(confidence) == 0:
        return torch.zeros(0, dtype=torch.bool, device=confidence.device)
    threshold = torch.quantile(confidence, 1.0 - conf_percentile / 100.0)
    reliable_mask = confidence >= threshold
    return reliable_mask


def compute_class_centroids(probs, pseudo_labels, num_classes):
    """Compute mean probability vector per class from reliable samples."""
    centroids = torch.zeros(num_classes, probs.shape[1], device=probs.device)
    counts = torch.zeros(num_classes, device=probs.device)
    for c in range(num_classes):
        mask = pseudo_labels == c
        if mask.sum() > 0:
            centroids[c] = probs[mask].mean(dim=0)
            counts[c] = mask.sum()
    # For classes with no reliable samples, use uniform distribution
    empty = counts == 0
    if empty.any():
        centroids[empty] = 1.0 / num_classes
    return centroids


def train_csfda(model, train_loader, val_loader, test_loader, args, device,
                result_dir, class_names, num_classes):
    """C-SFDA: curriculum source-free DA extending SHOT with reliable/unreliable split."""

    # Freeze the classifier head, only adapt backbone (source-free, same as SHOT)
    if hasattr(model, 'videomae'):
        # DANN model
        backbone = model.videomae
        classifier = backbone.classifier
    else:
        backbone = model
        classifier = model.classifier

    for param in classifier.parameters():
        param.requires_grad = False

    # Also freeze domain classifier if DANN
    if hasattr(model, 'domain_classifier'):
        for param in model.domain_classifier.parameters():
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=args.fp16)

    is_dann = hasattr(model, 'domain_classifier')
    best_val_bal_acc = 0.0
    patience_counter = 0
    start_epoch = 0

    if args.resume_dir:
        start_epoch, best_val_bal_acc, patience_counter, _ = load_checkpoint(
            result_dir, device, model, optimizer, scheduler, scaler, args.resume_epoch)

    print(f"[C-SFDA] Adapting from epoch {start_epoch+1} for {args.epochs} epochs (classifier frozen)")
    print(f"[C-SFDA] Trainable params: {sum(p.numel() for p in trainable_params)/1e6:.2f}M")
    print(f"[C-SFDA] CE weight={args.csfda_ce_weight}, MSE weight={args.csfda_mse_weight}")
    print(f"[C-SFDA] Conf percentile: {args.conf_percentile_start}% -> {args.conf_percentile_end}%")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Keep classifier in eval mode (frozen)
        classifier.eval()
        train_loss = 0.0
        all_preds, all_labels = [], []

        stage = get_curriculum_stage(epoch, args.epochs)

        # Linearly interpolate confidence percentile over epochs
        progress = epoch / max(args.epochs - 1, 1)
        conf_pct = args.conf_percentile_start + \
            (args.conf_percentile_end - args.conf_percentile_start) * progress

        for imgs, labels in tqdm(train_loader, desc=f"E{epoch+1}/{args.epochs} S{stage}"):
            imgs = imgs.to(device, non_blocking=True)

            with autocast("cuda", enabled=args.fp16):
                if is_dann:
                    logits, _, _ = model(imgs, 0.0)
                else:
                    logits = model(pixel_values=imgs).logits

                # Reliability scores
                confidence, uncertainty = compute_reliability_scores(logits.detach())
                reliable_mask = get_sample_masks(confidence, conf_pct)
                unreliable_mask = ~reliable_mask
                pseudo_labels = logits.detach().argmax(dim=1)

                # Loss 1: Entropy minimization (all samples, same as SHOT)
                loss = args.alpha_ent * shot_entropy_loss(logits)

                # Loss 2: CE on reliable samples only
                if reliable_mask.sum() > 0:
                    loss = loss + args.csfda_ce_weight * F.cross_entropy(
                        logits[reliable_mask], pseudo_labels[reliable_mask]
                    )

                # Loss 3 (Stage 2 only): MSE guidance for unreliable samples
                if stage == 2 and unreliable_mask.sum() > 0 and reliable_mask.sum() > 0:
                    with torch.no_grad():
                        reliable_probs = F.softmax(logits[reliable_mask].detach(), dim=1)
                        centroid_probs = compute_class_centroids(
                            reliable_probs, pseudo_labels[reliable_mask], num_classes
                        )
                        # Target for unreliable: softmax centroid of nearest (pseudo) class
                        targets = centroid_probs[pseudo_labels[unreliable_mask]]

                    unreliable_probs = F.softmax(logits[unreliable_mask], dim=1)
                    loss = loss + args.csfda_mse_weight * F.mse_loss(
                        unreliable_probs, targets
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        scheduler.step()

        train_bal_acc = balanced_accuracy_score(all_labels, all_preds)
        avg_loss = train_loss / len(train_loader)

        # Validation
        val_bal_acc = evaluate(model, val_loader, device, is_dann, args.fp16)
        print(f"  Loss={avg_loss:.4f} TrainBA={train_bal_acc:.4f} | ValBA={val_bal_acc:.4f}"
              f" (stage={stage}, conf_pct={conf_pct:.1f}%)")

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pt"))
            patience_counter = 0
            print(f"  -> Best model saved (ValBA={val_bal_acc:.4f})")
        else:
            patience_counter += 1

        save_checkpoint(result_dir, epoch, model, optimizer, scheduler, scaler,
                        best_val_bal_acc, patience_counter)

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_val_bal_acc


# ========== Shared ==========

def evaluate(model, loader, device, is_dann, fp16=True):
    """Evaluate model on a loader, return balanced accuracy."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda", enabled=fp16):
                if is_dann:
                    logits, _, _ = model(imgs, 0.0)
                else:
                    logits = model(pixel_values=imgs).logits
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return balanced_accuracy_score(all_labels, all_preds)


def set_seed(seed):
    """Seed RNGs for reproducibility. Does NOT force deterministic cuDNN algorithms
    (avoids crashes on VideoMAE ops); residual GPU nondeterminism is accepted."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    start_time = time.time()
    if getattr(args, "seed", None) is not None:
        set_seed(args.seed)
        print(f"[Info] Seeded RNGs with {args.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = get_num_unified_classes()
    class_names = get_unified_class_names()
    print(f"[Info] Classes ({num_classes}): {class_names}")
    print(f"[Info] Method: {args.method}, Fold: {args.fold}")

    # Result directory
    if args.resume_dir:
        result_dir = args.resume_dir
        assert os.path.isdir(result_dir), f"Resume dir not found: {result_dir}"
        print(f"[Resume] Resuming from {result_dir}")
    else:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tgt_suffix = "" if args.target_dataset == "driveandact" else f"_{args.target_dataset}"
        tag_suffix = f"_{args.run_tag}" if getattr(args, "run_tag", "") else ""
        result_dir = os.path.join("results", f"{now}_cross_{args.method}{tgt_suffix}{tag_suffix}_fold{args.fold}")
        os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # Datasets (target only — source-free)
    if args.memmap_path and args.memmap_metadata:
        # Tier-2: pre-decoded uint8 memmap target (no decord I/O at train time).
        # val split tagged in metadata (2026-06-05): selection on val, eval on test.
        from cross_dataset.memmap_dataset import MemmapVideoDataset
        print(f"[memmap] using {args.memmap_path} (metadata={args.memmap_metadata})")
        _mm_crop = args.memmap_crop if args.memmap_crop > 0 else None
        dna_train = MemmapVideoDataset(
            args.memmap_path, args.memmap_metadata,
            fold=args.fold, split="train",
            num_frames=args.num_frames, augment=True, color_jitter=0.3,
            crop_size=_mm_crop,
        )
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
    else:
        if args.target_dataset == "driveandact":
            TargetDataset = DriveAndActDataset
            data_root = args.dna_root
        elif args.target_dataset == "dmd":
            TargetDataset = DMDDataset
            data_root = args.dmd_root
        elif args.target_dataset == "3mdad":
            TargetDataset = MDADDataset
            data_root = args.mdad_root

        dna_train = TargetDataset(
            data_root, split="train", fold=args.fold,
            num_frames=args.num_frames, augment=True,
            temporal_jitter=2, color_jitter=0.3,
        )
        dna_val = TargetDataset(
            data_root, split="val", fold=args.fold,
            num_frames=args.num_frames, augment=False,
        )
        dna_test = TargetDataset(
            data_root, split="test", fold=args.fold,
            num_frames=args.num_frames, augment=False,
        )

    train_loader = DataLoader(
        dna_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        dna_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        dna_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Load source model
    model = load_source_model(args.source_model, args.source_model_type,
                              num_classes, device)

    # Pre-adaptation evaluation
    is_dann = hasattr(model, 'domain_classifier')
    pre_val = evaluate(model, val_loader, device, is_dann, args.fp16)
    pre_test = evaluate(model, test_loader, device, is_dann, args.fp16)
    print(f"[Pre-adapt] Val BA={pre_val:.4f}, Test BA={pre_test:.4f}")

    # Run adaptation
    if args.method == "shot":
        best_val = train_shot(model, train_loader, val_loader, test_loader,
                              args, device, result_dir, class_names, num_classes)
    elif args.method == "cleanadapt":
        best_val = train_cleanadapt(model, train_loader, val_loader, test_loader,
                                    args, device, result_dir, class_names, num_classes)
    elif args.method == "csfda":
        best_val = train_csfda(model, train_loader, val_loader, test_loader,
                               args, device, result_dir, class_names, num_classes)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Final test with best model
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
                if is_dann:
                    logits, _, _ = model(imgs, 0.0)
                else:
                    logits = model(pixel_values=imgs).logits
            test_preds.extend(logits.argmax(-1).cpu().numpy())
            test_labels_list.extend(labels.numpy())

    test_acc = accuracy_score(test_labels_list, test_preds)
    test_bal_acc = balanced_accuracy_score(test_labels_list, test_preds)

    print(f"\n{'='*60}")
    print(f"[RESULT] Method={args.method} Fold={args.fold}")
    print(f"  Pre-adapt Test BA:  {pre_test:.4f}")
    print(f"  Post-adapt Test BA: {test_bal_acc:.4f}")
    print(f"  Improvement:        {test_bal_acc - pre_test:+.4f}")
    print(f"  Test Accuracy:      {test_acc:.4f}")
    print(f"{'='*60}")

    # Save results FIRST (before classification_report which can crash on missing classes)
    results = {
        "method": args.method,
        "fold": args.fold,
        "target_dataset": args.target_dataset,
        "source_model": args.source_model,
        "source_model_type": args.source_model_type,
        "pre_adapt_val_ba": pre_val,
        "pre_adapt_test_ba": pre_test,
        "test_accuracy": test_acc,
        "test_balanced_accuracy": test_bal_acc,
        "best_val_balanced_accuracy": best_val,
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
        f.write(f"Method: {args.method}, Fold: {args.fold}\n")
        f.write(f"Source model: {args.source_model}\n")
        f.write(f"Pre-adapt Test BA: {pre_test:.4f}\n")
        f.write(f"Post-adapt Test BA: {test_bal_acc:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(test_labels_list, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.title(f"Confusion Matrix — {args.method} (fold {args.fold})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    elapsed = (time.time() - start_time) / 60
    print(f"\n[Done] Total time: {elapsed:.1f} min. Results: {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--method", type=str, required=True,
                        choices=["shot", "cleanadapt", "csfda"])

    # Source model
    parser.add_argument("--source_model", type=str, required=True,
                        help="Path to source-trained model checkpoint")
    parser.add_argument("--source_model_type", type=str, default="dann",
                        choices=["dann", "cdan", "mcc", "bnm", "finetune", "target_only"],
                        help="Type of source model architecture")

    # Data
    parser.add_argument("--target_dataset", type=str, default="driveandact",
                        choices=["driveandact", "dmd", "3mdad"],
                        help="Target dataset: driveandact, dmd, or 3mdad")
    parser.add_argument("--dna_root", type=str,
                        default="<DATA_ROOT>/DriveAndAct")
    parser.add_argument("--dmd_root", type=str,
                        default="<DATA_ROOT>/DMD/distraction/dmd")
    parser.add_argument("--mdad_root", type=str,
                        default="<DATA_ROOT>/MDAD/MDAD")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed (ITEM1 parity with train_cross.py).")
    parser.add_argument("--run_tag", type=str, default="",
                        help="Suffix for result_dir to disambiguate cross-dataset UDA "
                             "SFDA runs from within-dataset SFDA (e.g. item1).")
    # Tier-2: pre-decoded uint8 memmap for the target dataset (D&A/DMD; 3MDAD stays decord).
    parser.add_argument("--memmap_path", type=str, default=None,
                        help="Pre-decoded uint8 memmap (.npy) for target. Requires --memmap_metadata.")
    parser.add_argument("--memmap_metadata", type=str, default=None,
                        help="Metadata json companion to --memmap_path.")
    parser.add_argument("--memmap_crop", type=int, default=0,
                        help="Tier-2: crop memmap frames to this size at load "
                             "(use 224 with 256-resolution memmaps to restore spatial-crop aug).")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=2)

    # SHOT
    parser.add_argument("--alpha_ent", type=float, default=1.0,
                        help="Entropy loss weight")
    parser.add_argument("--shot_pseudo_weight", type=float, default=0.3,
                        help="Pseudo-label CE weight for SHOT (0=entropy only)")
    parser.add_argument("--shot_threshold", type=float, default=0.9,
                        help="Confidence threshold for pseudo-labels")

    # CleanAdapt
    parser.add_argument("--clean_ratio", type=float, default=0.5,
                        help="Initial ratio of clean (small-loss) samples")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA teacher decay rate")

    # Resume
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Existing result directory to resume training from checkpoint.pt")
    parser.add_argument("--resume_epoch", type=int, default=None,
                        help="Override start epoch when resuming without checkpoint.pt")

    # C-SFDA
    parser.add_argument("--csfda_ce_weight", type=float, default=0.3,
                        help="CE weight on reliable pseudo-labels for C-SFDA")
    parser.add_argument("--csfda_mse_weight", type=float, default=0.1,
                        help="MSE guidance weight for unreliable samples (Stage 2)")
    parser.add_argument("--conf_percentile_start", type=float, default=80.0,
                        help="Starting confidence percentile (higher = stricter)")
    parser.add_argument("--conf_percentile_end", type=float, default=70.0,
                        help="Ending confidence percentile (gradually relaxed)")

    args = parser.parse_args()
    main(args)

"""Test-Time Adaptation evaluation for cross-dataset driver behavior recognition.

Implements test-time adaptation methods that adjust the model at inference time:
  - bn_adapt:   BN statistics adaptation (update BN with target statistics)
  - tent:       TENT (Wang et al., ICLR 2021) — entropy minimization on BN params
  - tta:        Full TTA — BN adapt + entropy minimization + temporal consistency
  - vitta:      ViTTA (Lin et al., CVPR 2023) — feature alignment + temporal consistency

These methods require no source data and can be applied on top of any trained model.
(ViTTA uses source model statistics collected once and cached.)

Usage:
  python -m cross_dataset.eval_tta \
      --method tent \
      --source_model results/XXXX_cross_dann_fold0/best_model.pt \
      --source_model_type dann \
      --dna_root <DATA_ROOT>/DriveAndAct \
      --fold 0
"""

import argparse
import copy
import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
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

from cross_dataset.class_mapping_utils import (
    get_unified_class_names,
    get_num_unified_classes,
)
from cross_dataset.driveandact_dataset import DriveAndActDataset
from cross_dataset.dmd_dataset import DMDDataset
from cross_dataset.mdad_dataset import MDADDataset
from cross_dataset.train_sfda import load_source_model


def collect_bn_layers(model):
    """Collect all BatchNorm layers from model."""
    bn_layers = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.LayerNorm)):
            bn_layers.append(module)
    return bn_layers


def bn_adapt(model, loader, device, model_type, fp16=True, num_batches=None):
    """Adapt BatchNorm/LayerNorm statistics using target data.
    Reset running stats and compute fresh statistics from target domain.
    """
    model.train()  # Enable BN in training mode to update running stats
    is_dann = model_type in ("dann", "cdan", "mcc", "bnm")

    # Reset running statistics for all norm layers
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            module.momentum = None  # Use cumulative moving average

    with torch.no_grad():
        for i, (imgs, _) in enumerate(tqdm(loader, desc="BN Adapt")):
            if num_batches and i >= num_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda", enabled=fp16):
                if is_dann:
                    model(imgs, 0.0)
                else:
                    model(pixel_values=imgs)

    model.eval()
    return model


def tent_adapt(model, loader, device, model_type, args):
    """TENT: entropy minimization on normalization layer parameters.
    Only optimizes affine parameters (gamma, beta) of norm layers.
    """
    is_dann = model_type in ("dann", "cdan", "mcc", "bnm")

    # Collect norm layer affine parameters
    params = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                params.append(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                params.append(module.bias)

    if not params:
        print("[TENT] No norm layer parameters found, skipping adaptation")
        return model

    print(f"[TENT] Adapting {len(params)} norm parameters")
    optimizer = torch.optim.Adam(params, lr=args.tta_lr)

    model.train()
    for epoch in range(args.tta_epochs):
        total_loss = 0.0
        for imgs, _ in tqdm(loader, desc=f"TENT E{epoch+1}/{args.tta_epochs}"):
            imgs = imgs.to(device, non_blocking=True)

            with autocast("cuda", enabled=args.fp16):
                if is_dann:
                    logits, _, _ = model(imgs, 0.0)
                else:
                    logits = model(pixel_values=imgs).logits

                # Entropy minimization
                softmax_out = F.softmax(logits, dim=1)
                entropy = -(softmax_out * torch.log(softmax_out + 1e-8)).sum(dim=1).mean()

            entropy.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += entropy.item()

        avg_loss = total_loss / len(loader)
        print(f"  TENT epoch {epoch+1}: avg entropy = {avg_loss:.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# ViTTA  (Lin et al., CVPR 2023) — Video Test-Time Adaptation
# ---------------------------------------------------------------------------


def _get_encoder_layernorms(model, model_type):
    """Return {name: module} for encoder LayerNorm layers only.

    We target the transformer encoder norms (not final classifier head) so
    that forward hooks capture intermediate feature statistics.
    """
    is_dann = model_type in ("dann", "cdan", "mcc", "bnm")
    norms = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        # Select encoder norms: for DANN wrapper they live under
        # videomae.videomae.encoder.*; for plain VideoMAE under
        # videomae.encoder.* — we also include fc_norm.
        if is_dann:
            if "videomae.videomae.encoder" in name or "fc_norm" in name:
                norms[name] = module
        else:
            if "videomae.encoder" in name or "fc_norm" in name:
                norms[name] = module
    return norms


def collect_source_stats(model, loader, device, model_type, fp16=True,
                         max_batches=50):
    """Collect per-LayerNorm feature mean/var from source-trained model.

    Uses forward hooks to accumulate running statistics over *max_batches*
    batches of the target train set (run through the model with no grad).
    We use the model right after loading (source weights), so these are
    effectively source-like statistics computed on target data distribution,
    which serves as the "source reference" for ViTTA alignment.

    Alternatively, if a cached file exists we load from there.

    Returns:
        dict  {layer_name: {'mean': Tensor, 'var': Tensor}}
    """
    is_dann = model_type in ("dann", "cdan", "mcc", "bnm")
    norm_layers = _get_encoder_layernorms(model, model_type)
    print(f"[ViTTA] Collecting source stats from {len(norm_layers)} LayerNorm layers "
          f"(max {max_batches} batches)")

    # Accumulators: Welford online algorithm
    accum = {}
    for name, module in norm_layers.items():
        accum[name] = {"count": 0, "mean": None, "M2": None}

    handles = []

    def _make_hook(layer_name):
        def hook_fn(module, input, output):
            feat = output.detach().float()
            # feat shape: (B, SeqLen, D) for encoder layers; (B, D) for fc_norm
            if feat.ndim == 3:
                feat = feat.reshape(-1, feat.shape[-1])  # (B*S, D)
            elif feat.ndim == 2:
                pass  # (B, D)
            else:
                return  # skip unexpected shapes
            n = feat.shape[0]
            batch_mean = feat.mean(dim=0)
            batch_var = feat.var(dim=0, unbiased=False)

            acc = accum[layer_name]
            if acc["mean"] is None:
                acc["mean"] = batch_mean
                acc["M2"] = batch_var * n
                acc["count"] = n
            else:
                # Combine: parallel Welford
                old_n = acc["count"]
                new_n = old_n + n
                delta = batch_mean - acc["mean"]
                acc["mean"] = acc["mean"] + delta * (n / new_n)
                acc["M2"] = acc["M2"] + batch_var * n + delta ** 2 * old_n * n / new_n
                acc["count"] = new_n
        return hook_fn

    for name, module in norm_layers.items():
        h = module.register_forward_hook(_make_hook(name))
        handles.append(h)

    model.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(tqdm(loader, desc="ViTTA src stats",
                                            total=min(max_batches, len(loader)))):
            if i >= max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda", enabled=fp16):
                if is_dann:
                    model(imgs, 0.0)
                else:
                    model(pixel_values=imgs)

    for h in handles:
        h.remove()

    stats = {}
    for name, acc in accum.items():
        if acc["mean"] is not None:
            stats[name] = {
                "mean": acc["mean"].cpu(),
                "var": (acc["M2"] / acc["count"]).cpu(),
            }
    print(f"[ViTTA] Collected stats for {len(stats)} layers")
    return stats


def _save_source_stats(stats, path):
    """Save source stats dict to disk."""
    save_dict = {}
    for name, sv in stats.items():
        save_dict[name] = {"mean": sv["mean"], "var": sv["var"]}
    torch.save(save_dict, path)


def _load_source_stats(path):
    """Load source stats dict from disk."""
    return torch.load(path, map_location="cpu", weights_only=False)


def temporal_augment(imgs, max_shift=2):
    """Create a temporally augmented view by rolling frames along time axis.

    Args:
        imgs: (B, T, C, H, W) tensor
        max_shift: maximum number of frames to shift

    Returns:
        Augmented tensor of same shape.
    """
    shift = random.randint(1, max_shift)
    if random.random() < 0.5:
        shift = -shift
    return torch.roll(imgs, shifts=shift, dims=1)


def _kl_gaussian(mu_p, var_p, mu_q, var_q, eps=1e-8):
    """KL divergence D_KL(P || Q) for diagonal Gaussians.

    P = source (reference), Q = current batch.
    """
    var_p = var_p + eps
    var_q = var_q + eps
    kl = 0.5 * (
        (var_p / var_q).log()      # log(sigma_q / sigma_p) but reversed sign
        + var_q / var_p
        + (mu_p - mu_q) ** 2 / var_p
        - 1.0
    )
    # Actually we want D_KL(Q || P) — align Q toward P (source).
    # D_KL(Q || P) = 0.5 * [ log(var_p/var_q) + var_q/var_p + (mu_p-mu_q)^2/var_p - 1 ]
    return kl.mean()


class _FeatureAlignmentHooksV2:
    """Register hooks on LayerNorm layers to compute alignment loss.

    Uses a shared mutable container so that ``reset()`` correctly clears
    the loss list referenced by all hook closures.

    Usage: call ``reset()`` before each forward pass, then ``get_loss()``
    after the forward to retrieve the accumulated KL alignment loss.
    """

    def __init__(self, model, model_type, source_stats, device):
        self._container = {"losses": []}  # mutable dict shared by closures
        self._handles = []
        norm_layers = _get_encoder_layernorms(model, model_type)

        for name, module in norm_layers.items():
            if name not in source_stats:
                continue
            src_mean = source_stats[name]["mean"].to(device)
            src_var = source_stats[name]["var"].to(device)
            h = module.register_forward_hook(
                self._make_hook(src_mean, src_var)
            )
            self._handles.append(h)

        self._num_hooks = len(self._handles)
        print(f"[ViTTA] Registered {self._num_hooks} alignment hooks")

    def _make_hook(self, src_mean, src_var):
        container = self._container

        def hook_fn(module, input, output):
            feat = output.float()
            if feat.ndim == 3:
                batch_mean = feat.mean(dim=[0, 1])
                batch_var = feat.var(dim=[0, 1], unbiased=False)
            elif feat.ndim == 2:
                batch_mean = feat.mean(dim=0)
                batch_var = feat.var(dim=0, unbiased=False)
            else:
                return
            kl = _kl_gaussian(src_mean, src_var, batch_mean, batch_var)
            container["losses"].append(kl)

        return hook_fn

    def get_loss(self):
        losses = self._container["losses"]
        if not losses:
            return torch.tensor(0.0)
        return sum(losses) / len(losses)

    def reset(self):
        self._container["losses"] = []

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def vitta_adapt(model, loader, device, model_type, source_stats, args):
    """ViTTA: Video Test-Time Adaptation (Lin et al., CVPR 2023).

    Two loss terms:
    1) Feature distribution alignment — KL divergence between current batch
       LayerNorm feature statistics and source reference statistics.
    2) Temporal consistency — encourage consistent predictions across two
       temporal views (original + temporally shifted) of each video.

    Only norm layer affine parameters (weight, bias) are updated.
    """
    is_dann = model_type in ("dann", "cdan", "mcc", "bnm")

    # Collect norm layer affine parameters for optimization
    params = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                params.append(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                params.append(module.bias)

    if not params:
        print("[ViTTA] No norm layer parameters found, skipping adaptation")
        return model

    print(f"[ViTTA] Adapting {len(params)} norm parameters, "
          f"lr={args.tta_lr}, epochs={args.vitta_epochs}, "
          f"lambda={args.vitta_lambda}, align_w={args.vitta_align_weight}")

    optimizer = torch.optim.Adam(params, lr=args.tta_lr)

    # Set up alignment hooks
    align_hooks = _FeatureAlignmentHooksV2(model, model_type, source_stats, device)

    model.train()
    for epoch in range(args.vitta_epochs):
        total_loss = 0.0
        total_align = 0.0
        total_consist = 0.0

        for imgs, _ in tqdm(loader, desc=f"ViTTA E{epoch+1}/{args.vitta_epochs}"):
            imgs = imgs.to(device, non_blocking=True)

            # View 1: original
            align_hooks.reset()
            with autocast("cuda", enabled=args.fp16):
                if is_dann:
                    logits1, _, _ = model(imgs, 0.0)
                else:
                    logits1 = model(pixel_values=imgs).logits
                loss_align1 = align_hooks.get_loss()

            # View 2: temporally augmented
            imgs2 = temporal_augment(imgs, max_shift=2)
            align_hooks.reset()
            with autocast("cuda", enabled=args.fp16):
                if is_dann:
                    logits2, _, _ = model(imgs2, 0.0)
                else:
                    logits2 = model(pixel_values=imgs2).logits
                loss_align2 = align_hooks.get_loss()

            # Feature alignment loss (average over both views)
            loss_align = (loss_align1 + loss_align2) / 2.0

            # Temporal consistency loss
            with autocast("cuda", enabled=args.fp16):
                p1 = F.softmax(logits1, dim=1)
                p2 = F.softmax(logits2, dim=1)
                avg_p = (p1 + p2) / 2.0
                loss_consist = (F.l1_loss(p1, avg_p) + F.l1_loss(p2, avg_p)) / 2.0

            # Combined loss
            loss = (args.vitta_align_weight * loss_align
                    + args.vitta_lambda * loss_consist)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_align += loss_align.item()
            total_consist += loss_consist.item()

        n = len(loader)
        print(f"  ViTTA epoch {epoch+1}: loss={total_loss/n:.4f} "
              f"(align={total_align/n:.4f}, consist={total_consist/n:.4f})")

    align_hooks.remove()
    model.eval()
    return model


def evaluate(model, loader, device, model_type, fp16=True):
    """Evaluate model and return predictions + labels."""
    is_dann = model_type in ("dann", "cdan", "mcc", "bnm")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Eval"):
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda", enabled=fp16):
                if is_dann:
                    logits, _, _ = model(imgs, 0.0)
                else:
                    logits = model(pixel_values=imgs).logits
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def main(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = get_num_unified_classes()
    class_names = get_unified_class_names()
    print(f"[Info] Classes ({num_classes}): {class_names}")
    print(f"[Info] TTA Method: {args.method}, Fold: {args.fold}")

    # Result directory
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tgt_suffix = "" if args.target_dataset == "driveandact" else f"_{args.target_dataset}"
    result_dir = os.path.join("results", f"{now}_cross_{args.method}{tgt_suffix}_fold{args.fold}")
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # Dataset
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
        num_frames=args.num_frames, augment=False,  # No augmentation for TTA
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
        num_workers=args.num_workers, pin_memory=True,
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
    print("\n[Pre-adapt evaluation]")
    pre_preds, pre_labels = evaluate(model, test_loader, device,
                                     args.source_model_type, args.fp16)
    pre_acc = accuracy_score(pre_labels, pre_preds)
    pre_bal_acc = balanced_accuracy_score(pre_labels, pre_preds)
    print(f"  Test Acc={pre_acc:.4f}, BA={pre_bal_acc:.4f}")

    # Apply TTA method
    print(f"\n[Applying {args.method}]")
    if args.method == "bn_adapt":
        model = bn_adapt(model, train_loader, device, args.source_model_type,
                         args.fp16)
    elif args.method == "tent":
        model = tent_adapt(model, train_loader, device, args.source_model_type,
                           args)
    elif args.method == "tta":
        # Combined: BN adapt first, then TENT
        model = bn_adapt(model, train_loader, device, args.source_model_type,
                         args.fp16)
        model = tent_adapt(model, train_loader, device, args.source_model_type,
                           args)
    elif args.method == "vitta":
        # ViTTA: collect source stats (or load cached), then adapt
        stats_cache = os.path.join(result_dir, "source_stats.pt")
        if args.vitta_stats_cache and os.path.isfile(args.vitta_stats_cache):
            print(f"[ViTTA] Loading cached source stats: {args.vitta_stats_cache}")
            source_stats = _load_source_stats(args.vitta_stats_cache)
        else:
            source_stats = collect_source_stats(
                model, train_loader, device, args.source_model_type,
                fp16=args.fp16, max_batches=args.vitta_stats_batches,
            )
            _save_source_stats(source_stats, stats_cache)
            print(f"[ViTTA] Saved source stats to {stats_cache}")
        model = vitta_adapt(model, train_loader, device,
                            args.source_model_type, source_stats, args)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Post-adaptation evaluation
    print("\n[Post-adapt evaluation]")
    post_preds, post_labels = evaluate(model, test_loader, device,
                                       args.source_model_type, args.fp16)
    test_acc = accuracy_score(post_labels, post_preds)
    test_bal_acc = balanced_accuracy_score(post_labels, post_preds)

    print(f"\n{'='*60}")
    print(f"[RESULT] TTA Method={args.method} Fold={args.fold}")
    print(f"  Pre-adapt:  Acc={pre_acc:.4f}, BA={pre_bal_acc:.4f}")
    print(f"  Post-adapt: Acc={test_acc:.4f}, BA={test_bal_acc:.4f}")
    print(f"  Change:     Acc={test_acc-pre_acc:+.4f}, BA={test_bal_acc-pre_bal_acc:+.4f}")
    print(f"{'='*60}")

    # Save results FIRST (before classification_report which can crash on missing classes)
    results = {
        "method": args.method,
        "fold": args.fold,
        "target_dataset": args.target_dataset,
        "source_model": args.source_model,
        "source_model_type": args.source_model_type,
        "pre_adapt_accuracy": pre_acc,
        "pre_adapt_balanced_accuracy": pre_bal_acc,
        "test_accuracy": test_acc,
        "test_balanced_accuracy": test_bal_acc,
    }
    with open(os.path.join(result_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    report = classification_report(
        post_labels, post_preds,
        labels=list(range(len(class_names))),
        target_names=class_names, digits=4, zero_division=0,
    )
    print(report)

    with open(os.path.join(result_dir, "classification_report.txt"), "w") as f:
        f.write(f"TTA Method: {args.method}, Fold: {args.fold}\n")
        f.write(f"Source: {args.source_model}\n")
        f.write(f"Pre-adapt: Acc={pre_acc:.4f}, BA={pre_bal_acc:.4f}\n")
        f.write(f"Post-adapt: Acc={test_acc:.4f}, BA={test_bal_acc:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(post_labels, post_preds)
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

    parser.add_argument("--method", type=str, required=True,
                        choices=["bn_adapt", "tent", "tta", "vitta"])

    # Source model
    parser.add_argument("--source_model", type=str, required=True)
    parser.add_argument("--source_model_type", type=str, default="dann",
                        choices=["dann", "cdan", "mcc", "bnm", "finetune", "target_only"])

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

    # TTA params
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tta_lr", type=float, default=1e-4,
                        help="Learning rate for TENT optimization")
    parser.add_argument("--tta_epochs", type=int, default=3,
                        help="Number of TENT adaptation epochs")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=2)

    # ViTTA-specific params
    parser.add_argument("--vitta_epochs", type=int, default=3,
                        help="Number of ViTTA adaptation epochs")
    parser.add_argument("--vitta_lambda", type=float, default=0.05,
                        help="Temporal consistency loss weight for ViTTA")
    parser.add_argument("--vitta_align_weight", type=float, default=1.0,
                        help="Feature alignment loss weight for ViTTA")
    parser.add_argument("--vitta_stats_batches", type=int, default=50,
                        help="Max batches for source statistics collection")
    parser.add_argument("--vitta_stats_cache", type=str, default=None,
                        help="Path to cached source_stats.pt (skip collection)")

    args = parser.parse_args()
    main(args)

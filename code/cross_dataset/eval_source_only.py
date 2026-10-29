"""Evaluate OBC-trained model on Drive&Act (source-only / zero-shot).

Measures domain gap by applying OBC model directly to Drive&Act test set
after re-mapping both to unified class labels.

Usage:
    python -m cross_dataset.eval_source_only \
        --model_path results/20260305_191532_videomae_da_dann_none/best_model.pt \
        --obc_metadata features/keyframe_video_metadata_dualcam.pkl \
        --dna_root <DATA_ROOT>/DriveAndAct \
        --fold 0 \
        --batch_size 8
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

from cross_dataset.class_mapping_utils import (
    get_unified_class_names,
    get_unified_label_maps,
    get_num_unified_classes,
    load_class_mapping,
)
from cross_dataset.obc_dataset import OBCCrossDataset
from cross_dataset.driveandact_dataset import DriveAndActDataset
from cross_dataset.dmd_dataset import DMDDataset
from cross_dataset.mdad_dataset import MDADDataset


def load_obc_model_with_unified_head(model_path, num_unified_classes=6, device="cuda", seed=42):
    """Load OBC 33-class model and replace head with unified class head.

    Maps OBC's 33-class predictions to unified classes via probability aggregation.
    """
    torch.manual_seed(seed)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Determine model type
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Check if DANN model (has domain_classifier keys)
    is_dann = any("domain_classifier" in k for k in state_dict.keys())

    # The original training script (train_videomae_da.py) saves the classification
    # head under "action_classifier.*", not "classifier.*" (this module's
    # VideoMAEForVideoClassification attribute name). Without this rename,
    # load_state_dict(strict=False) silently leaves `classifier` at its random
    # initialization instead of the trained weights (bn_adapt_verification_report_20260715.md).
    KEY_RENAME = {
        "action_classifier.weight": "classifier.weight",
        "action_classifier.bias": "classifier.bias",
    }
    # subject_head.* is an auxiliary adversarial head from the original training
    # script; intentionally unused here (33-class action classification only).
    DROP_PREFIXES = ("subject_head.",)
    state_dict = {
        KEY_RENAME.get(k, k): v
        for k, v in state_dict.items()
        if not any(k.startswith(p) for p in DROP_PREFIXES)
    }

    # Build model
    config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    config.num_labels = 33
    model = VideoMAEForVideoClassification(config)

    if is_dann:
        # Extract only the VideoMAE weights (ignore domain_classifier)
        videomae_state = {}
        for k, v in state_dict.items():
            if k.startswith("videomae.") or k.startswith("classifier."):
                videomae_state[k] = v
            elif k.startswith("fc_norm."):
                videomae_state[k] = v
        load_dict = videomae_state
    else:
        load_dict = state_dict

    missing, unexpected = model.load_state_dict(load_dict, strict=False)
    assert missing == [], (
        f"Checkpoint failed to populate model parameters: {missing}. "
        f"Aborting rather than silently evaluating a partially-random model."
    )
    bad_unexpected = [k for k in unexpected if not k.startswith("subject_head.")]
    assert bad_unexpected == [], (
        f"Unaccounted-for checkpoint keys (not loaded, not in the allowed-drop list): {bad_unexpected}"
    )

    model.eval()
    model.to(device)
    return model


def build_obc_to_unified_mapping(mapping_path=None):
    """Build mapping from OBC 33-class indices to unified labels.

    Returns:
        dict: OBC class index (0-32) -> unified label (0-5), or None if unmapped
    """
    mapping = load_class_mapping(mapping_path)
    obc_class_names = mapping["obc_class_names"]
    label_maps = get_unified_label_maps(mapping_path)
    obc_map = label_maps["obc"]

    idx_map = {}
    for i, name in enumerate(obc_class_names):
        if name in obc_map:
            idx_map[i] = obc_map[name]
    return idx_map


def aggregate_logits_to_unified(logits_33, obc_to_unified, num_unified=6):
    """Aggregate 33-class logits to unified classes by averaging probabilities.

    For classes that map many-to-one (e.g., 5 OBC reaching classes -> 1 unified),
    we average their softmax probabilities to avoid count bias.

    Args:
        logits_33: Tensor [B, 33]
        obc_to_unified: dict mapping OBC idx -> unified label
        num_unified: Number of unified classes

    Returns:
        Tensor [B, num_unified] aggregated log-probabilities
    """
    probs_33 = torch.softmax(logits_33, dim=-1)  # [B, 33]
    B = probs_33.shape[0]
    probs_unified = torch.zeros(B, num_unified, device=probs_33.device)
    counts = torch.zeros(num_unified, device=probs_33.device)

    for obc_idx, unified_idx in obc_to_unified.items():
        probs_unified[:, unified_idx] += probs_33[:, obc_idx]
        counts[unified_idx] += 1

    counts = counts.clamp(min=1)
    probs_unified = probs_unified / counts.unsqueeze(0)

    # Convert back to log-probs for consistency
    return torch.log(probs_unified + 1e-8)


def bn_adapt_source_model(model, loader, device="cuda"):
    """Adapt BatchNorm running statistics using target training data.

    Resets BN running stats and recomputes them from the target domain,
    while keeping all model parameters frozen. This tests whether
    BN statistic mismatch contributes to the cross-dataset domain gap.
    """
    model.train()  # Enable BN training mode to update running stats

    # Reset running statistics for all BatchNorm layers
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            module.momentum = None  # Use cumulative moving average

    with torch.no_grad():
        for batch_tensor, _ in tqdm(loader, desc="BN Adapt (source model)"):
            batch_tensor = batch_tensor.to(device)
            model(pixel_values=batch_tensor)

    model.eval()
    return model


@torch.no_grad()
def evaluate(model, dataloader, obc_to_unified, num_unified, device="cuda"):
    """Run evaluation and return predictions."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch_tensor, batch_labels in tqdm(dataloader, desc="Evaluating"):
        batch_tensor = batch_tensor.to(device)
        outputs = model(pixel_values=batch_tensor)
        logits_33 = outputs.logits  # [B, 33]

        # Aggregate to unified classes
        logits_unified = aggregate_logits_to_unified(
            logits_33, obc_to_unified, num_unified
        )
        preds = logits_unified.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels.numpy().tolist())

    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to OBC trained model checkpoint")
    parser.add_argument("--obc_metadata", type=str,
                        default="features/keyframe_video_metadata_dualcam.pkl")
    parser.add_argument("--dna_root", type=str,
                        default="<DATA_ROOT>/DriveAndAct")
    parser.add_argument("--dmd_root", type=str,
                        default="<DATA_ROOT>/DMD/distraction/dmd")
    parser.add_argument("--mdad_root", type=str,
                        default="<DATA_ROOT>/MDAD/MDAD")
    parser.add_argument("--target_dataset", type=str, default="driveandact",
                        choices=["driveandact", "dmd", "3mdad"],
                        help="Target dataset for source-only eval")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mapping_path", type=str, default=None)
    parser.add_argument("--bn_adapt", action="store_true",
                        help="Apply BN adaptation using target train data before evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (model construction is now fully overwritten by the "
                             "checkpoint, so this only affects any stray non-loaded init/ops)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_unified = get_num_unified_classes(args.mapping_path)
    unified_names = get_unified_class_names(args.mapping_path)

    # Output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_bn_adapt" if args.bn_adapt else ""
        tgt_suffix = f"_{args.target_dataset}" if args.target_dataset != "driveandact" else ""
        args.output_dir = f"results/{timestamp}_source_only{suffix}_eval{tgt_suffix}_fold{args.fold}"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Unified classes ({num_unified}): {unified_names}")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading OBC model from: {args.model_path}")
    model = load_obc_model_with_unified_head(args.model_path, num_unified, device, seed=args.seed)
    obc_to_unified = build_obc_to_unified_mapping(args.mapping_path)
    print(f"OBC->Unified mapping: {len(obc_to_unified)} classes mapped")

    results = {}

    # 1. Evaluate on OBC test set (unified classes only)
    print("\n" + "=" * 60)
    print("Evaluating on OBC test set (unified classes)")
    print("=" * 60)
    obc_test = OBCCrossDataset(
        args.obc_metadata, split="test",
        num_frames=16, augment=False,
        mapping_path=args.mapping_path,
    )
    obc_loader = DataLoader(
        obc_test, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
    )
    obc_preds, obc_labels = evaluate(model, obc_loader, obc_to_unified, num_unified, device)

    obc_acc = accuracy_score(obc_labels, obc_preds)
    obc_bal_acc = balanced_accuracy_score(obc_labels, obc_preds)
    obc_report = classification_report(
        obc_labels, obc_preds,
        target_names=unified_names, digits=4, zero_division=0,
    )
    print(f"OBC Test — Acc: {obc_acc:.4f}, Bal Acc: {obc_bal_acc:.4f}")
    print(obc_report)
    results["obc_test"] = {"accuracy": obc_acc, "balanced_accuracy": obc_bal_acc}

    # 2. (Optional) BN Adaptation using target training data
    tgt = args.target_dataset
    if args.bn_adapt:
        print("\n" + "=" * 60)
        print(f"BN Adapt: updating BN stats with {tgt} train data (fold={args.fold})")
        print("=" * 60)
        if tgt == "dmd":
            tgt_train = DMDDataset(
                args.dmd_root, split="train", fold=args.fold,
                num_frames=16, augment=False,
            )
        elif tgt == "3mdad":
            tgt_train = MDADDataset(
                args.mdad_root, split="train", fold=args.fold,
                num_frames=16, augment=False,
            )
        else:
            tgt_train = DriveAndActDataset(
                args.dna_root, split="train", fold=args.fold,
                num_frames=16, augment=False,
                mapping_path=args.mapping_path,
            )
        train_loader = DataLoader(
            tgt_train, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, shuffle=False,
        )
        model = bn_adapt_source_model(model, train_loader, device)
        print(f"BN Adapt complete. {len(tgt_train)} training samples processed.")

    # 3. Evaluate on target test set
    print("\n" + "=" * 60)
    print(f"Evaluating on {tgt} test set (fold={args.fold}, source-only{' + BN-Adapt' if args.bn_adapt else ''})")
    print("=" * 60)
    if tgt == "dmd":
        tgt_test = DMDDataset(
            args.dmd_root, split="test", fold=args.fold,
            num_frames=16, augment=False,
        )
    elif tgt == "3mdad":
        tgt_test = MDADDataset(
            args.mdad_root, split="test", fold=args.fold,
            num_frames=16, augment=False,
        )
    else:
        tgt_test = DriveAndActDataset(
            args.dna_root, split="test", fold=args.fold,
            num_frames=16, augment=False,
            mapping_path=args.mapping_path,
        )
    tgt_loader = DataLoader(
        tgt_test, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
    )
    tgt_preds, tgt_labels = evaluate(model, tgt_loader, obc_to_unified, num_unified, device)

    tgt_acc = accuracy_score(tgt_labels, tgt_preds)
    tgt_bal_acc = balanced_accuracy_score(tgt_labels, tgt_preds)
    tgt_report = classification_report(
        tgt_labels, tgt_preds,
        labels=list(range(num_unified)),
        target_names=unified_names, digits=4, zero_division=0,
    )
    tgt_key = f"{tgt}_test" if tgt != "driveandact" else "driveandact_test"
    print(f"{tgt} Test — Acc: {tgt_acc:.4f}, Bal Acc: {tgt_bal_acc:.4f}")
    print(tgt_report)
    results[tgt_key] = {"accuracy": tgt_acc, "balanced_accuracy": tgt_bal_acc}

    # Domain gap
    gap = obc_bal_acc - tgt_bal_acc
    print(f"\n{'='*60}")
    print(f"Domain Gap (OBC bal_acc - {tgt} bal_acc): {gap:.4f}")
    print(f"{'='*60}")
    results["domain_gap"] = gap

    # Save results
    results_path = os.path.join(args.output_dir, "source_only_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save reports
    with open(os.path.join(args.output_dir, "obc_test_report.txt"), "w") as f:
        f.write(f"OBC Test (unified classes)\n")
        f.write(f"Accuracy: {obc_acc:.4f}\n")
        f.write(f"Balanced Accuracy: {obc_bal_acc:.4f}\n\n")
        f.write(obc_report)

    with open(os.path.join(args.output_dir, f"{tgt}_test_report.txt"), "w") as f:
        f.write(f"{tgt} Test (source-only, fold={args.fold})\n")
        f.write(f"Accuracy: {tgt_acc:.4f}\n")
        f.write(f"Balanced Accuracy: {tgt_bal_acc:.4f}\n\n")
        f.write(tgt_report)


if __name__ == "__main__":
    main()

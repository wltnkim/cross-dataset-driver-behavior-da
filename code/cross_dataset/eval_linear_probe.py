"""Linear probing experiment: VideoMAE pretrained features + LogisticRegression.

Tests how discriminative the pretrained VideoMAE (Kinetics-400) features are
for the unified 6-class taxonomy WITHOUT any task-specific fine-tuning.
This empirically grounds the claim that strong pretrained representations
make DA unnecessary.

Usage:
    python -m cross_dataset.eval_linear_probe \
        --target_dataset driveandact \
        --fold 0 \
        --label_ratios 0.005 0.01 0.025 0.05 0.1 1.0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

from cross_dataset.class_mapping_utils import get_num_unified_classes
from cross_dataset.driveandact_dataset import DriveAndActDataset
from cross_dataset.dmd_dataset import DMDDataset
from cross_dataset.mdad_dataset import MDADDataset


def load_pretrained_videomae(device="cuda", model_name="MCG-NJU/videomae-base-finetuned-kinetics"):
    """Load VideoMAE pretrained on Kinetics-400 (NO task-specific fine-tuning).

    Default is the base variant; pass model_name='MCG-NJU/videomae-small-finetuned-kinetics'
    for the bbSmall ablation."""
    config = VideoMAEConfig.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(model_name, config=config)
    model.eval()
    model.to(device)
    return model


def load_source_adapted_videomae(source_model_path, device="cuda", seed=42):
    """Load a task-adapted VideoMAE (e.g., DANN ckpt) and return the same HF wrapper
    so extract_features() can reuse model.videomae / model.fc_norm unchanged.

    The DANN ckpt (train_cross.py) is trained end-to-end on the unified six-class
    cross-dataset task, not OBC's original 33-class task, so its classifier head is
    [num_unified, 768]. We keep num_labels=get_num_unified_classes() and load the
    videomae/fc_norm/classifier weights, discarding domain_classifier. (The previous
    hardcoded num_labels=33 was a leftover assumption from the unrelated OBC-checkpoint
    loader in eval_source_only.py and never actually mattered under the old buggy
    load, since the classifier key never matched by name in the first place -- see
    paper/p0_fix_and_rerun_report_20260715.md.)
    """
    torch.manual_seed(seed)
    checkpoint = torch.load(source_model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    is_dann = any(k.startswith("domain_classifier.") for k in state_dict.keys())

    config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    config.num_labels = get_num_unified_classes()
    model = VideoMAEForVideoClassification(config)

    # VideoMAEDANN (train_cross.py) wraps the whole HF classification model under a
    # top-level `videomae` attribute, so a DANN checkpoint's backbone/classifier/fc_norm
    # keys carry one extra "videomae." prefix (e.g. "videomae.videomae.embeddings...",
    # "videomae.classifier.weight") relative to this function's plain
    # VideoMAEForVideoClassification. Strip that one wrapper level for DANN checkpoints
    # so keys actually line up (previously silently left the whole backbone at random
    # init -- see paper/p0_fix_and_rerun_report_20260715.md). domain_classifier.*
    # (adversarial discriminator) and subject_head.* (auxiliary head present in some
    # older external checkpoints) are intentionally dropped; neither exists in the
    # plain classification model.
    DROP_PREFIXES = ("domain_classifier.", "subject_head.")
    backbone_state = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in DROP_PREFIXES):
            continue
        if is_dann and k.startswith("videomae."):
            k = k[len("videomae."):]
        if k.startswith("videomae.") or k.startswith("fc_norm.") or k.startswith("classifier."):
            backbone_state[k] = v

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"[source_model] loaded {len(backbone_state)} keys from {source_model_path}")
    print(f"[source_model] missing={len(missing)} unexpected={len(unexpected)}")
    assert missing == [], (
        f"Checkpoint failed to populate model parameters: {missing}. "
        f"Aborting rather than silently evaluating a partially-random backbone."
    )
    assert unexpected == [], (
        f"Unaccounted-for checkpoint keys (not loaded, not in the allowed-drop list): {unexpected}"
    )

    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def extract_features(model, loader, device="cuda"):
    """Extract 768-dim features using mean pooling on VideoMAE sequence output."""
    all_features = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device)
        outputs = model.videomae(pixel_values=imgs)
        seq_out = outputs.last_hidden_state  # [B, seq_len, 768]
        if model.fc_norm is not None:
            features = model.fc_norm(seq_out.mean(1))  # [B, 768]
        else:
            features = seq_out[:, 0]  # CLS token

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def subsample_by_ratio(labels, ratio, seed=42):
    """Subsample indices preserving class balance (at least 1 per class)."""
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    selected = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        n_select = max(1, int(len(idx_c) * ratio))
        selected.extend(rng.choice(idx_c, size=n_select, replace=False).tolist())
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dataset", type=str, default="driveandact",
                        choices=["driveandact", "dmd", "3mdad"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--label_ratios", type=float, nargs="+",
                        default=[0.005, 0.01, 0.025, 0.05, 0.1, 1.0])
    parser.add_argument("--dna_root", type=str,
                        default="<DATA_ROOT>/DriveAndAct")
    parser.add_argument("--dmd_root", type=str,
                        default="<DATA_ROOT>/DMD/distraction/dmd")
    parser.add_argument("--mdad_root", type=str,
                        default="<DATA_ROOT>/MDAD/MDAD")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--source_model", type=str, default=None,
                        help="Path to DANN (or other source-adapted) ckpt. If set, "
                             "use its videomae backbone instead of K400 pretrained.")
    parser.add_argument("--tag", type=str, default="linear_probe",
                        help="Output-dir tag (e.g. 'linear_probe_dann')")
    parser.add_argument("--model_name", type=str,
                        default="MCG-NJU/videomae-base-finetuned-kinetics",
                        help="HF model id for K400-pretrained backbone (ignored if --source_model is set)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (subsampling + LogisticRegression already default to "
                             "this value; also seeds torch for the backbone load)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tgt = args.target_dataset
    fold = args.fold

    # Output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        tgt_suffix = f"_{tgt}" if tgt != "driveandact" else ""
        args.output_dir = f"results/{timestamp}_{args.tag}{tgt_suffix}_fold{fold}"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Linear Probe ({args.tag}): {tgt} fold{fold} ===")
    print(f"Device: {device}")
    print(f"Label ratios: {args.label_ratios}")

    # Load backbone: source-adapted (e.g. DANN) if given, else K400 pretrained
    if args.source_model:
        print(f"\nLoading source-adapted VideoMAE from: {args.source_model}")
        model = load_source_adapted_videomae(args.source_model, device, seed=args.seed)
    else:
        print(f"\nLoading pretrained VideoMAE (Kinetics-400) from: {args.model_name}")
        model = load_pretrained_videomae(device, model_name=args.model_name)

    # Load target datasets
    ds_kwargs = dict(num_frames=16, augment=False)
    if tgt == "dmd":
        train_ds = DMDDataset(args.dmd_root, split="train", fold=fold, **ds_kwargs)
        test_ds = DMDDataset(args.dmd_root, split="test", fold=fold, **ds_kwargs)
    elif tgt == "3mdad":
        train_ds = MDADDataset(args.mdad_root, split="train", fold=fold, **ds_kwargs)
        test_ds = MDADDataset(args.mdad_root, split="test", fold=fold, **ds_kwargs)
    else:
        train_ds = DriveAndActDataset(args.dna_root, split="train", fold=fold, **ds_kwargs)
        test_ds = DriveAndActDataset(args.dna_root, split="test", fold=fold, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True, shuffle=False)

    # Extract features
    print(f"\nExtracting train features ({len(train_ds)} samples)...")
    train_feats, train_labels = extract_features(model, train_loader, device)
    print(f"Extracting test features ({len(test_ds)} samples)...")
    test_feats, test_labels = extract_features(model, test_loader, device)

    # Standardize features
    scaler = StandardScaler()
    train_feats_scaled = scaler.fit_transform(train_feats)
    test_feats_scaled = scaler.transform(test_feats)

    # Linear probing at each label ratio
    results = {
        "target_dataset": tgt,
        "fold": fold,
        "tag": args.tag,
        "source_model": args.source_model,
        "n_train_total": len(train_labels),
        "n_test": len(test_labels),
        "ratios": {}
    }

    print(f"\n{'Ratio':>8} {'N_train':>8} {'Bal Acc':>8}")
    print("-" * 28)

    for ratio in args.label_ratios:
        if ratio >= 1.0:
            idx = list(range(len(train_labels)))
        else:
            idx = subsample_by_ratio(train_labels, ratio)

        n_train = len(idx)
        X_train = train_feats_scaled[idx]
        y_train = train_labels[idx]

        # Check if we have at least 2 classes
        if len(np.unique(y_train)) < 2:
            print(f"{ratio:>8.3f} {n_train:>8d} {'SKIP':>8} (insufficient classes)")
            continue

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(test_feats_scaled)
        bal_acc = balanced_accuracy_score(test_labels, y_pred)

        results["ratios"][str(ratio)] = {
            "n_train": n_train,
            "balanced_accuracy": float(bal_acc),
        }
        print(f"{ratio:>8.3f} {n_train:>8d} {bal_acc:>8.4f}")

    # Save results
    results_path = os.path.join(args.output_dir, "linear_probe_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

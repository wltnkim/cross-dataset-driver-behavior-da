"""Analysis and visualization for cross-dataset domain adaptation experiments.

Subcommands:
  table   — Collect results, print comparison table, save CSV
  tsne    — t-SNE visualization of source/target features
  adist   — Proxy A-distance estimation via linear SVM
  confmat — Side-by-side confusion matrices for selected methods
  perclass — Per-class F1 bar chart across methods
  all     — Run all analyses

Usage:
  python -m cross_dataset.analyze_results table
  python -m cross_dataset.analyze_results tsne --model_path X --model_type dann --fold 0
  python -m cross_dataset.analyze_results adist --model_path X --model_type dann --fold 0
  python -m cross_dataset.analyze_results confmat
  python -m cross_dataset.analyze_results perclass
  python -m cross_dataset.analyze_results all --model_path X --model_type dann --fold 0
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict


# ========== Constants ==========

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OBC_METADATA = os.path.join(BASE_DIR, "features", "keyframe_video_metadata_dualcam.pkl")
DNA_ROOT_DEFAULT = "<DATA_ROOT>/DriveAndAct"
DMD_ROOT_DEFAULT = "<DATA_ROOT>/DMD/distraction/dmd"
MDAD_ROOT_DEFAULT = "<DATA_ROOT>/MDAD/MDAD"

TARGET_DISPLAY = {
    "driveandact": "D&A",
    "dmd": "DMD",
    "3mdad": "3MDAD",
}

# Canonical method display order
METHOD_ORDER = [
    "source_only", "target_only", "finetune",
    "dann", "cdan", "mcc", "bnm", "cbdann_ent", "cbdann_freq",
    "shot", "cleanadapt", "csfda",
    "bn_adapt", "tent", "tta", "vitta",
]


# ========== 1. Result Collection & Table ==========

def _parse_target_from_dir(dir_name):
    """Infer target dataset from directory name.

    Convention: *_cross_{mode}[_{target}][_r{ratio}]_fold{fold}
    - No suffix → driveandact
    - _dmd_ → dmd
    - _3mdad_ → 3mdad
    """
    if "_dmd_" in dir_name or dir_name.endswith("_dmd"):
        return "dmd"
    if "_3mdad_" in dir_name or dir_name.endswith("_3mdad"):
        return "3mdad"
    return "driveandact"


# Directories that are NOT part of the main-results population, shared verbatim with
# code/cross_dataset/paper/regen_figures.py / regen_reviewer_figs.py (keep in sync).
EXC = re.compile(r"reverse|bbSmall|bbRandom|bbsmall|bbrandom|item1|srconly|"
                 r"tier2sanity|sanity|smoke|_uni_|_r\d{3}(_|$)")


def _dedup_key(rec):
    """Prefer the run with more trained epochs; only then the latest directory name.
    A truncated re-verification run (3 epochs, June) must not outrank the complete
    23-epoch April run it re-checked (dmd/target_only/fold2, found 2026-09-11). Runs
    without an epochs_trained field (eval-only methods) rank below any run that has one."""
    ep = rec.get("epochs")
    return (ep if ep is not None else -1, rec["dir_name"])


def collect_results(target_dataset=None):
    """Scan results/ for cross-dataset experiment results.

    Looks for:
      - results/*_cross_*_fold*/results.json  (standard experiments)
      - results/*_source_only_eval/source_only_results.json  (source-only baselines)

    Args:
        target_dataset: If specified, only return results for this target
                        ('driveandact', 'dmd', '3mdad'). None returns all.

    Returns list of dicts with keys: target, mode, fold, test_accuracy,
    test_balanced_accuracy, dir_path.
    For directories with the same (target, mode, fold), keeps only the latest
    (by directory name timestamp).
    """
    records = []

    # Standard cross-dataset results (exclude SSDA/LE dirs with _r\d{3}_ pattern)
    pattern = os.path.join(RESULTS_DIR, "*_cross_*_fold*", "results.json")
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                data = json.load(f)
            dir_path = os.path.dirname(path)
            dir_name = os.path.basename(dir_path)
            # Skip SSDA/UDA/Label-Efficiency dirs (contain _r000_, _r005_, etc.) AND every
            # non-main-table population, same list as paper/regen_*_figs.py (2026-09-16):
            # reverse-direction, backbone ablations, item1, source-only-start, sanity/smoke
            # test runs, and the `_uni_` unified-head variants. Before this, a 2-epoch
            # ISAAC smoke-test run outranked the real Drive&Act/DANN fold-0 run under the
            # "latest dirname wins" rule below.
            if EXC.search(dir_name):
                continue
            # Prefer target_dataset from JSON, fall back to dir name parsing
            tgt = data.get("target_dataset") or _parse_target_from_dir(dir_name)
            mode = data.get("mode", data.get("method", "unknown"))
            # Distinguish cbdann entropy vs frequency via config.yaml
            if mode == "cbdann":
                config_path = os.path.join(dir_path, "config.yaml")
                cbw = data.get("cbdann_class_weight")
                if not cbw and os.path.exists(config_path):
                    with open(config_path) as cf:
                        for line in cf:
                            if line.strip().startswith("cbdann_class_weight:"):
                                cbw = line.strip().split(":", 1)[1].strip()
                                break
                if cbw == "entropy":
                    mode = "cbdann_ent"
                elif cbw == "frequency":
                    mode = "cbdann_freq"
                # else: keep "cbdann" as-is (uniform or unknown)
            records.append({
                "target": tgt,
                "mode": mode,
                "fold": int(data.get("fold", 0)),
                "test_accuracy": data.get("test_accuracy", float("nan")),
                "test_balanced_accuracy": data.get("test_balanced_accuracy", float("nan")),
                "epochs": data.get("epochs_trained"),
                "dir_path": dir_path,
                "dir_name": dir_name,
            })
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"[WARN] Skipping {path}: {e}")

    # Source-only eval results (matches both old and new naming conventions)
    pattern_src = os.path.join(RESULTS_DIR, "*_source_only_eval*", "source_only_results.json")
    for path in glob.glob(pattern_src):
        try:
            with open(path) as f:
                data = json.load(f)
            dir_path = os.path.dirname(path)
            dir_name = os.path.basename(dir_path)

            # Determine target and fold from directory name
            # Patterns: *_source_only_eval, *_source_only_eval_fold{N},
            #           *_source_only_eval_{target}_fold{N}
            fold_match = re.search(r"_fold(\d+)$", dir_name)
            fold = int(fold_match.group(1)) if fold_match else 0

            if "_dmd_" in dir_name:
                tgt = "dmd"
                metrics = data.get("dmd_test", {})
            elif "_3mdad_" in dir_name:
                tgt = "3mdad"
                metrics = data.get("3mdad_test", {})
            else:
                tgt = "driveandact"
                metrics = data.get("driveandact_test", {})

            if not metrics:
                continue

            records.append({
                "target": tgt,
                "mode": "source_only",
                "fold": fold,
                "test_accuracy": metrics.get("accuracy", float("nan")),
                "test_balanced_accuracy": metrics.get("balanced_accuracy", float("nan")),
                "dir_path": dir_path,
                "dir_name": dir_name,
            })
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"[WARN] Skipping {path}: {e}")

    # Deduplicate: keep latest run per (target, mode, fold)
    latest = {}
    for rec in records:
        key = (rec["target"], rec["mode"], rec["fold"])
        if key not in latest or _dedup_key(rec) > _dedup_key(latest[key]):
            latest[key] = rec

    results = list(latest.values())

    # Filter by target_dataset if specified
    if target_dataset:
        results = [r for r in results if r["target"] == target_dataset]

    return results


def build_table(records):
    """Build a structured table: method x fold with BalAcc and Acc."""
    # Group by method
    by_method = defaultdict(dict)
    for rec in records:
        by_method[rec["mode"]][rec["fold"]] = rec

    # Determine all folds present
    all_folds = sorted(set(rec["fold"] for rec in records))

    return by_method, all_folds


def print_and_save_table(records, output_dir, target_label=None):
    """Print comparison table and save as CSV."""
    by_method, all_folds = build_table(records)

    os.makedirs(output_dir, exist_ok=True)

    # Console output
    fold_headers = [f"Fold {f}" for f in all_folds]
    header = f"{'Method':<18}" + "".join(f"{'BalAcc':>10}{'Acc':>10}" for _ in all_folds)
    header += f"{'Mean BalAcc':>14}{'Std':>8}"
    sep = "-" * len(header)

    print("\n" + sep)
    print("Cross-Dataset Domain Adaptation Results")
    print(sep)
    print(f"{'Method':<18}", end="")
    for f in all_folds:
        print(f"{'Fold ' + str(f) + ' BA':>10}{'Fold ' + str(f) + ' Acc':>10}", end="")
    print(f"{'Mean BA':>14}{'Std BA':>8}")
    print(sep)

    # CSV lines
    csv_lines = []
    csv_header = "method," + ",".join(
        f"fold{f}_balacc,fold{f}_acc" for f in all_folds
    ) + ",mean_balacc,std_balacc"
    csv_lines.append(csv_header)

    # Print in canonical order, then any extras
    methods_present = list(by_method.keys())
    ordered = [m for m in METHOD_ORDER if m in methods_present]
    extras = [m for m in methods_present if m not in METHOD_ORDER]
    ordered.extend(sorted(extras))

    for method in ordered:
        fold_data = by_method[method]
        ba_values = []
        line = f"{method:<18}"
        csv_parts = [method]

        for f in all_folds:
            if f in fold_data:
                ba = fold_data[f]["test_balanced_accuracy"]
                acc = fold_data[f]["test_accuracy"]
                ba_values.append(ba)
                line += f"{ba:>10.4f}{acc:>10.4f}"
                csv_parts.append(f"{ba:.6f},{acc:.6f}")
            else:
                line += f"{'—':>10}{'—':>10}"
                csv_parts.append(",")

        if ba_values:
            mean_ba = np.mean(ba_values)
            std_ba = np.std(ba_values) if len(ba_values) > 1 else 0.0
            line += f"{mean_ba:>14.4f}{std_ba:>8.4f}"
            csv_parts.append(f"{mean_ba:.6f},{std_ba:.6f}")
        else:
            line += f"{'—':>14}{'—':>8}"
            csv_parts.append(",")

        print(line)
        csv_lines.append(",".join(csv_parts))

    print(sep + "\n")

    # Save CSV
    suffix = f"_{target_label}" if target_label else ""
    csv_path = os.path.join(output_dir, f"results_table{suffix}.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"[Saved] {csv_path}")

    return by_method, all_folds


# ========== 2. Feature Extraction Helper ==========

def _load_model_and_datasets(model_path, model_type, fold, batch_size=8,
                              num_workers=2, obc_metadata=None, dna_root=None,
                              target_dataset="driveandact", dmd_root=None,
                              mdad_root=None):
    """Load model and create source/target dataloaders. Requires GPU."""
    import torch
    from cross_dataset.train_sfda import load_source_model
    from cross_dataset.class_mapping_utils import get_num_unified_classes
    from cross_dataset.obc_dataset import OBCCrossDataset
    from torch.utils.data import DataLoader

    if obc_metadata is None:
        obc_metadata = OBC_METADATA
    if dna_root is None:
        dna_root = DNA_ROOT_DEFAULT
    if dmd_root is None:
        dmd_root = DMD_ROOT_DEFAULT
    if mdad_root is None:
        mdad_root = MDAD_ROOT_DEFAULT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = get_num_unified_classes()

    model = load_source_model(model_path, model_type, num_classes, device)
    model.eval()

    # Source dataset (OBC test)
    src_ds = OBCCrossDataset(obc_metadata, split="test", num_frames=16, augment=False)
    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Target dataset
    if target_dataset == "dmd":
        from cross_dataset.dmd_dataset import DMDDataset
        tgt_ds = DMDDataset(dmd_root, split="test", fold=fold,
                            num_frames=16, augment=False)
    elif target_dataset == "3mdad":
        from cross_dataset.mdad_dataset import MDADDataset
        tgt_ds = MDADDataset(mdad_root, split="test", fold=fold,
                             num_frames=16, augment=False)
    else:
        from cross_dataset.driveandact_dataset import DriveAndActDataset
        tgt_ds = DriveAndActDataset(dna_root, split="test", fold=fold,
                                     num_frames=16, augment=False)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return model, src_loader, tgt_loader, device


def extract_features(model, loader, device, model_type, max_samples=2000):
    """Extract 768-dim features, predicted labels, and true labels from model."""
    import torch
    from torch.amp import autocast
    from tqdm import tqdm

    all_features = []
    all_preds = []
    all_labels = []
    n_collected = 0

    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting features"):
            if n_collected >= max_samples:
                break
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda", enabled=True):
                if model_type in ("dann", "cdan", "mcc", "bnm"):
                    logits, _, features = model(imgs, 0.0)
                else:
                    # Standard model: need to get intermediate features
                    outputs = model.videomae(pixel_values=imgs)
                    seq_out = outputs.last_hidden_state
                    if model.fc_norm is not None:
                        features = model.fc_norm(seq_out.mean(1))
                    else:
                        features = seq_out[:, 0]
                    logits = model.classifier(features)

            preds = logits.argmax(dim=1)
            all_features.append(features.float().cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            n_collected += len(labels)

    features = np.concatenate(all_features, axis=0)[:max_samples]
    preds = np.concatenate(all_preds, axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]

    return features, preds, labels


# ========== 3. t-SNE Visualization ==========

def run_tsne(args):
    """Generate t-SNE plots colored by class and by domain."""
    from sklearn.manifold import TSNE
    from cross_dataset.class_mapping_utils import get_unified_class_names

    print("\n=== t-SNE Visualization ===")

    tgt_ds_name = getattr(args, "target_dataset", "driveandact") or "driveandact"
    tgt_display = TARGET_DISPLAY.get(tgt_ds_name, tgt_ds_name)
    tgt_suffix = f"_{tgt_ds_name}" if tgt_ds_name != "driveandact" else ""

    model, src_loader, tgt_loader, device = _load_model_and_datasets(
        args.model_path, args.model_type, args.fold,
        batch_size=args.batch_size, num_workers=args.num_workers,
        obc_metadata=args.obc_metadata, dna_root=args.dna_root,
        target_dataset=tgt_ds_name,
        dmd_root=getattr(args, "dmd_root", None),
        mdad_root=getattr(args, "mdad_root", None),
    )

    max_per_domain = args.tsne_max_samples // 2

    print("Extracting source features...")
    src_feats, src_preds, src_labels = extract_features(
        model, src_loader, device, args.model_type, max_samples=max_per_domain
    )
    print(f"  Source: {len(src_feats)} samples")

    print(f"Extracting target features ({tgt_display})...")
    tgt_feats, tgt_preds, tgt_labels = extract_features(
        model, tgt_loader, device, args.model_type, max_samples=max_per_domain
    )
    print(f"  Target: {len(tgt_feats)} samples")

    # Combine
    all_feats = np.concatenate([src_feats, tgt_feats], axis=0)
    all_labels = np.concatenate([src_labels, tgt_labels], axis=0)
    domain_labels = np.array(
        [0] * len(src_feats) + [1] * len(tgt_feats)
    )

    # Fit t-SNE
    print("Running t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                n_iter=1000, learning_rate="auto", init="pca")
    embeddings = tsne.fit_transform(all_feats)

    class_names = get_unified_class_names()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Plot 1: Colored by class ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    palette = sns.color_palette("husl", len(class_names))
    for i, cn in enumerate(class_names):
        mask = all_labels == i
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=[palette[i]], label=cn, alpha=0.5, s=15, edgecolors="none")
    ax.legend(fontsize=9, loc="best", markerscale=2)
    ax.set_title(f"t-SNE by Class — OBC→{tgt_display} ({args.model_type}, fold {args.fold})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    path1 = os.path.join(args.output_dir, f"tsne_class_{args.model_type}{tgt_suffix}_fold{args.fold}.png")
    fig.savefig(path1, dpi=200)
    plt.close(fig)
    print(f"[Saved] {path1}")

    # --- Plot 2: Colored by domain ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    domain_names = ["OBC (source)", f"{tgt_display} (target)"]
    colors = ["#1f77b4", "#ff7f0e"]
    for d in [0, 1]:
        mask = domain_labels == d
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=colors[d], label=domain_names[d], alpha=0.4, s=15, edgecolors="none")
    ax.legend(fontsize=11, loc="best", markerscale=2)
    ax.set_title(f"t-SNE by Domain — OBC→{tgt_display} ({args.model_type}, fold {args.fold})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    path2 = os.path.join(args.output_dir, f"tsne_domain_{args.model_type}{tgt_suffix}_fold{args.fold}.png")
    fig.savefig(path2, dpi=200)
    plt.close(fig)
    print(f"[Saved] {path2}")

    # --- Plot 3: Combined (class + domain shape) ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # Left: by class
    ax = axes[0]
    for i, cn in enumerate(class_names):
        mask = all_labels == i
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=[palette[i]], label=cn, alpha=0.5, s=15, edgecolors="none")
    ax.legend(fontsize=8, loc="best", markerscale=2)
    ax.set_title("By Class")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    # Right: by domain
    ax = axes[1]
    for d in [0, 1]:
        mask = domain_labels == d
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=colors[d], label=domain_names[d], alpha=0.4, s=15, edgecolors="none")
    ax.legend(fontsize=11, loc="best", markerscale=2)
    ax.set_title("By Domain")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.suptitle(f"Feature Visualization — OBC→{tgt_display} ({args.model_type}, fold {args.fold})", fontsize=14)
    plt.tight_layout()
    path3 = os.path.join(args.output_dir, f"tsne_combined_{args.model_type}{tgt_suffix}_fold{args.fold}.png")
    fig.savefig(path3, dpi=200)
    plt.close(fig)
    print(f"[Saved] {path3}")


# ========== 4. A-distance Estimation ==========

def run_adist(args):
    """Compute proxy A-distance using a linear SVM on source vs target features."""
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    print("\n=== Proxy A-distance Estimation ===")

    tgt_ds_name = getattr(args, "target_dataset", "driveandact") or "driveandact"
    tgt_display = TARGET_DISPLAY.get(tgt_ds_name, tgt_ds_name)
    tgt_suffix = f"_{tgt_ds_name}" if tgt_ds_name != "driveandact" else ""

    model, src_loader, tgt_loader, device = _load_model_and_datasets(
        args.model_path, args.model_type, args.fold,
        batch_size=args.batch_size, num_workers=args.num_workers,
        obc_metadata=args.obc_metadata, dna_root=args.dna_root,
        target_dataset=tgt_ds_name,
        dmd_root=getattr(args, "dmd_root", None),
        mdad_root=getattr(args, "mdad_root", None),
    )

    max_per_domain = 2000

    print("Extracting source features...")
    src_feats, _, _ = extract_features(
        model, src_loader, device, args.model_type, max_samples=max_per_domain
    )

    print(f"Extracting target features ({tgt_display})...")
    tgt_feats, _, _ = extract_features(
        model, tgt_loader, device, args.model_type, max_samples=max_per_domain
    )

    # Build domain classification dataset
    X = np.concatenate([src_feats, tgt_feats], axis=0)
    y = np.array([0] * len(src_feats) + [1] * len(tgt_feats))

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Shuffle
    perm = np.random.RandomState(42).permutation(len(X))
    X, y = X[perm], y[perm]

    # Train linear SVM with 5-fold cross-validation
    print("Training linear SVM for domain classification...")
    svm = LinearSVC(max_iter=5000, random_state=42)
    scores = cross_val_score(svm, X, y, cv=5, scoring="accuracy")

    domain_acc = scores.mean()
    # proxy A-distance = 2 * (1 - 2 * error) = 2 * (2 * acc - 1)
    proxy_a_distance = 2.0 * (2.0 * domain_acc - 1.0)

    print(f"\nDomain classification accuracy: {domain_acc:.4f} (+/- {scores.std():.4f})")
    print(f"Proxy A-distance: {proxy_a_distance:.4f}")
    print(f"  (0 = identical distributions, 2 = perfectly separable)")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    adist_results = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "target_dataset": tgt_ds_name,
        "fold": args.fold,
        "domain_accuracy": float(domain_acc),
        "domain_accuracy_std": float(scores.std()),
        "proxy_a_distance": float(proxy_a_distance),
        "n_source": int(len(src_feats)),
        "n_target": int(len(tgt_feats)),
    }
    out_path = os.path.join(args.output_dir, f"a_distance_{args.model_type}{tgt_suffix}_fold{args.fold}.json")
    with open(out_path, "w") as f:
        json.dump(adist_results, f, indent=2)
    print(f"[Saved] {out_path}")

    return proxy_a_distance


# ========== 5. Confusion Matrix Comparison ==========

def run_confmat(args):
    """Generate side-by-side confusion matrices from classification_report.txt files."""
    from cross_dataset.class_mapping_utils import get_unified_class_names

    print("\n=== Confusion Matrix Comparison ===")

    target_filter = getattr(args, "target", None)
    records = collect_results(target_dataset=target_filter)
    if not records:
        print("[ERROR] No results found.")
        return

    class_names = get_unified_class_names()
    os.makedirs(args.output_dir, exist_ok=True)

    # Find confusion matrix images or classification reports
    methods_with_cm = []
    for rec in records:
        cm_path = os.path.join(rec["dir_path"], "confusion_matrix.png")
        report_path = os.path.join(rec["dir_path"], "classification_report.txt")
        if os.path.exists(report_path):
            methods_with_cm.append(rec)

    # Select methods to compare (prefer fold 0 for comparability)
    target_fold = getattr(args, "fold", 0)
    selected = [r for r in methods_with_cm if r["fold"] == target_fold]
    if not selected:
        selected = methods_with_cm

    # Sort by method order
    def sort_key(r):
        try:
            return METHOD_ORDER.index(r["mode"])
        except ValueError:
            return 999

    selected.sort(key=sort_key)

    # Parse classification reports to extract per-class F1
    all_reports = {}
    for rec in selected:
        report_path = os.path.join(rec["dir_path"], "classification_report.txt")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report_text = f.read()
            all_reports[rec["mode"]] = _parse_classification_report(report_text, class_names)

    if not all_reports:
        print("[WARN] No classification reports found to parse.")
        return

    # If confusion_matrix.png exists, create a side-by-side comparison
    cm_images = []
    cm_labels = []
    for rec in selected:
        cm_path = os.path.join(rec["dir_path"], "confusion_matrix.png")
        if os.path.exists(cm_path):
            cm_images.append(cm_path)
            cm_labels.append(f"{rec['mode']} (fold {rec['fold']})\nBA={rec['test_balanced_accuracy']:.4f}")

    if cm_images:
        n = len(cm_images)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
        if n == 1:
            axes = [axes]
        elif rows > 1:
            axes = axes.flatten()

        for i, (img_path, label) in enumerate(zip(cm_images, cm_labels)):
            img = plt.imread(img_path)
            axes[i].imshow(img)
            axes[i].set_title(label, fontsize=10)
            axes[i].axis("off")

        # Hide unused axes
        for j in range(n, len(axes) if isinstance(axes, (list, np.ndarray)) else 1):
            if isinstance(axes, (list, np.ndarray)):
                axes[j].axis("off")

        tgt_label = target_filter or "all"
        fig.suptitle(f"Confusion Matrix Comparison ({tgt_label})", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"confmat_comparison_{tgt_label}_fold{target_fold}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Saved] {out_path}")
    else:
        print("[INFO] No confusion_matrix.png files found. Skipping image comparison.")


def _parse_classification_report(text, class_names):
    """Parse sklearn-style classification report text into per-class metrics dict."""
    metrics = {}
    for cn in class_names:
        # Match line like: "   phone_call     0.9744    0.8941    0.9325        85"
        pattern = re.escape(cn) + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
        match = re.search(pattern, text)
        if match:
            metrics[cn] = {
                "precision": float(match.group(1)),
                "recall": float(match.group(2)),
                "f1": float(match.group(3)),
                "support": int(match.group(4)),
            }
    return metrics


# ========== 6. Per-class Performance Bar Chart ==========

def run_perclass(args):
    """Generate per-class F1 bar chart comparing methods."""
    from cross_dataset.class_mapping_utils import get_unified_class_names

    print("\n=== Per-Class F1 Bar Chart ===")

    target_filter = getattr(args, "target", None)
    records = collect_results(target_dataset=target_filter)
    class_names = get_unified_class_names()
    os.makedirs(args.output_dir, exist_ok=True)

    target_fold = getattr(args, "fold", 0)
    selected = [r for r in records if r["fold"] == target_fold]
    if not selected:
        selected = records

    # Parse classification reports
    all_reports = {}
    for rec in selected:
        report_path = os.path.join(rec["dir_path"], "classification_report.txt")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report_text = f.read()
            parsed = _parse_classification_report(report_text, class_names)
            if parsed:
                all_reports[rec["mode"]] = parsed

    if not all_reports:
        print("[WARN] No classification reports with per-class metrics found.")
        return

    # Sort methods
    methods = sorted(all_reports.keys(),
                     key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 999)

    # Build data matrix: methods x classes
    n_methods = len(methods)
    n_classes = len(class_names)
    f1_matrix = np.zeros((n_methods, n_classes))

    for i, method in enumerate(methods):
        for j, cn in enumerate(class_names):
            if cn in all_reports[method]:
                f1_matrix[i, j] = all_reports[method][cn]["f1"]

    # Plot grouped bar chart
    x = np.arange(n_classes)
    width = 0.8 / n_methods
    palette = sns.color_palette("Set2", n_methods)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, method in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_matrix[i], width, label=method,
                      color=palette[i], edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, f1_matrix[i]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6, rotation=45)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([cn.replace("_", "\n") for cn in class_names],
                       fontsize=9, ha="center")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    tgt_label = target_filter or "all"
    # Fix title (was referencing tgt_label before assignment)
    ax.set_title(f"Per-Class F1 Comparison ({tgt_label}, fold {target_fold})", fontsize=14)
    out_path = os.path.join(args.output_dir, f"perclass_f1_{tgt_label}_fold{target_fold}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")

    # Also print table
    print(f"\n{'Class':<22}" + "".join(f"{m:>12}" for m in methods))
    print("-" * (22 + 12 * n_methods))
    for j, cn in enumerate(class_names):
        print(f"{cn:<22}" + "".join(f"{f1_matrix[i, j]:>12.4f}" for i in range(n_methods)))


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="Analysis and visualization for cross-dataset DA experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # Common output_dir default
    default_output_dir = os.path.join(RESULTS_DIR, "analysis")

    # Helper to add common args
    def _add_output_dir(sub):
        sub.add_argument("--output_dir", type=str, default=default_output_dir,
                         help="Directory to save analysis outputs")

    def _add_model_args(sub, model_required=True):
        sub.add_argument("--model_path", type=str, required=model_required,
                         help="Path to model checkpoint")
        sub.add_argument("--model_type", type=str, required=model_required,
                         choices=["dann", "cdan", "mcc", "bnm", "finetune", "target_only"],
                         help="Model type")
        sub.add_argument("--fold", type=int, default=0, help="Fold index")
        sub.add_argument("--target_dataset", type=str, default="driveandact",
                         choices=["driveandact", "dmd", "3mdad"],
                         help="Target dataset for feature extraction")
        sub.add_argument("--dmd_root", type=str, default=None)
        sub.add_argument("--mdad_root", type=str, default=None)
        sub.add_argument("--batch_size", type=int, default=8)
        sub.add_argument("--num_workers", type=int, default=2)
        sub.add_argument("--obc_metadata", type=str, default=None)
        sub.add_argument("--dna_root", type=str, default=None)

    # Common target_dataset arg for all subcommands
    def _add_target_arg(sub):
        sub.add_argument("--target", type=str, default=None,
                         choices=["driveandact", "dmd", "3mdad"],
                         help="Filter results by target dataset (default: all)")

    # --- table ---
    sub_table = subparsers.add_parser("table", help="Print comparison table and save CSV")
    _add_output_dir(sub_table)
    _add_target_arg(sub_table)

    # --- tsne ---
    sub_tsne = subparsers.add_parser("tsne", help="Generate t-SNE visualization")
    _add_model_args(sub_tsne)
    _add_output_dir(sub_tsne)
    sub_tsne.add_argument("--tsne_max_samples", type=int, default=2000,
                          help="Max total samples for t-SNE (split between source/target)")

    # --- adist ---
    sub_adist = subparsers.add_parser("adist", help="Compute proxy A-distance")
    _add_model_args(sub_adist)
    _add_output_dir(sub_adist)

    # --- confmat ---
    sub_confmat = subparsers.add_parser("confmat", help="Side-by-side confusion matrix comparison")
    sub_confmat.add_argument("--fold", type=int, default=0)
    _add_output_dir(sub_confmat)
    _add_target_arg(sub_confmat)

    # --- perclass ---
    sub_perclass = subparsers.add_parser("perclass", help="Per-class F1 bar chart")
    sub_perclass.add_argument("--fold", type=int, default=0)
    _add_output_dir(sub_perclass)
    _add_target_arg(sub_perclass)

    # --- all ---
    sub_all = subparsers.add_parser("all", help="Run all analyses")
    _add_model_args(sub_all)
    _add_output_dir(sub_all)
    _add_target_arg(sub_all)
    sub_all.add_argument("--tsne_max_samples", type=int, default=2000)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Resolve target filter
    target_filter = getattr(args, "target", None)

    if args.command == "table":
        records = collect_results(target_dataset=target_filter)
        if not records:
            print("[ERROR] No results found in", RESULTS_DIR)
            sys.exit(1)
        print_and_save_table(records, args.output_dir, target_label=target_filter)

    elif args.command == "tsne":
        run_tsne(args)

    elif args.command == "adist":
        run_adist(args)

    elif args.command == "confmat":
        run_confmat(args)

    elif args.command == "perclass":
        run_perclass(args)

    elif args.command == "all":
        # 1. Table
        records = collect_results(target_dataset=target_filter)
        if records:
            print_and_save_table(records, args.output_dir, target_label=target_filter)
        else:
            print("[WARN] No results found for table.")

        # 2. t-SNE
        run_tsne(args)

        # 3. A-distance
        run_adist(args)

        # 4. Confusion matrix comparison
        run_confmat(args)

        # 5. Per-class F1
        run_perclass(args)

        print(f"\n=== All analyses complete. Outputs in {args.output_dir} ===")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

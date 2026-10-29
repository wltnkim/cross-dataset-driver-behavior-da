"""Recompute Fig 2 (t-SNE) and proxy A-distance (main.tex Section VI-A) using the
TRUE UDA (zero-label, target_label_ratio=0.0) DANN checkpoints that fed Table 4 --
instead of the full-supervision (ratio=1.0) checkpoints analyze_results.py's default
MODEL_PATH previously used (see paper/fig2_adist_provenance_report_20260716.md).

Pipeline deviation from analyze_results.py's default (documented, not silent):
  TARGET-domain loader here uses MemmapVideoDataset (memmap256/), matching what
  these r000_uni checkpoints were actually trained/evaluated with (config.yaml has
  memmap_path + memmap_crop=224). analyze_results.py's default target loader
  (DriveAndActDataset/DMDDataset/MDADDataset, on-the-fly decord decoding) would not
  match the training-time preprocessing for these checkpoints.
  SOURCE (OBC) loader is unchanged (OBCCrossDataset) -- no OBC memmap exists, and
  train_cross.py itself only memmaps the target side for these runs (grep-confirmed
  in train_cross.py: memmap_path/memmap_metadata only feed the target dataset
  construction; source is always OBCCrossDataset regardless of memmap args).

Does not modify analyze_results.py, train_sfda.py, or main.tex. Reuses
extract_features() from analyze_results.py unmodified; replicates run_adist()'s and
run_tsne()'s Plot-2 (domain-colored) logic inline with the corrected target loader.

Usage (from Occupant_Behavior/code, on a GPU node):
    python -m cross_dataset.run_uda_tsne_adist_20260718
"""
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from cross_dataset.train_sfda import load_source_model
from cross_dataset.obc_dataset import OBCCrossDataset
from cross_dataset.memmap_dataset import MemmapVideoDataset
from cross_dataset.class_mapping_utils import get_num_unified_classes
from cross_dataset.analyze_results import extract_features, TARGET_DISPLAY

OBC_METADATA = "features/keyframe_video_metadata_dualcam.pkl"
MEMMAP_DIR = "memmap256"
OUT_DIR = "results/analysis_uda_20260718"
os.makedirs(OUT_DIR, exist_ok=True)

CASES = [
    dict(target="driveandact", fold=0, dir="results/20260607_175800_cross_dann_r000_uni_fold0"),
    dict(target="driveandact", fold=1, dir="results/20260607_181909_cross_dann_r000_uni_fold1"),
    dict(target="driveandact", fold=2, dir="results/20260607_182936_cross_dann_r000_uni_fold2"),
    dict(target="dmd", fold=0, dir="results/20260607_184703_cross_dann_dmd_r000_uni_fold0"),
    dict(target="dmd", fold=1, dir="results/20260607_185850_cross_dann_dmd_r000_uni_fold1"),
    dict(target="dmd", fold=2, dir="results/20260607_191438_cross_dann_dmd_r000_uni_fold2"),
    dict(target="3mdad", fold=0, dir="results/20260607_192858_cross_dann_3mdad_r000_uni_fold0"),
    dict(target="3mdad", fold=1, dir="results/20260607_193154_cross_dann_3mdad_r000_uni_fold1"),
    dict(target="3mdad", fold=2, dir="results/20260607_193524_cross_dann_3mdad_r000_uni_fold2"),
]


@torch.no_grad()
def eval_balanced_accuracy(model, loader, device):
    from sklearn.metrics import balanced_accuracy_score
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        logits, _, _ = model(imgs, 0.0)
        preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        labels.extend(lbls.numpy().tolist())
    return balanced_accuracy_score(labels, preds)


def compute_adist(src_feats, tgt_feats):
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([src_feats, tgt_feats], axis=0)
    y = np.array([0] * len(src_feats) + [1] * len(tgt_feats))
    X = StandardScaler().fit_transform(X)
    perm = np.random.RandomState(42).permutation(len(X))
    X, y = X[perm], y[perm]
    svm = LinearSVC(max_iter=5000, random_state=42)
    scores = cross_val_score(svm, X, y, cv=5, scoring="accuracy")
    domain_acc = scores.mean()
    proxy_a_distance = 2.0 * (2.0 * domain_acc - 1.0)
    return domain_acc, float(scores.std()), proxy_a_distance


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_unified_classes()

    summary = []
    for c in CASES:
        tgt, fold, ckdir = c["target"], c["fold"], c["dir"]
        ckpt = os.path.join(ckdir, "best_model.pt")
        cfg = yaml.safe_load(open(os.path.join(ckdir, "config.yaml")))
        assert cfg["mode"] == "dann" and cfg["target_label_ratio"] == 0.0, \
            f"{ckdir}: expected mode=dann, target_label_ratio=0.0, got {cfg.get('mode')}, {cfg.get('target_label_ratio')}"
        recorded = json.load(open(os.path.join(ckdir, "results.json")))["test_balanced_accuracy"]

        print(f"\n=== {tgt} fold{fold} ({ckdir}) ===", flush=True)

        entry = dict(target=tgt, fold=fold, dir=ckdir, recorded_test_balanced_accuracy=recorded)
        try:
            model = load_source_model(ckpt, "dann", num_classes, device)
        except Exception as e:
            print(f"[LOAD FAILED] {e}", flush=True)
            entry["load_status"] = f"FAILED: {e}"
            summary.append(entry)
            continue
        entry["load_status"] = "OK (strict=True, no exception)"

        src_ds = OBCCrossDataset(OBC_METADATA, split="test", num_frames=16, augment=False)
        src_loader = DataLoader(src_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

        tgt_ds = MemmapVideoDataset(
            memmap_path=f"{MEMMAP_DIR}/{tgt}_frames.npy",
            metadata_path=f"{MEMMAP_DIR}/{tgt}_metadata.json",
            fold=fold, split="test", num_frames=16, augment=False,
            crop_size=cfg.get("memmap_crop", 224),
        )
        tgt_loader = DataLoader(tgt_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

        measured_ba = eval_balanced_accuracy(model, tgt_loader, device)
        diff = measured_ba - recorded
        print(f"  recorded={recorded:.4f}  measured={measured_ba:.4f}  diff={diff:+.4f}", flush=True)
        entry["measured_test_balanced_accuracy"] = float(measured_ba)
        entry["diff_vs_recorded"] = float(diff)
        if measured_ba > 0.6:
            msg = f"ABORT-CHECK TRIGGERED: measured balanced accuracy {measured_ba:.4f} looks like full-supervision, not UDA"
            print(f"  [{msg}]", flush=True)
            entry["abort_check"] = msg
            summary.append(entry)
            continue

        # ---- A-distance (max_samples=2000/domain, matches run_adist) ----
        print("  Extracting features for A-distance (max 2000/domain)...", flush=True)
        src_feats_a, _, _ = extract_features(model, src_loader, device, "dann", max_samples=2000)
        tgt_feats_a, _, _ = extract_features(model, tgt_loader, device, "dann", max_samples=2000)
        domain_acc, domain_acc_std, proxy_a = compute_adist(src_feats_a, tgt_feats_a)
        print(f"  domain_acc={domain_acc:.4f} (+/-{domain_acc_std:.4f})  proxy_A_distance={proxy_a:.4f}", flush=True)

        adist_out = dict(
            model_path=ckpt, model_type="dann", target_dataset=tgt, fold=fold,
            target_label_ratio=0.0,
            pipeline="target=MemmapVideoDataset(memmap256, crop=%s) / source=OBCCrossDataset" % cfg.get("memmap_crop", 224),
            domain_accuracy=float(domain_acc), domain_accuracy_std=domain_acc_std,
            proxy_a_distance=float(proxy_a),
            n_source=int(len(src_feats_a)), n_target=int(len(tgt_feats_a)),
            recorded_test_balanced_accuracy=recorded,
            measured_test_balanced_accuracy=float(measured_ba),
        )
        with open(os.path.join(OUT_DIR, f"a_distance_uda_dann_{tgt}_fold{fold}.json"), "w") as f:
            json.dump(adist_out, f, indent=2)
        entry.update(adist_out)
        summary.append(entry)

        # ---- t-SNE domain plot, fold0 only (max_samples=1000/domain, matches run_tsne default) ----
        if fold == 0:
            from sklearn.manifold import TSNE
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            print("  Extracting features for t-SNE (max 1000/domain)...", flush=True)
            src_feats_t, _, _ = extract_features(model, src_loader, device, "dann", max_samples=1000)
            tgt_feats_t, _, _ = extract_features(model, tgt_loader, device, "dann", max_samples=1000)
            all_feats = np.concatenate([src_feats_t, tgt_feats_t], axis=0)
            domain_labels = np.array([0] * len(src_feats_t) + [1] * len(tgt_feats_t))

            print("  Running t-SNE (perplexity=30)...", flush=True)
            tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                        n_iter=1000, learning_rate="auto", init="pca")
            embeddings = tsne.fit_transform(all_feats)

            tgt_display = TARGET_DISPLAY.get(tgt, tgt)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            domain_names = ["OBC (source)", f"{tgt_display} (target)"]
            colors = ["#1f77b4", "#ff7f0e"]
            for d in [0, 1]:
                mask = domain_labels == d
                ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                           c=colors[d], label=domain_names[d], alpha=0.4, s=15, edgecolors="none")
            ax.legend(fontsize=11, loc="best", markerscale=2)
            # No in-figure title: the subpanel letter (a)/(b)/(c) and the LaTeX caption in
            # main.tex already state the target and method for each panel (removed 2026-09-18;
            # the previous debug-plot title duplicated that and leaked script-internal wording).
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            plt.tight_layout()
            pdf_path = os.path.join(OUT_DIR, f"tsne_domain_uda_dann_{tgt}_fold{fold}.pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  [Saved] {pdf_path}", flush=True)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== DONE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

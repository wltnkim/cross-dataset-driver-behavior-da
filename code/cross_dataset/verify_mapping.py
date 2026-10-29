"""Verify cross-dataset class mapping and print statistics.

Run:
    python -m cross_dataset.verify_mapping \
        --obc_metadata features/keyframe_video_metadata_dualcam.pkl \
        --dna_root <DATA_ROOT>/DriveAndAct
"""

import argparse
import csv
import os
import pickle
import sys
from collections import Counter

from cross_dataset.class_mapping_utils import (
    get_unified_label_maps,
    get_unified_class_names,
    get_num_unified_classes,
    load_class_mapping,
)


def verify_obc(metadata_path, mapping_path=None):
    """Check OBC class coverage."""
    print("=" * 60)
    print("OBC Dataset Verification")
    print("=" * 60)

    with open(metadata_path, "rb") as f:
        meta = pickle.load(f)

    mapping = load_class_mapping(mapping_path)
    obc_class_names = mapping["obc_class_names"]
    label_maps = get_unified_label_maps(mapping_path)
    obc_map = label_maps["obc"]
    unified_names = get_unified_class_names(mapping_path)

    labels = [s["label"] for s in meta["samples"]]
    label_counts = Counter(labels)

    total_mapped = 0
    total_unmapped = 0

    print(f"\nTotal samples: {len(meta['samples'])}")
    print(f"Total classes: {meta['num_classes']}")
    print(f"\n{'OBC Class':<20} {'Count':>6} {'Unified Label':<25}")
    print("-" * 55)

    for label_id in sorted(label_counts.keys()):
        cls_name = obc_class_names[label_id]
        count = label_counts[label_id]
        if cls_name in obc_map:
            unified = f"{obc_map[cls_name]}: {unified_names[obc_map[cls_name]]}"
            total_mapped += count
        else:
            unified = "--- (excluded)"
            total_unmapped += count
        print(f"{cls_name:<20} {count:>6} {unified:<25}")

    print(f"\nMapped: {total_mapped} samples ({total_mapped/len(labels)*100:.1f}%)")
    print(f"Excluded: {total_unmapped} samples ({total_unmapped/len(labels)*100:.1f}%)")

    # Per unified class
    print(f"\nUnified class distribution (OBC):")
    unified_counts = Counter()
    for s in meta["samples"]:
        cls_name = obc_class_names[s["label"]]
        if cls_name in obc_map:
            unified_counts[obc_map[cls_name]] += 1

    for uid in range(get_num_unified_classes(mapping_path)):
        print(f"  {uid}: {unified_names[uid]:<25} {unified_counts.get(uid, 0):>6}")


def verify_driveandact(data_root, mapping_path=None):
    """Check Drive&Act class coverage."""
    print("\n" + "=" * 60)
    print("Drive&Act Dataset Verification")
    print("=" * 60)

    mapping = load_class_mapping(mapping_path)
    label_maps = get_unified_label_maps(mapping_path)
    dna_map = label_maps["driveandact"]
    unified_names = get_unified_class_names(mapping_path)

    csv_path = os.path.join(data_root, mapping["driveandact_annotation_file"])
    if not os.path.exists(csv_path):
        print(f"  ERROR: Annotation file not found: {csv_path}")
        return

    all_activities = Counter()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_activities[row["activity"]] += 1

    total_mapped = 0
    total_unmapped = 0

    print(f"\nTotal chunks: {sum(all_activities.values())}")
    print(f"Total activities: {len(all_activities)}")
    print(f"\n{'Activity':<45} {'Count':>6} {'Unified Label':<25}")
    print("-" * 80)

    for act in sorted(all_activities.keys()):
        count = all_activities[act]
        if act in dna_map:
            unified = f"{dna_map[act]}: {unified_names[dna_map[act]]}"
            total_mapped += count
        else:
            unified = "--- (excluded)"
            total_unmapped += count
        print(f"{act:<45} {count:>6} {unified:<25}")

    total = sum(all_activities.values())
    print(f"\nMapped: {total_mapped} chunks ({total_mapped/total*100:.1f}%)")
    print(f"Excluded: {total_unmapped} chunks ({total_unmapped/total*100:.1f}%)")

    # Per unified class
    print(f"\nUnified class distribution (Drive&Act):")
    unified_counts = Counter()
    for act, count in all_activities.items():
        if act in dna_map:
            unified_counts[dna_map[act]] += count

    for uid in range(get_num_unified_classes(mapping_path)):
        print(f"  {uid}: {unified_names[uid]:<25} {unified_counts.get(uid, 0):>6}")

    # Per-split stats
    print(f"\nPer-split statistics:")
    for fold in range(3):
        for split in ["train", "val", "test"]:
            csv_rel = mapping["driveandact_splits"][f"split_{fold}"][split]
            csv_p = os.path.join(data_root, csv_rel)
            mapped = 0
            with open(csv_p) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["activity"] in dna_map:
                        mapped += 1
            print(f"  fold={fold} {split:5s}: {mapped:>5} mapped segments")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obc_metadata", type=str, default=None)
    parser.add_argument("--dna_root", type=str, default=None)
    parser.add_argument("--mapping_path", type=str, default=None)
    args = parser.parse_args()

    unified_names = get_unified_class_names(args.mapping_path)
    n_classes = get_num_unified_classes(args.mapping_path)

    print(f"Unified classes ({n_classes}):")
    for i, name in enumerate(unified_names):
        print(f"  {i}: {name}")

    if args.obc_metadata:
        verify_obc(args.obc_metadata, args.mapping_path)

    if args.dna_root:
        verify_driveandact(args.dna_root, args.mapping_path)


if __name__ == "__main__":
    main()

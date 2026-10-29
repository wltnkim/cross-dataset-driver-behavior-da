#!/usr/bin/env python3
"""Generate 3MDAD annotation CSVs for cross-dataset experiments.

Scans Day/RGB1 frame directories, maps AC numbers to unified activity names,
and outputs per-fold train/val/test CSVs.

3MDAD has 50 subjects × 16 action classes, each as a short clip (~100-164 frames
at 30fps). Only 9 action classes (AC3-8, AC13-15) map to our 5 unified classes.

Usage:
    python -m cross_dataset.generate_3mdad_annotations \
        --data_root <DATA_ROOT>/MDAD/MDAD
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cross_dataset.class_mapping_utils import get_unified_label_maps, get_unified_class_names


# ── 3MDAD 16 action classes (Jegham et al., 2020) ─────────────────────────
# AC1:  Safe driving          AC9:  Having picture
# AC2:  Hair and makeup       AC10: Talking to passenger
# AC3:  Adjusting radio       AC11: Singing or dancing
# AC4:  GPS operating         AC12: Fatigue and somnolence
# AC5:  Texting (right)       AC13: Drinking (right)
# AC6:  Texting (left)        AC14: Drinking (left)
# AC7:  Phone call (right)    AC15: Reaching behind
# AC8:  Phone call (left)     AC16: Smoking

AC_TO_ACTIVITY = {
    3: "adjusting_radio",
    4: "gps_operating",
    5: "texting_right",
    6: "texting_left",
    7: "phone_call_right",
    8: "phone_call_left",
    13: "drinking_right",
    14: "drinking_left",
    15: "reaching_behind",
}

# Excluded ACs (not in unified classes):
# AC1 (safe driving), AC2 (hair/makeup), AC9 (picture), AC10 (passenger),
# AC11 (singing), AC12 (fatigue), AC16 (smoking)


# ── Subject-disjoint 3-fold splits ──────────────────────────────────────────
# 50 subjects split into 3 roughly equal groups, rotate train/val/test.
SUBJECT_GROUP_A = list(range(1, 18))    # S1-S17  (17 subjects)
SUBJECT_GROUP_B = list(range(18, 35))   # S18-S34 (17 subjects)
SUBJECT_GROUP_C = list(range(35, 51))   # S35-S50 (16 subjects)

FOLD_SPLITS = {
    0: {"train": SUBJECT_GROUP_A, "val": SUBJECT_GROUP_B, "test": SUBJECT_GROUP_C},
    1: {"train": SUBJECT_GROUP_B, "val": SUBJECT_GROUP_C, "test": SUBJECT_GROUP_A},
    2: {"train": SUBJECT_GROUP_C, "val": SUBJECT_GROUP_A, "test": SUBJECT_GROUP_B},
}


def discover_samples(data_root):
    """Scan Day/RGB1 frame directories and build sample list.

    Returns:
        List of dicts: {participant_id, frame_dir, activity, num_frames}
    """
    rgb1_dir = os.path.join(data_root, "Day", "RGB1")
    if not os.path.isdir(rgb1_dir):
        raise FileNotFoundError(f"RGB1 directory not found: {rgb1_dir}")

    samples = []
    for subj_name in sorted(os.listdir(rgb1_dir)):
        m = re.match(r"S(\d+)$", subj_name)
        if not m:
            continue
        subj_id = int(m.group(1))
        subj_dir = os.path.join(rgb1_dir, subj_name)
        if not os.path.isdir(subj_dir):
            continue

        for ac_name in sorted(os.listdir(subj_dir)):
            m2 = re.match(r"AC(\d+)$", ac_name)
            if not m2:
                continue
            ac_num = int(m2.group(1))

            if ac_num not in AC_TO_ACTIVITY:
                continue

            frame_dir = os.path.join(subj_dir, ac_name)
            if not os.path.isdir(frame_dir):
                continue

            # Count JPG frames
            frames = [f for f in os.listdir(frame_dir) if f.endswith(".jpg")]
            num_frames = len(frames)
            if num_frames < 4:
                print(f"  [WARN] Too few frames ({num_frames}): {frame_dir}")
                continue

            # Relative path from data_root
            rel_dir = os.path.relpath(frame_dir, data_root)

            samples.append({
                "participant_id": subj_id,
                "frame_dir": rel_dir,
                "activity": AC_TO_ACTIVITY[ac_num],
                "num_frames": num_frames,
            })

    return samples


def write_csvs(samples, data_root):
    """Write per-fold train/val/test CSV files."""
    out_dir = os.path.join(data_root, "annotations")
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["participant_id", "frame_dir", "activity", "num_frames"]

    for fold in range(3):
        split_subjects = FOLD_SPLITS[fold]
        for split_name, subjects in split_subjects.items():
            subject_set = set(subjects)
            split_samples = [s for s in samples if s["participant_id"] in subject_set]

            csv_name = f"3mdad.split_{fold}.{split_name}.csv"
            csv_path = os.path.join(out_dir, csv_name)

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(split_samples)

            print(f"  Written: {csv_path}  ({len(split_samples)} samples)")


def print_statistics(samples):
    """Print detailed statistics."""
    class_names = get_unified_class_names()
    label_maps = get_unified_label_maps()
    activity_to_label = label_maps["threeMDAD"]

    print("\n" + "=" * 70)
    print("3MDAD Annotation Statistics")
    print("=" * 70)

    total = len(samples)
    print(f"\nTotal samples: {total}")

    # By activity
    activity_counts = Counter(s["activity"] for s in samples)
    print(f"\nSamples by 3MDAD activity:")
    for act, count in sorted(activity_counts.items(), key=lambda x: -x[1]):
        ulabel = activity_to_label[act]
        print(f"  {act:25s} -> {class_names[ulabel]:20s} (label={ulabel}): {count:4d}")

    # By unified class
    label_counts = Counter(activity_to_label[s["activity"]] for s in samples)
    print(f"\nSamples by unified class:")
    for label in sorted(label_counts.keys()):
        print(f"  [{label}] {class_names[label]:20s}: {label_counts[label]:4d}")

    # Frame stats
    frame_counts = [s["num_frames"] for s in samples]
    print(f"\nFrames per clip: min={min(frame_counts)}, max={max(frame_counts)}, "
          f"mean={sum(frame_counts)/len(frame_counts):.1f}")

    # By subject
    subj_counts = Counter(s["participant_id"] for s in samples)
    print(f"\nSubjects: {len(subj_counts)} (each has {total // len(subj_counts)} samples)")

    # Per-fold statistics
    for fold in range(3):
        print(f"\n--- Fold {fold} ---")
        for split_name in ["train", "val", "test"]:
            subject_set = set(FOLD_SPLITS[fold][split_name])
            split_samples = [s for s in samples if s["participant_id"] in subject_set]
            split_labels = Counter(activity_to_label[s["activity"]] for s in split_samples)
            parts = [f"{class_names[l]}={split_labels.get(l, 0)}"
                     for l in sorted(split_labels.keys())]
            print(f"  {split_name:5s}: {len(split_samples):4d} samples  "
                  f"({len(subject_set)} subjects)  [{', '.join(parts)}]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3MDAD annotation CSVs for cross-dataset experiments")
    parser.add_argument("--data_root", type=str,
                        default="<DATA_ROOT>/MDAD/MDAD",
                        help="Path to 3MDAD data root (contains Day/, Night/, Annotations 3MDAD/)")
    args = parser.parse_args()

    print(f"Data root: {args.data_root}")

    # Discover samples
    print("\nScanning frame directories...")
    samples = discover_samples(args.data_root)
    print(f"Found {len(samples)} samples (mapped action classes only)")

    # Print statistics
    print_statistics(samples)

    # Write CSVs
    print("\nWriting CSV files...")
    write_csvs(samples, args.data_root)

    print("\nDone!")


if __name__ == "__main__":
    main()

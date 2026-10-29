#!/usr/bin/env python3
"""Generate DMD annotation CSVs for cross-dataset experiments.

Scans all DMD distraction sessions, parses OpenLABEL annotations,
chunks mappable actions into ~3-second (90-frame) segments, and
outputs per-fold train/val/test CSVs matching the Drive&Act format.

Usage:
    python -m cross_dataset.generate_dmd_annotations \
        --data_root <DATA_ROOT>/DMD/distraction/dmd

Output:
    {data_root}/annotations_3s/dmd.chunks_90.split_{fold}.{split}.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cross_dataset.class_mapping_utils import get_unified_label_maps, get_unified_class_names


# ── Subject-disjoint 3-fold splits ──────────────────────────────────────────
# 31 subjects sorted: 1-5 (gA), 6-10 (gB), 11-15 (gC), 22-25 (gF), 26-30 (gE), 31-37 (gZ)
# Split into 3 roughly equal groups (~10-11 each), rotate train/val/test.
#   Group A: subjects 1-10   (gA + gB, 10 subjects)
#   Group B: subjects 11-15, 22-25  (gC + gF, 9 subjects)
#   Group C: subjects 26-30, 31-37  (gE + gZ, 12 subjects)
#
# Fold 0: train=A(10), val=B(9),  test=C(12)
# Fold 1: train=B(9),  val=C(12), test=A(10)
# Fold 2: train=C(12), val=A(10), test=B(9)

SUBJECT_GROUP_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SUBJECT_GROUP_B = [11, 12, 13, 14, 15, 22, 23, 24, 25]
SUBJECT_GROUP_C = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]

FOLD_SPLITS = {
    0: {"train": SUBJECT_GROUP_A, "val": SUBJECT_GROUP_B, "test": SUBJECT_GROUP_C},
    1: {"train": SUBJECT_GROUP_B, "val": SUBJECT_GROUP_C, "test": SUBJECT_GROUP_A},
    2: {"train": SUBJECT_GROUP_C, "val": SUBJECT_GROUP_A, "test": SUBJECT_GROUP_B},
}

CHUNK_SIZE = 90  # ~3 seconds at 30fps


def discover_sessions(data_root):
    """Discover all DMD sessions with annotation and body video files.

    Returns:
        List of dicts: {group, subject_id, session, ann_path, video_path}
    """
    sessions = []
    valid_groups = {"gA", "gB", "gC", "gE", "gF", "gZ"}
    for group in sorted(os.listdir(data_root)):
        if group not in valid_groups:
            continue
        group_dir = os.path.join(data_root, group)
        if not os.path.isdir(group_dir):
            continue
        for subj in sorted(os.listdir(group_dir), key=lambda x: int(x)):
            subj_dir = os.path.join(group_dir, subj)
            if not os.path.isdir(subj_dir):
                continue
            for sess in sorted(os.listdir(subj_dir)):
                sess_dir = os.path.join(subj_dir, sess)
                if not os.path.isdir(sess_dir):
                    continue

                # Find all annotation+video pairs (some sessions have
                # multiple recordings with different timestamps)
                ann_files = {}
                video_files = {}
                for fname in os.listdir(sess_dir):
                    # Extract timestamp prefix: {group}_{subj}_{sess}_{timestamp}
                    if fname.endswith("_rgb_ann_distraction.json"):
                        prefix = fname.replace("_rgb_ann_distraction.json", "")
                        ann_files[prefix] = os.path.join(sess_dir, fname)
                    elif fname.endswith("_rgb_body.mp4"):
                        prefix = fname.replace("_rgb_body.mp4", "")
                        video_files[prefix] = os.path.join(sess_dir, fname)

                # Match annotation with video by prefix
                matched = 0
                for prefix in sorted(ann_files.keys()):
                    if prefix not in video_files:
                        print(f"  [WARN] No rgb_body video for: {ann_files[prefix]}")
                        continue
                    sessions.append({
                        "group": group,
                        "subject_id": int(subj),
                        "session": sess,
                        "ann_path": ann_files[prefix],
                        "video_path": video_files[prefix],
                    })
                    matched += 1

                if matched == 0:
                    print(f"  [WARN] No matched annotation+video: {sess_dir}")
    return sessions


def parse_and_chunk(sessions, action_to_label, data_root):
    """Parse annotations and chunk mapped actions into ~90-frame segments.

    Returns:
        List of dicts with CSV row data.
    """
    samples = []
    chunk_id_counter = 0

    for sess_info in sessions:
        with open(sess_info["ann_path"]) as f:
            data = json.load(f)

        actions = data["openlabel"]["actions"]

        # Make video path relative to data_root
        video_rel = os.path.relpath(sess_info["video_path"], data_root)
        # Strip .mp4 extension for file_id (matching D&A convention)
        file_id = video_rel.replace(".mp4", "")

        for action_id, action_info in actions.items():
            action_type = action_info["type"]
            if action_type not in action_to_label:
                continue

            unified_label = action_to_label[action_type]

            for interval in action_info.get("frame_intervals", []):
                fs = interval["frame_start"]
                fe = interval["frame_end"]
                seg_len = fe - fs + 1  # inclusive end

                if seg_len < 4:
                    continue

                # Chunk into ~CHUNK_SIZE segments
                pos = fs
                while pos < fe + 1:
                    chunk_end = min(pos + CHUNK_SIZE, fe + 1)
                    if chunk_end - pos < 4:
                        break

                    samples.append({
                        "participant_id": sess_info["subject_id"],
                        "file_id": file_id,
                        "annotation_id": action_id,
                        "frame_start": pos,
                        "frame_end": chunk_end,
                        "activity": action_type,
                        "chunk_id": chunk_id_counter,
                    })
                    chunk_id_counter += 1
                    pos += CHUNK_SIZE

    return samples


def write_csvs(samples, data_root):
    """Write per-fold train/val/test CSV files."""
    out_dir = os.path.join(data_root, "annotations_3s")
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["participant_id", "file_id", "annotation_id",
                   "frame_start", "frame_end", "activity", "chunk_id"]

    for fold in range(3):
        split_subjects = FOLD_SPLITS[fold]
        for split_name, subjects in split_subjects.items():
            subject_set = set(subjects)
            split_samples = [s for s in samples if s["participant_id"] in subject_set]

            csv_name = f"dmd.chunks_90.split_{fold}.{split_name}.csv"
            csv_path = os.path.join(out_dir, csv_name)

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(split_samples)

            print(f"  Written: {csv_path}  ({len(split_samples)} segments)")


def print_statistics(samples, action_to_label):
    """Print detailed statistics."""
    class_names = get_unified_class_names()

    print("\n" + "=" * 70)
    print("DMD Annotation Statistics")
    print("=" * 70)

    # Overall
    total = len(samples)
    print(f"\nTotal segments: {total}")

    # By action type
    action_counts = Counter(s["activity"] for s in samples)
    print(f"\nSegments by DMD action type:")
    for act, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        ulabel = action_to_label[act]
        print(f"  {act:45s} -> {class_names[ulabel]:20s} (label={ulabel}): {count:5d}")

    # By unified class
    label_counts = Counter(action_to_label[s["activity"]] for s in samples)
    print(f"\nSegments by unified class:")
    for label in sorted(label_counts.keys()):
        print(f"  [{label}] {class_names[label]:20s}: {label_counts[label]:5d}")

    # By subject
    subj_counts = Counter(s["participant_id"] for s in samples)
    print(f"\nSegments by subject ({len(subj_counts)} subjects):")
    for subj in sorted(subj_counts.keys()):
        print(f"  Subject {subj:2d}: {subj_counts[subj]:5d}")

    # Per-fold statistics
    for fold in range(3):
        print(f"\n--- Fold {fold} ---")
        for split_name in ["train", "val", "test"]:
            subject_set = set(FOLD_SPLITS[fold][split_name])
            split_samples = [s for s in samples if s["participant_id"] in subject_set]
            split_labels = Counter(action_to_label[s["activity"]] for s in split_samples)
            parts = [f"{class_names[l]}={split_labels.get(l, 0)}"
                     for l in range(len(class_names))]
            print(f"  {split_name:5s}: {len(split_samples):5d} segments  "
                  f"({len(subject_set)} subjects)  [{', '.join(parts)}]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DMD annotation CSVs for cross-dataset experiments")
    parser.add_argument("--data_root", type=str,
                        default="<DATA_ROOT>/DMD/distraction/dmd",
                        help="Path to DMD distraction data root")
    parser.add_argument("--mapping_path", type=str, default=None,
                        help="Path to class_mapping.json (default: auto)")
    args = parser.parse_args()

    print(f"Data root: {args.data_root}")

    # Get DMD action -> unified label mapping
    label_maps = get_unified_label_maps(args.mapping_path)
    action_to_label = label_maps["dmd"]
    print(f"Mapped DMD actions: {list(action_to_label.keys())}")

    # Discover sessions
    print("\nDiscovering sessions...")
    sessions = discover_sessions(args.data_root)
    print(f"Found {len(sessions)} sessions with annotation + rgb_body video")

    # Parse and chunk
    print("\nParsing annotations and chunking...")
    samples = parse_and_chunk(sessions, action_to_label, args.data_root)
    print(f"Generated {len(samples)} segments")

    # Print statistics
    print_statistics(samples, action_to_label)

    # Write CSVs
    print("\nWriting CSV files...")
    write_csvs(samples, args.data_root)

    print("\nDone!")


if __name__ == "__main__":
    main()

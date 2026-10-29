"""Generate CSV annotation files for UNITE/TAMT compatibility.

Both UNITE and TAMT expect CSV format: video_path,label (integer)

For OBC: frames are JPEGs in directories, NOT videos.
  -> Creates a CSV listing frame directories with labels.
  -> Alternative: convert to video format later if needed.

For Drive&Act: MP4 videos with frame_start/frame_end segments.
  -> Creates CSV with video_path,start_frame,end_frame,label

Usage:
    python -m cross_dataset.generate_csv_annotations \
        --obc_metadata features/keyframe_video_metadata_dualcam.pkl \
        --dna_root <DATA_ROOT>/DriveAndAct \
        --output_dir cross_dataset/annotations
"""

import argparse
import csv
import os
import pickle

import numpy as np

from cross_dataset.class_mapping_utils import (
    get_unified_label_maps,
    load_class_mapping,
)


def generate_obc_csv(metadata_path, output_dir, seed=42, mapping_path=None):
    """Generate OBC CSV annotation files (train/val/test).

    Format per line: frame_dir_path,label
    frame_dir_path is the parent directory of frame_paths[0].
    """
    with open(metadata_path, "rb") as f:
        meta = pickle.load(f)

    mapping = load_class_mapping(mapping_path)
    obc_class_names = mapping["obc_class_names"]
    label_maps = get_unified_label_maps(mapping_path)
    obc_map = label_maps["obc"]

    subjects = [s["subject"] for s in meta["samples"]]
    unique_subjects = sorted(set(subjects))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_subjects)

    train_subjects = set(unique_subjects[:26])
    val_subjects = set(unique_subjects[26:34])
    test_subjects = set(unique_subjects[34:])

    splits = {"train": train_subjects, "val": val_subjects, "test": test_subjects}

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_subjects in splits.items():
        rows = []
        for sample in meta["samples"]:
            if sample["subject"] not in split_subjects:
                continue
            obc_name = obc_class_names[sample["label"]]
            if obc_name not in obc_map:
                continue
            unified_label = obc_map[obc_name]

            # Use parent directory of first frame as "video path"
            frame_dir = os.path.dirname(sample["frame_paths"][0])
            rows.append((frame_dir, unified_label))

        csv_path = os.path.join(output_dir, f"obc_{split_name}.csv")
        with open(csv_path, "w", newline="") as f:
            for path, label in rows:
                f.write(f"{path},{label}\n")

        print(f"[OBC] {split_name}: {len(rows)} samples -> {csv_path}")


def generate_driveandact_csv(data_root, output_dir, mapping_path=None):
    """Generate Drive&Act CSV annotation files for all 3 folds."""
    mapping = load_class_mapping(mapping_path)
    label_maps = get_unified_label_maps(mapping_path)
    dna_map = label_maps["driveandact"]

    os.makedirs(output_dir, exist_ok=True)

    for fold in range(3):
        for split in ["train", "val", "test"]:
            csv_rel = mapping["driveandact_splits"][f"split_{fold}"][split]
            src_path = os.path.join(data_root, csv_rel)

            rows = []
            with open(src_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    activity = row["activity"]
                    if activity not in dna_map:
                        continue
                    unified_label = dna_map[activity]

                    video_path = os.path.join(
                        data_root, "kinect_color", row["file_id"] + ".mp4"
                    )
                    frame_start = int(row["frame_start"])
                    frame_end = int(row["frame_end"])

                    if frame_end - frame_start < 4:
                        continue

                    # Extended format: path,label,start,end
                    rows.append((video_path, unified_label, frame_start, frame_end))

            # Standard CSV (video_path,label) — for tools that don't support segments
            csv_simple = os.path.join(output_dir, f"dna_fold{fold}_{split}.csv")
            with open(csv_simple, "w", newline="") as f:
                for path, label, start, end in rows:
                    f.write(f"{path},{label}\n")

            # Extended CSV (video_path,label,start_frame,end_frame)
            csv_ext = os.path.join(output_dir, f"dna_fold{fold}_{split}_segments.csv")
            with open(csv_ext, "w", newline="") as f:
                for path, label, start, end in rows:
                    f.write(f"{path},{label},{start},{end}\n")

            print(f"[D&A] fold={fold} {split}: {len(rows)} segments -> {csv_simple}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obc_metadata", type=str,
                        default="features/keyframe_video_metadata_dualcam.pkl")
    parser.add_argument("--dna_root", type=str,
                        default="<DATA_ROOT>/DriveAndAct")
    parser.add_argument("--output_dir", type=str,
                        default="cross_dataset/annotations")
    parser.add_argument("--mapping_path", type=str, default=None)
    args = parser.parse_args()

    generate_obc_csv(args.obc_metadata, args.output_dir, mapping_path=args.mapping_path)
    generate_driveandact_csv(args.dna_root, args.output_dir, mapping_path=args.mapping_path)

    print(f"\nAll annotation CSVs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

"""Pre-decode MP4 segments → uint8 memmap (Option D, 2026-05-30).

For each unique (video_path, frame_start, frame_end) segment across all folds,
decode 16 frames uniformly and store as uint8 [T, H, W, C] in a numpy memmap.
Eliminates runtime decord I/O bottleneck.

Outputs:
  {output_dir}/{target}_frames.npy        : np.memmap (N, T, H, W, C) uint8
  {output_dir}/{target}_frames.npy.shape  : "N,T,H,W,C" ascii (for mmap reopen)
  {output_dir}/{target}_metadata.json     : list[dict] with:
      {video_id, frame_start, frame_end, unified_label,
       fold_assignment: {"0":"train"|"test"|"none", "1":..., "2":...}}

Usage:
  python -m cross_dataset.preprocess_to_memmap \
    --target dmd \
    --data_root <DATA_ROOT>/DMD/distraction/dmd \
    --output_dir <DATA_ROOT>/UMTRI/Occupant_Behavior/code/memmap \
    --num_frames 16 --resize 224
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import decord
import numpy as np
from PIL import Image

from cross_dataset.class_mapping_utils import get_unified_label_maps, load_class_mapping


def get_csv_paths(target, data_root, mapping):
    """Return dict {fold: {'train': path, 'test': path}} for all splits."""
    splits = {}
    key = {"dmd": "dmd_splits",
           "driveandact": None,  # special
           "3mdad": "threeMDAD_splits"}[target]
    if target == "driveandact":
        # D&A CSVs are at fixed path
        for fold in [0, 1, 2]:
            base = "activities_3s/kinect_color"
            splits[fold] = {
                "train": os.path.join(data_root, base, f"midlevel.chunks_90.split_{fold}.train.csv"),
                "test": os.path.join(data_root, base, f"midlevel.chunks_90.split_{fold}.test.csv"),
            }
    else:
        for fold in [0, 1, 2]:
            split_key = f"split_{fold}"
            entry = mapping[key][split_key]
            splits[fold] = {
                "train": os.path.join(data_root, entry["train"]),
                "test": os.path.join(data_root, entry["test"]),
            }
            # 3MDAD ships a native subject-disjoint val split (D&A/DMD do not);
            # include it so the memmap has a leak-free val for model selection,
            # matching the on-the-fly 3MDAD runs (mdad_dataset split="val").
            if "val" in entry:
                splits[fold]["val"] = os.path.join(data_root, entry["val"])
    return splits


def read_segments_from_csv(target, csv_path, data_root, activity_to_label):
    """Return list of dicts (video_path, frame_start, frame_end, unified_label)."""
    out = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            activity = row.get("activity") or row.get("activity_class")
            if activity is None or activity not in activity_to_label:
                continue
            unified_label = activity_to_label[activity]

            if target == "dmd":
                video_path = os.path.join(data_root, row["file_id"] + ".mp4")
                fs = int(row["frame_start"])
                fe = int(row["frame_end"])
            elif target == "driveandact":
                # file_id includes kinect_color/ prefix per D&A loader convention
                fid = row["file_id"]
                if not fid.startswith("kinect_color"):
                    fid = os.path.join("kinect_color", fid)
                video_path = os.path.join(data_root, fid + ".mp4")
                fs = int(row["frame_start"])
                fe = int(row["frame_end"])
            elif target == "3mdad":
                # 3MDAD: one action per video. Derive path from frame_dir exactly as
                # mdad_dataset.py ("Day/RGB1/S{s}/AC{a}" -> "videos/S{s}_AC{a}.mp4") and
                # use the whole clip (CSV has num_frames, no frame_start/frame_end;
                # decode_segment clamps fe to the real video length).
                mobj = re.search(r"S(\d+)/AC(\d+)$", row["frame_dir"])
                if not mobj:
                    continue
                video_path = os.path.join(
                    data_root, "videos",
                    f"S{int(mobj.group(1))}_AC{int(mobj.group(2))}.mp4")
                fs = 0
                fe = int(row["num_frames"])
            else:
                raise ValueError(target)
            if fe - fs < 4:
                continue
            if not os.path.isfile(video_path):
                continue

            out.append({
                "video_path": video_path,
                "frame_start": fs,
                "frame_end": fe,
                "unified_label": unified_label,
            })
    return out


def collect_unique_segments(target, data_root, mapping):
    """Walk all fold/split CSVs, dedupe segments, attach fold_assignment."""
    label_maps = get_unified_label_maps()
    activity_to_label = label_maps[target if target != "3mdad" else "threeMDAD"]

    csv_paths = get_csv_paths(target, data_root, mapping)

    # key: (video_path, frame_start, frame_end) → {label, fold_assignment}
    bucket = {}
    for fold, splits in csv_paths.items():
        for split, path in splits.items():
            segs = read_segments_from_csv(target, path, data_root, activity_to_label)
            for s in segs:
                key = (s["video_path"], s["frame_start"], s["frame_end"])
                if key not in bucket:
                    bucket[key] = {
                        "video_path": s["video_path"],
                        "frame_start": s["frame_start"],
                        "frame_end": s["frame_end"],
                        "unified_label": s["unified_label"],
                        "fold_assignment": {"0": "none", "1": "none", "2": "none"},
                    }
                bucket[key]["fold_assignment"][str(fold)] = split

    items = list(bucket.values())
    print(f"[{target}] unique segments: {len(items)}", flush=True)
    return items


def decode_segment(item, num_frames, resize):
    """Decode one segment → uint8 numpy [T, H, W, C]."""
    vr = decord.VideoReader(item["video_path"], num_threads=1)
    total = len(vr)
    fs = max(0, min(item["frame_start"], total - 1))
    fe = max(fs + 1, min(item["frame_end"], total))
    seg_len = fe - fs
    if seg_len >= num_frames:
        indices = np.linspace(fs, fe - 1, num_frames, dtype=int)
    else:
        indices = np.linspace(fs, fe - 1, seg_len, dtype=int)
        indices = np.pad(indices, (0, num_frames - seg_len), mode="edge")

    frames = vr.get_batch(indices.tolist()).asnumpy()  # [T, H, W, C] uint8
    del vr  # release decord handle immediately

    # Resize each frame to (resize, resize) using PIL (BILINEAR + antialias)
    out = np.empty((num_frames, resize, resize, 3), dtype=np.uint8)
    for t in range(num_frames):
        img = Image.fromarray(frames[t])
        img = img.resize((resize, resize), Image.BILINEAR)
        out[t] = np.array(img, dtype=np.uint8)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=["dmd", "driveandact", "3mdad"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--mapping_path", default=None)
    p.add_argument("--limit", type=int, default=-1,
                   help="for debug: process only first N segments")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mapping = load_class_mapping(args.mapping_path)
    items = collect_unique_segments(args.target, args.data_root, mapping)
    if args.limit > 0:
        items = items[: args.limit]

    N = len(items)
    T, H, W, C = args.num_frames, args.resize, args.resize, 3
    frames_path = os.path.join(args.output_dir, f"{args.target}_frames.npy")
    shape_path = frames_path + ".shape"
    meta_path = os.path.join(args.output_dir, f"{args.target}_metadata.json")

    print(f"[{args.target}] allocating memmap: {N} × {T} × {H} × {W} × {C} "
          f"= {N * T * H * W * C / 2**30:.2f} GB", flush=True)
    mm = np.memmap(frames_path, dtype=np.uint8, mode="w+", shape=(N, T, H, W, C))

    # Process. Periodically flush + report.
    failed = []
    for i, item in enumerate(items):
        try:
            mm[i] = decode_segment(item, args.num_frames, args.resize)
        except Exception as e:
            print(f"[{args.target}] FAIL idx={i} {item['video_path']}: {e}", flush=True)
            failed.append(i)
            # zero-fill the failed slot (already zero from w+ allocation)
        if (i + 1) % 200 == 0:
            mm.flush()
            print(f"[{args.target}] {i+1}/{N} ({(i+1)/N*100:.1f}%) "
                  f"failed={len(failed)}", flush=True)

    mm.flush()
    del mm

    # Write shape sidecar for memmap_dataset reopen
    with open(shape_path, "w") as f:
        f.write(",".join(str(x) for x in (N, T, H, W, C)))

    # Drop failed entries from metadata so dataset doesn't serve them
    final_meta = [items[i] for i in range(N) if i not in set(failed)]
    # NOTE: index in frames.npy MUST match index in metadata. Keep all items.
    # Failed slots are zero arrays — downstream training will see black frames.
    # Trade-off: simpler indexing. Document failed count separately.
    with open(meta_path, "w") as f:
        json.dump(items, f)
    with open(meta_path + ".failed", "w") as f:
        json.dump(failed, f)

    print(f"[{args.target}] DONE. N={N}, failed={len(failed)}", flush=True)
    print(f"  frames: {frames_path}")
    print(f"  shape:  {shape_path}")
    print(f"  meta:   {meta_path}")
    print(f"  failed: {meta_path}.failed ({len(failed)} indices)")


if __name__ == "__main__":
    main()

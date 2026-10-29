#!/usr/bin/env python3
"""Convert 3MDAD frame directories to MP4 videos for fast decord loading.

Scans Day/RGB1/S*/AC*/ directories and converts each to a single MP4 file
using ffmpeg. Only converts action classes that map to unified labels.

Output: {data_root}/videos/S{subj}_AC{class}.mp4

Usage:
    python -m cross_dataset.convert_3mdad_to_mp4 \
        --data_root <DATA_ROOT>/MDAD/MDAD
"""

import argparse
import os
import re
import subprocess
import sys

# Same AC mapping as generate_3mdad_annotations.py
MAPPED_ACS = {3, 4, 5, 6, 7, 8, 13, 14, 15}


def convert_clip(frame_dir, output_path, fps=30):
    """Convert a directory of JPGs to MP4 using ffmpeg."""
    # Get frame pattern: RGB1_SUB{s}ACT{a}F%d.jpg
    # ffmpeg needs a pattern, but frame numbers may not be contiguous
    # Use concat demuxer with file list instead
    frames = []
    for f in os.listdir(frame_dir):
        if not f.endswith(".jpg"):
            continue
        m = re.search(r"F(\d+)\.jpg$", f)
        if m:
            frames.append((int(m.group(1)), f))
    frames.sort(key=lambda x: x[0])

    if not frames:
        return False

    # Write temporary file list for ffmpeg concat
    list_path = output_path + ".filelist.txt"
    with open(list_path, "w") as fout:
        for _, fname in frames:
            fpath = os.path.join(frame_dir, fname)
            fout.write(f"file '{fpath}'\n")
            fout.write(f"duration {1.0/fps:.6f}\n")

    try:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-vf", "scale=256:256",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [ERROR] ffmpeg failed for {frame_dir}: {result.stderr[-200:]}")
            return False
        return True
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3MDAD frame dirs to MP4 videos")
    parser.add_argument("--data_root", type=str,
                        default="<DATA_ROOT>/MDAD/MDAD")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    rgb1_dir = os.path.join(args.data_root, "Day", "RGB1")
    video_dir = os.path.join(args.data_root, "videos")
    os.makedirs(video_dir, exist_ok=True)

    total = 0
    converted = 0

    for subj_name in sorted(os.listdir(rgb1_dir)):
        m = re.match(r"S(\d+)$", subj_name)
        if not m:
            continue
        subj_id = int(m.group(1))

        for ac_name in sorted(os.listdir(os.path.join(rgb1_dir, subj_name))):
            m2 = re.match(r"AC(\d+)$", ac_name)
            if not m2:
                continue
            ac_num = int(m2.group(1))

            if ac_num not in MAPPED_ACS:
                continue

            frame_dir = os.path.join(rgb1_dir, subj_name, ac_name)
            output_path = os.path.join(video_dir, f"S{subj_id}_AC{ac_num}.mp4")
            total += 1

            if os.path.exists(output_path):
                converted += 1
                continue

            ok = convert_clip(frame_dir, output_path, args.fps)
            if ok:
                converted += 1
                if converted % 50 == 0:
                    print(f"  Converted {converted}/{total}...")

    print(f"\nDone: {converted}/{total} clips converted to {video_dir}")


if __name__ == "__main__":
    main()

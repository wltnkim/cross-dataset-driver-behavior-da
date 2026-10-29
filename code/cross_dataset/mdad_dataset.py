"""3MDAD dataset loader for cross-dataset experiments.

Loads pre-converted MP4 videos using decord, maps action classes to unified
labels, and outputs tensors compatible with VideoMAE.

Pre-conversion required: run convert_3mdad_to_mp4.py first to create
{data_root}/videos/S{subj}_AC{class}.mp4 from raw frame directories.

Usage:
    from cross_dataset.mdad_dataset import MDADDataset

    ds = MDADDataset(
        data_root="<DATA_ROOT>/MDAD/MDAD",
        split="train",
        fold=0,
        num_frames=16,
        augment=True,
    )
"""

import collections
import csv
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from cross_dataset.class_mapping_utils import get_unified_label_maps, load_class_mapping

# VideoMAE ImageNet normalization
PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# AC number -> video filename mapping
_AC_TO_ACTIVITY = {
    3: "adjusting_radio", 4: "gps_operating",
    5: "texting_right", 6: "texting_left",
    7: "phone_call_right", 8: "phone_call_left",
    13: "drinking_right", 14: "drinking_left",
    15: "reaching_behind",
}


class MDADDataset(Dataset):
    """3MDAD video clip dataset (decord-based, from pre-converted MP4s).

    Each sample is a short clip (~3-5s at 30fps) of one action class.
    Videos are pre-resized to 256x256 during MP4 conversion.
    """

    def __init__(self, data_root, split="train", fold=0,
                 num_frames=16, resize=224, augment=False,
                 temporal_jitter=2, color_jitter=0.0, mapping_path=None):
        super().__init__()
        self.data_root = data_root
        self.num_frames = num_frames
        self.resize = resize
        self.augment = augment
        self.temporal_jitter = temporal_jitter
        self.color_jitter = color_jitter

        # Load class mapping
        label_maps = get_unified_label_maps(mapping_path)
        self.activity_to_label = label_maps["threeMDAD"]

        # Load annotation CSV
        mapping = load_class_mapping(mapping_path)
        split_key = f"split_{fold}"
        csv_rel = mapping["threeMDAD_splits"][split_key][split]
        csv_path = os.path.join(data_root, csv_rel)

        self.samples = []
        self._video_cache = collections.OrderedDict()
        self._cache_max = 2  # reduced 8→2 to bound decord VideoReader memory (OOM fix 2026-05-30)

        video_dir = os.path.join(data_root, "videos")

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                activity = row["activity"]
                if activity not in self.activity_to_label:
                    continue
                unified_label = self.activity_to_label[activity]

                # Derive video path from frame_dir: "Day/RGB1/S{s}/AC{a}" -> "videos/S{s}_AC{a}.mp4"
                frame_dir = row["frame_dir"]
                m = re.search(r"S(\d+)/AC(\d+)$", frame_dir)
                if not m:
                    continue
                subj_id = int(m.group(1))
                ac_num = int(m.group(2))
                video_path = os.path.join(video_dir, f"S{subj_id}_AC{ac_num}.mp4")

                if not os.path.isfile(video_path):
                    continue

                self.samples.append({
                    "video_path": video_path,
                    "label": unified_label,
                    "activity": activity,
                    "participant_id": subj_id,
                })

        print(f"[3MDAD] fold={fold} split={split}: "
              f"{len(self.samples)} clips from {csv_path}")

    def _get_video_reader(self, video_path):
        """Get or create a decord VideoReader for the given path (LRU, max 8)."""
        if video_path in self._video_cache:
            self._video_cache.move_to_end(video_path)
            return self._video_cache[video_path]
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path, num_threads=1)
        self._video_cache[video_path] = vr
        if len(self._video_cache) > self._cache_max:
            _, evicted = self._video_cache.popitem(last=False)
            del evicted  # explicit drop to release decord internal buffers
        return vr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        label = sample["label"]

        vr = self._get_video_reader(video_path)
        total_frames = len(vr)

        # Uniform sampling of num_frames from entire clip
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
            indices = np.pad(indices, (0, self.num_frames - total_frames),
                             mode="edge")

        # Temporal jitter
        if self.augment and self.temporal_jitter > 0:
            jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
            indices = np.clip(indices + jitter, 0, total_frames - 1)

        # Decode frames: decord returns [T, H, W, C] uint8 torch tensor
        frames = vr.get_batch(indices.tolist())  # [T, H, W, C]

        # Convert to [T, C, H, W] float [0, 1]
        tensor = frames.permute(0, 3, 1, 2).float().div_(255.0)

        # Videos are pre-resized to 256x256, so skip expensive resize.
        # Augmentation or center crop handles final sizing to self.resize.
        if self.augment:
            i, j, h, w = T.RandomResizedCrop.get_params(
                tensor[0], scale=(0.6, 1.0), ratio=(0.75, 1.333))
            tensor = torch.stack([
                TF.resized_crop(frame, i, j, h, w, [self.resize, self.resize])
                for frame in tensor
            ])
            if random.random() < 0.5:
                tensor = tensor.flip(-1)
            if self.color_jitter > 0 and random.random() < 0.8:
                s = self.color_jitter
                brightness = random.uniform(max(0, 1 - s), 1 + s)
                contrast = random.uniform(max(0, 1 - s), 1 + s)
                saturation = random.uniform(max(0, 1 - s), 1 + s)
                tensor = torch.stack([
                    TF.adjust_saturation(
                        TF.adjust_contrast(
                            TF.adjust_brightness(frame, brightness),
                            contrast),
                        saturation)
                    for frame in tensor
                ])
                tensor = tensor.clamp(0.0, 1.0)
        else:
            # Center crop 256x256 -> 224x224 (no resize needed)
            tensor = torch.stack([
                TF.center_crop(frame, [self.resize, self.resize])
                for frame in tensor
            ])

        # ImageNet normalization
        tensor = (tensor - PIXEL_MEAN) / PIXEL_STD

        return tensor, torch.tensor(label, dtype=torch.long)


def get_3mdad_class_distribution(data_root, fold=0, split="train", mapping_path=None):
    """Get class distribution for a given fold/split (without loading videos)."""
    label_maps = get_unified_label_maps(mapping_path)
    activity_to_label = label_maps["threeMDAD"]

    mapping = load_class_mapping(mapping_path)
    csv_rel = mapping["threeMDAD_splits"][f"split_{fold}"][split]
    csv_path = os.path.join(data_root, csv_rel)

    from collections import Counter
    counts = Counter()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            activity = row["activity"]
            if activity in activity_to_label:
                counts[activity_to_label[activity]] += 1
    return counts

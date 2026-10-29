"""DMD (Driver Monitoring Dataset) loader for cross-dataset experiments.

Loads video segments from rgb_body MP4 files using decord, maps distraction
actions to unified class labels, and outputs tensors compatible with VideoMAE.

Usage:
    from cross_dataset.dmd_dataset import DMDDataset

    ds = DMDDataset(
        data_root="<DATA_ROOT>/DMD/distraction/dmd",
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

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from cross_dataset.class_mapping_utils import get_unified_label_maps, load_class_mapping

# VideoMAE ImageNet normalization
PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DMDDataset(Dataset):
    """DMD distraction video segment dataset.

    Each sample is a video segment defined by (video_path, frame_start, frame_end, activity).
    Frames are decoded on-the-fly from MP4 using decord.
    """

    def __init__(self, data_root, split="train", fold=0,
                 num_frames=16, resize=224, augment=False,
                 temporal_jitter=2, color_jitter=0.0, mapping_path=None):
        """
        Args:
            data_root: Path to DMD distraction root (contains annotations_3s/)
            split: 'train', 'val', or 'test'
            fold: Cross-validation fold (0, 1, 2)
            num_frames: Number of frames to sample per clip
            resize: Spatial resize target
            augment: Whether to apply augmentation
            temporal_jitter: Temporal jitter range (frames)
            color_jitter: ColorJitter strength
            mapping_path: Path to class_mapping.json (default: auto)
        """
        super().__init__()
        self.data_root = data_root
        self.num_frames = num_frames
        self.resize = resize
        self.augment = augment
        self.temporal_jitter = temporal_jitter
        self.color_jitter = color_jitter

        # Load class mapping
        label_maps = get_unified_label_maps(mapping_path)
        self.activity_to_label = label_maps["dmd"]

        # Load annotation CSV
        mapping = load_class_mapping(mapping_path)
        split_key = f"split_{fold}"
        csv_rel = mapping["dmd_splits"][split_key][split]
        csv_path = os.path.join(data_root, csv_rel)

        self.samples = []
        self._video_cache = collections.OrderedDict()
        self._cache_max = 2  # reduced 8→2 to bound decord VideoReader memory (OOM fix 2026-05-30)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                activity = row["activity"]
                if activity not in self.activity_to_label:
                    continue
                unified_label = self.activity_to_label[activity]

                # file_id is like "gA/1/s1/gA_1_s1_..._rgb_body"
                # actual path is "{file_id}.mp4"
                video_path = os.path.join(data_root, row["file_id"] + ".mp4")
                frame_start = int(row["frame_start"])
                frame_end = int(row["frame_end"])

                if frame_end - frame_start < 4:
                    continue

                # Check video existence (only warn once per missing video)
                if not os.path.isfile(video_path):
                    continue

                self.samples.append({
                    "video_path": video_path,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "label": unified_label,
                    "activity": activity,
                    "participant_id": int(row["participant_id"]),
                })

        print(f"[DMD] fold={fold} split={split}: "
              f"{len(self.samples)} segments from {csv_path}")

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
        fs = sample["frame_start"]
        fe = sample["frame_end"]
        label = sample["label"]

        vr = self._get_video_reader(video_path)
        total_video_frames = len(vr)

        # Clamp to valid range
        fs = max(0, min(fs, total_video_frames - 1))
        fe = max(fs + 1, min(fe, total_video_frames))
        seg_len = fe - fs

        # Uniform sampling of num_frames from segment
        if seg_len >= self.num_frames:
            indices = np.linspace(fs, fe - 1, self.num_frames, dtype=int)
        else:
            # Pad by repeating last frame
            indices = np.linspace(fs, fe - 1, seg_len, dtype=int)
            indices = np.pad(indices, (0, self.num_frames - seg_len),
                             mode="edge")

        # Temporal jitter
        if self.augment and self.temporal_jitter > 0:
            jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
            indices = np.clip(indices + jitter, 0, total_video_frames - 1)

        # Decode frames: decord returns [T, H, W, C] uint8 torch tensor
        frames = vr.get_batch(indices.tolist())  # [T, H, W, C]

        # Convert to [T, C, H, W] float [0, 1]
        tensor = frames.permute(0, 3, 1, 2).float().div_(255.0)

        # Resize to target size
        tensor = torch.stack([
            TF.resize(frame, [self.resize, self.resize],
                      interpolation=TF.InterpolationMode.BILINEAR,
                      antialias=True)
            for frame in tensor
        ])

        # Augmentation
        if self.augment:
            # RandomResizedCrop
            i, j, h, w = T.RandomResizedCrop.get_params(
                tensor[0], scale=(0.6, 1.0), ratio=(0.75, 1.333))
            tensor = torch.stack([
                TF.resized_crop(frame, i, j, h, w, [self.resize, self.resize])
                for frame in tensor
            ])
            # Horizontal flip
            if random.random() < 0.5:
                tensor = tensor.flip(-1)
            # Color jitter
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
            # CenterCrop for eval
            tensor = torch.stack([
                TF.center_crop(frame, [self.resize, self.resize])
                for frame in tensor
            ])

        # ImageNet normalization
        tensor = (tensor - PIXEL_MEAN) / PIXEL_STD

        return tensor, torch.tensor(label, dtype=torch.long)


def get_dmd_class_distribution(data_root, fold=0, split="train", mapping_path=None):
    """Get class distribution for a given fold/split (without loading videos)."""
    label_maps = get_unified_label_maps(mapping_path)
    activity_to_label = label_maps["dmd"]

    mapping = load_class_mapping(mapping_path)
    csv_rel = mapping["dmd_splits"][f"split_{fold}"][split]
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

"""Generic memmap-backed video dataset (Option D, 2026-05-30).

Replaces on-the-fly decord MP4 decoding with pre-decoded uint8 memmap reads.
Eliminates I/O bottleneck observed in cross-DA training (18s/iter → ~1-2s/iter).

Storage format:
  - frames.npy : np.memmap, shape (N, T=16, H=224, W=224, C=3), dtype=uint8
  - metadata.json : list of dicts with keys:
      {video_id, frame_start, frame_end, unified_label,
       fold_assignment: {0: 'train'|'test', 1: ..., 2: ...}}

Compatible interface with DMDDataset / DriveAndActDataset / MDADDataset:
  __getitem__(idx) → (tensor[T,C,H,W] float32 normalized, label long)
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# ImageNet normalization (same as DMD/DA/MDAD datasets)
PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class MemmapVideoDataset(Dataset):
    """Read pre-decoded video frames from a numpy memmap.

    Args:
        memmap_path: path to frames.npy (uint8 memmap, shape (N, T, H, W, C))
        metadata_path: path to metadata.json (list of N dicts)
        fold: int (0/1/2)
        split: 'train' | 'test'
        num_frames: T (must match preprocessed value, default 16)
        resize: H == W (must match preprocessed value, default 224)
        augment: enable HFlip + ColorJitter (no spatial crop — preprocess fixed resolution)
        color_jitter: strength (0 disables)
        temporal_jitter: NOT supported in memmap mode (frames are pre-sampled).
                         If you need it, regenerate memmap with finer temporal grid.
    """

    def __init__(self, memmap_path, metadata_path, fold, split,
                 num_frames=16, resize=224, augment=False,
                 color_jitter=0.0, temporal_jitter=0, crop_size=None):
        self.memmap_path = memmap_path
        self.fold = int(fold)
        self.split = split
        self.num_frames = num_frames
        self.resize = resize
        self.augment = augment
        self.color_jitter = color_jitter
        # Tier-2: frames stored at >224 (e.g. 256); spatial-crop to crop_size at
        # load time. RandomCrop when augment (train), CenterCrop otherwise
        # (val/test). Same crop box across all T frames. None / >=stored = no-op.
        self.crop_size = crop_size
        if temporal_jitter > 0:
            # Document silently; do not crash. Preprocessed frames are fixed.
            pass

        with open(metadata_path, "r") as f:
            full_meta = json.load(f)

        # Filter to requested fold/split
        self._indices = [
            i for i, m in enumerate(full_meta)
            if m.get("fold_assignment", {}).get(str(fold)) == split
        ]
        self._labels = [full_meta[i]["unified_label"] for i in self._indices]

        # Lazy-open memmap (per-worker safe — np.memmap is fork-safe read-only)
        self._mm = None
        self._mm_shape = None
        self._mm_dtype = np.uint8
        # Probe shape from a small header read
        with open(memmap_path + ".shape", "r") as f:
            self._mm_shape = tuple(int(x) for x in f.read().strip().split(","))

    def _open_memmap(self):
        if self._mm is None:
            self._mm = np.memmap(
                self.memmap_path, dtype=self._mm_dtype,
                mode="r", shape=self._mm_shape)
        return self._mm

    def __len__(self):
        return len(self._indices)

    def get_labels(self):
        """For class-balanced samplers."""
        return list(self._labels)

    def __getitem__(self, idx):
        mm = self._open_memmap()
        global_idx = self._indices[idx]
        # Read [T, H, W, C] uint8
        frames = np.array(mm[global_idx], copy=True)  # copy to break mmap link

        # → [T, C, H, W] float [0, 1]
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)

        # Tier-2 spatial crop: stored HxW (e.g. 256) -> crop_size (e.g. 224).
        # Single crop box applied identically across all T frames.
        if self.crop_size is not None:
            H, W = tensor.shape[-2], tensor.shape[-1]
            ch = cw = int(self.crop_size)
            if H > ch and W > cw:
                if self.augment:  # train: random crop
                    top = random.randint(0, H - ch)
                    left = random.randint(0, W - cw)
                else:             # val/test: center crop
                    top = (H - ch) // 2
                    left = (W - cw) // 2
                tensor = tensor[:, :, top:top + ch, left:left + cw]

        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                tensor = tensor.flip(-1)
            # Color jitter (per-clip, applied uniformly across T)
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

        # ImageNet normalization
        tensor = (tensor - PIXEL_MEAN) / PIXEL_STD

        label = self._labels[idx]
        return tensor, torch.tensor(label, dtype=torch.long)

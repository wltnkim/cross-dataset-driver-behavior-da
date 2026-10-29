"""OBC dataset loader for cross-dataset experiments.

Wraps the existing OBC memmap cache and maps 33 OBC classes to
the unified cross-dataset label space (6 classes).

Usage:
    from cross_dataset.obc_dataset import OBCCrossDataset

    ds = OBCCrossDataset(
        metadata_path="features/keyframe_video_metadata_dualcam.pkl",
        split="train",
        num_frames=16,
        augment=True,
    )
"""

import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from cross_dataset.class_mapping_utils import (
    get_unified_label_maps,
    load_class_mapping,
)

# VideoMAE ImageNet normalization
PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_obc_cache(metadata_path):
    """Load existing OBC memmap cache (built by train_videomae.py)."""
    base = metadata_path.replace(".pkl", "")
    cache_npy = base + "_cache.npy"
    cache_meta = base + "_cache_meta.pkl"

    if not os.path.exists(cache_npy):
        raise FileNotFoundError(
            f"OBC cache not found: {cache_npy}\n"
            "Run train_videomae.py first to build the frame cache."
        )

    with open(cache_meta, "rb") as f:
        meta = pickle.load(f)

    cache = np.memmap(cache_npy, dtype="uint8", mode="r", shape=tuple(meta["shape"]))
    return cache, meta


class OBCCrossDataset(Dataset):
    """OBC dataset mapped to unified cross-dataset labels.

    Only includes samples whose OBC class maps to one of the unified classes.
    """

    def __init__(self, metadata_path, split="train", seed=42,
                 num_frames=16, augment=False, temporal_jitter=2,
                 color_jitter=0.0, mapping_path=None):
        """
        Args:
            metadata_path: Path to keyframe_video_metadata_dualcam.pkl
            split: 'train', 'val', or 'test'
            seed: Random seed for subject-disjoint split
            num_frames: Number of frames to sample
            augment: Whether to apply augmentation
            temporal_jitter: Temporal jitter range
            color_jitter: ColorJitter strength
            mapping_path: Path to class_mapping.json (default: auto)
        """
        super().__init__()
        self.num_frames = num_frames
        self.augment = augment
        self.temporal_jitter = temporal_jitter
        self.color_jitter = color_jitter

        # Load OBC cache
        self.cache, cache_meta = _load_obc_cache(metadata_path)
        all_labels = cache_meta["labels"]
        all_subjects = cache_meta["subjects"]

        # Build OBC class name -> unified label mapping
        label_maps = get_unified_label_maps(mapping_path)
        obc_name_to_unified = label_maps["obc"]

        # OBC numeric label -> class name
        mapping = load_class_mapping(mapping_path)
        obc_class_names = mapping["obc_class_names"]

        # Subject-disjoint split (same logic as train_videomae.py)
        unique_subjects = sorted(set(all_subjects))
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_subjects)

        train_subjects = set(unique_subjects[:26])
        val_subjects = set(unique_subjects[26:34])
        test_subjects = set(unique_subjects[34:])

        split_subjects = {
            "train": train_subjects,
            "val": val_subjects,
            "test": test_subjects,
        }[split]

        # Filter: only keep samples in the split AND with a mapped class
        self.indices = []
        self.labels = []
        n_skipped = 0

        for i, (label, subj) in enumerate(zip(all_labels, all_subjects)):
            if subj not in split_subjects:
                continue
            obc_name = obc_class_names[label]
            if obc_name not in obc_name_to_unified:
                n_skipped += 1
                continue
            self.indices.append(i)
            self.labels.append(obc_name_to_unified[obc_name])

        print(f"[OBC-Cross] split={split}: {len(self.indices)} samples "
              f"(skipped {n_skipped} non-mapped), "
              f"subjects={len(split_subjects)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        frames = self.cache[global_idx]  # [51, 224, 224, 3] uint8
        total_frames = frames.shape[0]

        # Uniform sampling
        sample_indices = np.linspace(0, total_frames - 1,
                                     self.num_frames, dtype=int)
        if self.augment and self.temporal_jitter > 0:
            jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
            sample_indices = np.clip(sample_indices + jitter, 0, total_frames - 1)

        selected = frames[sample_indices]  # [16, 224, 224, 3] uint8

        # To float tensor [T, C, H, W] [0, 1]
        tensor = torch.from_numpy(selected.copy()).permute(0, 3, 1, 2).float().div_(255.0)

        # Augmentation
        if self.augment:
            i, j, h, w = T.RandomResizedCrop.get_params(
                tensor[0], scale=(0.6, 1.0), ratio=(0.75, 1.333))
            tensor = torch.stack([
                TF.resized_crop(frame, i, j, h, w, [224, 224])
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

        # ImageNet normalization
        tensor = (tensor - PIXEL_MEAN) / PIXEL_STD

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label

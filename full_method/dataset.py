"""FullMethodDataset: returns sample_id, tier, has_spalling alongside image/mask.

Reuses baseline_unet I/O helpers; adds per-sample metadata for dynamic curriculum.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from baseline_unet.dataset import (
    decode_mask,
    image_path,
    mask_path,
    read_image_rgb,
    read_mask_rgb,
)
from . import config as C


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

_TIER_MAP = {"Easy": 0, "Medium": 1, "Hard": 2}


def build_records(train_files: List[str], root: Path) -> List[Dict]:
    """Build per-sample metadata records from file list.

    Returns list of {"id": str, "rel": str, "tier": int, "has_spalling": bool}.
    """
    records = []
    for rel in train_files:
        prefix = rel.split("/", 1)[0]
        tier = _TIER_MAP.get(prefix, 2)
        m = read_mask_rgb(mask_path(root, rel))
        has_spalling = bool((m[..., 2] > 127).any())
        records.append({
            "id": rel,
            "rel": rel,
            "tier": tier,
            "has_spalling": has_spalling,
        })
    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FullMethodDataset(Dataset):
    """Dataset returning dict with image, mask, sample_id, tier, has_spalling."""

    def __init__(self, root: Path, records: List[Dict], transform=None):
        self.root = Path(root)
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        rel = rec["rel"]
        img = read_image_rgb(image_path(self.root, rel))
        mask_rgb = read_mask_rgb(mask_path(self.root, rel))
        label, _ = decode_mask(mask_rgb)

        assert label.shape == img.shape[:2], f"shape mismatch for {rel}"
        uq = np.unique(label)
        assert uq.max() < C.NUM_CLASSES and uq.min() >= 0, f"bad label classes {uq} in {rel}"

        if self.transform is not None:
            out = self.transform(image=img, mask=label)
            image_t = out["image"]
            mask_t = out["mask"]
            if not torch.is_tensor(mask_t):
                mask_t = torch.from_numpy(mask_t)
            mask_t = mask_t.long()
        else:
            image_t = img
            mask_t = label

        return {
            "image": image_t,
            "mask": mask_t,
            "sample_id": rec["id"],
            "tier": rec["tier"],
            "has_spalling": rec["has_spalling"],
            "rel": rec["rel"],
        }


# ---------------------------------------------------------------------------
# Custom collate for dict-based dataset
# ---------------------------------------------------------------------------

def dict_collate(batch: List[Dict]) -> Dict:
    """Stack image/mask tensors, collect other fields as lists."""
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    return {
        "image": images,
        "mask": masks,
        "sample_id": [b["sample_id"] for b in batch],
        "tier": [b["tier"] for b in batch],
        "has_spalling": [b["has_spalling"] for b in batch],
        "rel": [b["rel"] for b in batch],
    }

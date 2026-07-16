"""FullMethodDataset: returns sample_id, tier, has_spalling alongside image/mask.

Reuses baseline_unet I/O helpers; adds per-sample metadata for dynamic curriculum.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
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
# Detection bbox helpers
# ---------------------------------------------------------------------------

_DETECT_VOC_DIR: Optional[Path] = None


def _get_detect_voc_dir() -> Path:
    global _DETECT_VOC_DIR
    if _DETECT_VOC_DIR is None:
        _DETECT_VOC_DIR = (
            Path(__file__).resolve().parent.parent
            / "Dataset" / "DamSegment" / "Damage Detection" / "Labels" / "Pascal VOC"
        )
    return _DETECT_VOC_DIR


def load_bbox_mask(rel: str, target_h: int, target_w: int) -> np.ndarray:
    """Load detection bboxes for a sample and create a binary mask.

    Args:
        rel: e.g. 'Easy/E (12).jpg'
        target_h, target_w: output mask size (matches segmentation mask).

    Returns:
        (H, W) uint8 array, 1 inside any damage bbox, 0 otherwise.
    """
    _, name = rel.split("/", 1)
    stem = Path(name).stem
    voc_path = _get_detect_voc_dir() / f"{stem}.json"

    if not voc_path.exists():
        return np.zeros((target_h, target_w), dtype=np.uint8)

    with open(voc_path) as f:
        det = json.load(f)

    img_w = det["image"]["width"]
    img_h = det["image"]["height"]

    bbox_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for ann in det["annotations"]:
        x, y, bw, bh = ann["bbox"]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img_w, int(x + bw)), min(img_h, int(y + bh))
        bbox_mask[y1:y2, x1:x2] = 1

    if (img_h, img_w) != (target_h, target_w):
        bbox_mask = cv2.resize(bbox_mask, (target_w, target_h),
                               interpolation=cv2.INTER_NEAREST)
    return bbox_mask


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

_TIER_MAP = {"Easy": 0, "Medium": 1, "Hard": 2}


def build_records(train_files: List[str], root: Path,
                   group_map: Optional[Dict[str, int]] = None) -> List[Dict]:
    """Build per-sample metadata records from file list.

    Returns list of {"id": str, "rel": str, "tier": int, "has_spalling": bool}.
    If group_map provided, also includes "group_id": int.
    """
    records = []
    for rel in train_files:
        prefix = rel.split("/", 1)[0]
        tier = _TIER_MAP.get(prefix, 2)
        m = read_mask_rgb(mask_path(root, rel))
        has_spalling = bool((m[..., 2] > 127).any())
        rec = {
            "id": rel,
            "rel": rel,
            "tier": tier,
            "has_spalling": has_spalling,
        }
        if group_map is not None:
            rec["group_id"] = group_map.get(rel, -1)
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FullMethodDataset(Dataset):
    """Dataset returning dict with image, mask, sample_id, tier, has_spalling.

    When ``compute_skel=True``, also returns a precomputed crack skeleton
    for Skeleton Recall Loss (SRL).

    When ``load_bbox=True``, also returns a binary bbox mask from detection
    annotations for bbox-guided loss.

    When ``tier_transforms`` is provided, uses tier-specific augmentation
    (Tier-Conditional Augmentation / TCA).
    """

    _skeletonize = None  # lazy import

    def __init__(self, root: Path, records: List[Dict], transform=None,
                 compute_skel: bool = False, load_bbox: bool = False,
                 tier_transforms: Optional[Dict[int, object]] = None):
        self.root = Path(root)
        self.records = records
        self.transform = transform
        self.compute_skel = compute_skel
        self.load_bbox = load_bbox
        self.tier_transforms = tier_transforms
        if compute_skel and FullMethodDataset._skeletonize is None:
            from skimage.morphology import skeletonize
            FullMethodDataset._skeletonize = staticmethod(skeletonize)

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

        # TCA: use tier-specific transform if available
        tfm = self.transform
        if self.tier_transforms is not None:
            tier = rec["tier"]
            tfm = self.tier_transforms.get(tier, self.transform)

        if tfm is not None:
            out = tfm(image=img, mask=label)
            image_t = out["image"]
            mask_t = out["mask"]
            if not torch.is_tensor(mask_t):
                mask_t = torch.from_numpy(mask_t)
            mask_t = mask_t.long()
        else:
            image_t = img
            mask_t = label

        result = {
            "image": image_t,
            "mask": mask_t,
            "sample_id": rec["id"],
            "tier": rec["tier"],
            "has_spalling": rec["has_spalling"],
            "rel": rec["rel"],
        }
        if "group_id" in rec:
            result["group_id"] = rec["group_id"]

        # Compute crack skeleton from the augmented mask for SRL.
        # Done post-augmentation so it matches the actual training mask.
        if self.compute_skel:
            mask_np = mask_t.numpy() if torch.is_tensor(mask_t) else mask_t
            crack_mask = (mask_np == 1)
            if crack_mask.any():
                skel = self._skeletonize(crack_mask).astype(np.float32)
            else:
                skel = np.zeros(mask_np.shape, dtype=np.float32)
            result["crack_skel"] = torch.from_numpy(skel).unsqueeze(0)  # (1,H,W)

        # Load detection bbox mask
        if self.load_bbox:
            h = mask_t.shape[-2] if torch.is_tensor(mask_t) else mask_t.shape[0]
            w = mask_t.shape[-1] if torch.is_tensor(mask_t) else mask_t.shape[1]
            bbox_np = load_bbox_mask(rel, h, w)
            result["bbox_mask"] = torch.from_numpy(bbox_np).float()  # (H, W)

        return result


# ---------------------------------------------------------------------------
# Custom collate for dict-based dataset
# ---------------------------------------------------------------------------

def dict_collate(batch: List[Dict]) -> Dict:
    """Stack image/mask tensors, collect other fields as lists."""
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    result = {
        "image": images,
        "mask": masks,
        "sample_id": [b["sample_id"] for b in batch],
        "tier": [b["tier"] for b in batch],
        "has_spalling": [b["has_spalling"] for b in batch],
        "rel": [b["rel"] for b in batch],
    }
    if "crack_skel" in batch[0]:
        result["crack_skel"] = torch.stack([b["crack_skel"] for b in batch])
    if "bbox_mask" in batch[0]:
        result["bbox_mask"] = torch.stack([b["bbox_mask"] for b in batch])
    if "group_id" in batch[0]:
        result["group_id"] = [b["group_id"] for b in batch]
    return result

"""DamSegment dataset + RGB→class decoding + albumentations pipelines.

Can be run standalone as ``python -m baseline_unet.dataset --peek`` to sanity
check the decoding (4 image/mask visualisations) and verify that crack maps to
red-channel and spalling maps to blue-channel in the raw PNG masks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception as e:  # pragma: no cover
    A = None
    ToTensorV2 = None

from . import config as C


# ----------------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------------

def image_path(root: Path, rel: str) -> Path:
    """rel is e.g. 'Easy/E (12).jpg'."""
    diff, name = rel.split("/", 1)
    return root / diff / "Images" / name


def mask_path(root: Path, rel: str) -> Path:
    diff, name = rel.split("/", 1)
    stem = Path(name).stem
    return root / diff / "Labels" / "Mask" / f"{stem}_mask.png"


def read_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask_rgb(path: Path) -> np.ndarray:
    """Return the mask as HxWx3 RGB (NOT BGR). cv2 reads BGR by default!"""
    m = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if m is None:
        raise FileNotFoundError(f"cannot read mask {path}")
    return cv2.cvtColor(m, cv2.COLOR_BGR2RGB)


def decode_mask(mask_rgb: np.ndarray) -> Tuple[np.ndarray, int]:
    """Channel-threshold decode. Returns (label[int64], overlap_px).

    Priority: spalling overwrites crack when both channels are active on the
    same pixel.
    """
    r = mask_rgb[..., 0]
    b = mask_rgb[..., 2]
    is_crack = r > 127
    is_spalling = b > 127
    overlap = int(np.logical_and(is_crack, is_spalling).sum())
    label = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    label[is_crack] = 1
    label[is_spalling] = 2
    return label, overlap


def has_spalling_from_mask(mask_rgb: np.ndarray) -> bool:
    return bool((mask_rgb[..., 2] > 127).any())


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool, aug_level: str = "basic"):
    if A is None:
        raise RuntimeError("albumentations is required but not installed")
    if train:
        aug_list = [
            A.Resize(img_size, img_size,
                     interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]
        if aug_level == "strong":
            aug_list += [
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.15, rotate_limit=30,
                    border_mode=cv2.BORDER_REFLECT_101,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=0.5,
                ),
                A.ElasticTransform(
                    alpha=80, sigma=10,
                    border_mode=cv2.BORDER_REFLECT_101,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=0.2,
                ),
            ]
        aug_list += [
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
        return A.Compose(aug_list)
    return A.Compose([
        A.Resize(img_size, img_size,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

def read_split_file(path: Path) -> List[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


class DamSegmentDataset(Dataset):
    def __init__(self, root: Path, file_list: List[str], transform=None):
        self.root = Path(root)
        self.files = list(file_list)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        rel = self.files[idx]
        img = read_image_rgb(image_path(self.root, rel))
        mask_rgb = read_mask_rgb(mask_path(self.root, rel))
        label, _ = decode_mask(mask_rgb)
        assert label.shape == img.shape[:2], f"shape mismatch for {rel}"
        assert label.dtype == np.int64
        uq = np.unique(label)
        assert uq.max() < C.NUM_CLASSES and uq.min() >= 0, f"bad label classes {uq} in {rel}"

        if self.transform is not None:
            out = self.transform(image=img, mask=label)
            image_t = out["image"]  # float tensor CHW
            mask_t = out["mask"]
            if not torch.is_tensor(mask_t):
                mask_t = torch.from_numpy(mask_t)
            mask_t = mask_t.long()
            # Post-transform sanity: nearest-neighbor resize must not create new classes.
            uq2 = torch.unique(mask_t)
            assert int(uq2.max().item()) < C.NUM_CLASSES and int(uq2.min().item()) >= 0, \
                f"bad post-transform classes {uq2.tolist()} in {rel}"
            return image_t, mask_t, rel
        return img, label, rel


# ----------------------------------------------------------------------------
# Class weights
# ----------------------------------------------------------------------------

def compute_class_weights(root: Path, files: List[str], eps: float = 1e-6,
                          clip: Tuple[float, float] = (0.25, 5.0)):
    """Scan training masks once and return (raw_freq, smoothed, clipped)."""
    counts = np.zeros(C.NUM_CLASSES, dtype=np.float64)
    for rel in files:
        m = read_mask_rgb(mask_path(root, rel))
        label, _ = decode_mask(m)
        for c in range(C.NUM_CLASSES):
            counts[c] += int((label == c).sum())
    total = counts.sum()
    freq = counts / max(total, 1.0)
    smoothed = 1.0 / np.sqrt(freq + eps)
    smoothed = smoothed / smoothed.mean()
    clipped = np.clip(smoothed, clip[0], clip[1])
    return freq, smoothed, clipped


# ----------------------------------------------------------------------------
# --peek mode
# ----------------------------------------------------------------------------

def _peek(n: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = C.DATA_ROOT
    # Load train split if available, else scan everything.
    train_txt = C.SPLITS_DIR / "train.txt"
    if train_txt.exists():
        files = read_split_file(train_txt)
    else:
        files = []
        for diff in C.DIFFICULTIES:
            imgs = sorted((root / diff / "Images").glob("*.jpg"))
            files.extend([f"{diff}/{p.name}" for p in imgs])

    # --- (A) Verify channel order on first 5 has_spalling samples -----------
    print("[peek] verifying BGR->RGB channel order on 5 spalling samples ...")
    verified = 0
    for rel in files:
        m = read_mask_rgb(mask_path(root, rel))
        if not has_spalling_from_mask(m):
            continue
        label, _ = decode_mask(m)
        # label==2 (spalling) must coincide with raw blue-channel hot pixels
        sp_mask = (label == 2)
        cr_mask = (label == 1)
        if sp_mask.sum() > 0:
            assert (m[..., 2][sp_mask] > 127).all(), f"spalling <-> blue channel mismatch in {rel}"
        if cr_mask.sum() > 0:
            assert (m[..., 0][cr_mask] > 127).all(), f"crack <-> red channel mismatch in {rel}"
        verified += 1
        if verified >= 5:
            break
    if verified == 0:
        print("[peek] WARNING: no spalling samples found for channel-order check")
    else:
        print(f"[peek] channel order OK on {verified} spalling samples")

    # --- (B) Save n visualisations -----------------------------------------
    sanity_dir = C.SANITY_DIR
    sanity_dir.mkdir(parents=True, exist_ok=True)
    # Pick first 2 crack-only and first 2 with spalling to be informative.
    picks: List[str] = []
    found_sp = 0
    found_cr = 0
    for rel in files:
        m = read_mask_rgb(mask_path(root, rel))
        sp = has_spalling_from_mask(m)
        if sp and found_sp < n // 2:
            picks.append(rel); found_sp += 1
        elif (not sp) and found_cr < n - n // 2:
            picks.append(rel); found_cr += 1
        if len(picks) >= n:
            break

    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    for i, rel in enumerate(picks):
        img = read_image_rgb(image_path(root, rel))
        m = read_mask_rgb(mask_path(root, rel))
        label, overlap = decode_mask(m)
        color = palette[label]
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img); axs[0].set_title(rel); axs[0].axis("off")
        axs[1].imshow(m); axs[1].set_title("raw mask (RGB)"); axs[1].axis("off")
        axs[2].imshow(color); axs[2].set_title(f"decoded (overlap={overlap})"); axs[2].axis("off")
        out = sanity_dir / f"peek_{i:02d}.png"
        fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)
        print(f"[peek] saved {out}")

    # --- (C) Full-set overlap statistics -----------------------------------
    print("[peek] scanning full file list for overlap stats ...")
    total_fg = 0
    total_overlap = 0
    hist_bins = [0, 0, 0, 0, 0]  # 0, (0,0.001], (0.001,0.005], (0.005,0.02], >0.02
    for rel in files:
        m = read_mask_rgb(mask_path(root, rel))
        label, ov = decode_mask(m)
        fg = int((label > 0).sum())
        total_fg += fg
        total_overlap += ov
        if fg == 0:
            hist_bins[0] += 1; continue
        r = ov / max(fg, 1)
        if r == 0: hist_bins[0] += 1
        elif r <= 0.001: hist_bins[1] += 1
        elif r <= 0.005: hist_bins[2] += 1
        elif r <= 0.02: hist_bins[3] += 1
        else: hist_bins[4] += 1
    ratio = total_overlap / max(total_fg, 1)
    print(f"[peek] global overlap/foreground = {ratio:.6f}")
    print(f"[peek] per-image overlap histogram (=0, <=0.1%, <=0.5%, <=2%, >2%): {hist_bins}")
    if ratio < 0.005:
        print("[peek] OK: overlap < 0.5%; spalling-overrides-crack policy is fine.")
    else:
        print("[peek] WARNING: overlap >= 0.5%; consider JSON polygon reconstruction before Step 1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--peek", action="store_true")
    args = parser.parse_args()
    if args.peek:
        _peek()
    else:
        print("usage: python -m baseline_unet.dataset --peek")

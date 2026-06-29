#!/usr/bin/env python
"""Fig 1: Dataset overview — 2 samples per difficulty tier (image + GT overlay).

Usage:
    python scripts/fig1_dataset_samples.py [--output figures/fig1_dataset.pdf]

Produces a 3-row × 4-col figure:
    Row 0 = Easy, Row 1 = Medium, Row 2 = Hard
    Each row: image1, overlay1, image2, overlay2
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---- paths (match baseline_unet.config) ----
CODES_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = CODES_DIR / "Dataset" / "DamSegment" / "Damage Segmentaion"
SPLITS_DIR = CODES_DIR / "baseline_unet" / "splits"

# ---- helpers (match baseline_unet.dataset) ----
def image_path(root, rel):
    diff, name = rel.split("/", 1)
    return root / diff / "Images" / name

def mask_path(root, rel):
    diff, name = rel.split("/", 1)
    stem = Path(name).stem
    return root / diff / "Labels" / "Mask" / f"{stem}_mask.png"

def read_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def decode_mask(mask_rgb):
    r, b = mask_rgb[..., 0], mask_rgb[..., 2]
    label = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    label[r > 127] = 1   # crack
    label[b > 127] = 2   # spalling
    return label

def overlay(img, label, alpha=0.45):
    """Blend semi-transparent colored mask onto image."""
    vis = img.copy()
    # crack = red, spalling = green
    crack_mask = label == 1
    spalling_mask = label == 2
    vis[crack_mask] = (np.array([255, 60, 60]) * alpha +
                       vis[crack_mask] * (1 - alpha)).astype(np.uint8)
    vis[spalling_mask] = (np.array([60, 200, 60]) * alpha +
                          vis[spalling_mask] * (1 - alpha)).astype(np.uint8)
    return vis


def pick_samples(split_file, tier, n=2, seed=42):
    """Pick n samples from a tier that have both crack and spalling if possible."""
    with open(split_file) as f:
        files = [l.strip() for l in f if l.strip()]
    tier_files = [f for f in files if f.startswith(tier + "/")]
    # Prefer samples with both classes
    both, rest = [], []
    for rel in tier_files:
        m = read_rgb(mask_path(DATA_ROOT, rel))
        label = decode_mask(m)
        has_crack = (label == 1).any()
        has_spalling = (label == 2).any()
        if has_crack and has_spalling:
            both.append(rel)
        else:
            rest.append(rel)
    rng = random.Random(seed)
    pool = both if len(both) >= n else both + rest
    return rng.sample(pool, min(n, len(pool)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/fig1_dataset.pdf")
    parser.add_argument("--split", default="test",
                        help="Which split to sample from (default: test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_file = SPLITS_DIR / f"{args.split}.txt"
    tiers = ["Easy", "Medium", "Hard"]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for row, tier in enumerate(tiers):
        samples = pick_samples(split_file, tier, n=2, seed=args.seed)
        for j, rel in enumerate(samples):
            img = read_rgb(image_path(DATA_ROOT, rel))
            mask_rgb = read_rgb(mask_path(DATA_ROOT, rel))
            label = decode_mask(mask_rgb)
            ov = overlay(img, label)

            # Image
            ax_img = axes[row, j * 2]
            ax_img.imshow(img)
            ax_img.set_title(f"{tier} — {Path(rel).stem}", fontsize=13)
            ax_img.axis("off")

            # Overlay
            ax_ov = axes[row, j * 2 + 1]
            ax_ov.imshow(ov)
            cr_px = int((label == 1).sum())
            sp_px = int((label == 2).sum())
            total_px = label.shape[0] * label.shape[1]
            ax_ov.set_title(
                f"Crack: {cr_px/total_px*100:.1f}%  Spall: {sp_px/total_px*100:.1f}%",
                fontsize=11)
            ax_ov.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=(1, 0.24, 0.24), label="Crack"),
        mpatches.Patch(color=(0.24, 0.78, 0.24), label="Spalling"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("DamSegment Dataset: Samples by Difficulty Tier", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    # Also save PNG
    png_path = out_path.with_suffix(".png")
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

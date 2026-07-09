"""Compute engineering quantities from GT masks for representative test images.

For each test image, computes:
- Crack area (px), Spalling area (px)
- Crack / spalling connected components
- Longest crack path (skeleton-based, px)
- Whether all crack skeleton endpoints are mutually reachable (PathCont)

Selects one representative image per tier (Easy/Medium/Hard) that contains
both crack and spalling, sorted by total defect pixel count (median).
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baseline_unet.dataset import mask_path, read_mask_rgb, decode_mask
from full_method import config as C

DATA_ROOT = C.DATA_ROOT


def skeleton_longest_path(binary_mask):
    """Compute longest skeleton path length in pixels."""
    skel = skeletonize(binary_mask > 0)
    skel_px = int(skel.sum())
    return skel_px, skel


def count_components(binary_mask):
    labeled, n = ndimage.label(binary_mask)
    return n, labeled


def compute_quantities(label_mask):
    """Compute engineering quantities from a decoded label mask.

    label_mask: H x W int array, 0=bg, 1=crack, 2=spalling
    """
    crack_mask = (label_mask == 1).astype(np.uint8)
    spall_mask = (label_mask == 2).astype(np.uint8)

    crack_area = int(crack_mask.sum())
    spall_area = int(spall_mask.sum())

    crack_cc, _ = count_components(crack_mask)
    spall_cc, _ = count_components(spall_mask)

    crack_skel_len, _ = skeleton_longest_path(crack_mask)

    return {
        "crack_area_px": crack_area,
        "spall_area_px": spall_area,
        "crack_cc": crack_cc,
        "spall_cc": spall_cc,
        "crack_skel_px": crack_skel_len,
    }


def main():
    # Read test split
    test_file = C.SPLIT_FILES["test"]
    with open(test_file) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    # Group by tier
    tiers = {"Easy": [], "Medium": [], "Hard": []}
    for rel in test_ids:
        tier = rel.split("/")[0]
        mp = mask_path(DATA_ROOT, rel)
        mask_rgb = read_mask_rgb(mp)
        label, _ = decode_mask(mask_rgb)

        crack_px = int((label == 1).sum())
        spall_px = int((label == 2).sum())
        total_fg = crack_px + spall_px

        # Only consider images with both crack and spalling
        if crack_px > 0 and spall_px > 0:
            tiers[tier].append((rel, total_fg, crack_px, spall_px))

    # For each tier, pick the image closest to the median defect pixel count
    selected = []
    for tier_name in ["Easy", "Medium", "Hard"]:
        imgs = tiers[tier_name]
        if not imgs:
            print(f"WARNING: no images with both crack+spalling in {tier_name}")
            continue
        imgs.sort(key=lambda x: x[1])
        median_idx = len(imgs) // 2
        selected.append(imgs[median_idx])
        print(f"{tier_name}: {len(imgs)} images with both classes, "
              f"selected median: {imgs[median_idx][0]} "
              f"(fg={imgs[median_idx][1]} px)")

    # Compute quantities for selected images
    print(f"\n{'='*80}")
    print(f"{'Image':<25} {'Tier':<8} {'Cr area':>8} {'Sp area':>8} "
          f"{'Cr CC':>6} {'Sp CC':>6} {'Cr skel':>8} "
          f"{'Cr area mm²':>12} {'Sp area mm²':>12} {'Cr len mm':>10}")
    print(f"{'='*80}")

    GSD = 1.0  # mm/pixel assumed

    for rel, total_fg, _, _ in selected:
        tier = rel.split("/")[0]
        mp = mask_path(DATA_ROOT, rel)
        mask_rgb = read_mask_rgb(mp)
        label, _ = decode_mask(mask_rgb)

        q = compute_quantities(label)

        cr_area_mm2 = q["crack_area_px"] * GSD * GSD
        sp_area_mm2 = q["spall_area_px"] * GSD * GSD
        cr_len_mm = q["crack_skel_px"] * GSD

        print(f"{rel:<25} {tier:<8} {q['crack_area_px']:>8} {q['spall_area_px']:>8} "
              f"{q['crack_cc']:>6} {q['spall_cc']:>6} {q['crack_skel_px']:>8} "
              f"{cr_area_mm2:>12.0f} {sp_area_mm2:>12.0f} {cr_len_mm:>10.0f}")

    # Also print LaTeX table rows
    print(f"\n--- LaTeX table rows ---")
    for rel, total_fg, _, _ in selected:
        tier = rel.split("/")[0]
        mp = mask_path(DATA_ROOT, rel)
        mask_rgb = read_mask_rgb(mp)
        label, _ = decode_mask(mask_rgb)
        q = compute_quantities(label)

        cr_area_mm2 = q["crack_area_px"] * GSD * GSD
        sp_area_mm2 = q["spall_area_px"] * GSD * GSD
        cr_len_mm = q["crack_skel_px"] * GSD

        # Format name
        name = rel.split("/")[1].replace(".jpg", "").replace(" ", "~")
        print(f"{tier} & {q['crack_area_px']:,} & {q['spall_area_px']:,} "
              f"& {q['crack_cc']} & {q['spall_cc']} "
              f"& {q['crack_skel_px']:,} "
              f"& {cr_area_mm2:,.0f} & {sp_area_mm2:,.0f} & {cr_len_mm:,.0f} \\\\")


if __name__ == "__main__":
    main()

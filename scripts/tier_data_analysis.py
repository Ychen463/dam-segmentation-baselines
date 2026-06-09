#!/usr/bin/env python3
"""Tier data analysis: per-tier segmentation stats + extra data inventory.

Step 0 of the tier-leveraging plan. Runs without torch dependency.

Usage:
    python scripts/tier_data_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CODES_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = CODES_DIR / "Dataset" / "DamSegment" / "Damage Segmentaion"
DETECT_ROOT = CODES_DIR / "Dataset" / "DamSegment" / "Damage Detection"
CLASSIF_ROOT = CODES_DIR / "Dataset" / "DamSegment" / "Damage Classification"
SPLITS_DIR = CODES_DIR / "baseline_unet" / "splits"


# =========================================================================
# Lightweight I/O (no torch needed)
# =========================================================================

def read_split_file(path: Path) -> list:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def image_path(root, rel):
    diff, name = rel.split("/", 1)
    return root / diff / "Images" / name


def mask_path(root, rel):
    diff, name = rel.split("/", 1)
    stem = Path(name).stem
    return root / diff / "Labels" / "Mask" / f"{stem}_mask.png"


def read_mask_rgb(path):
    m = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if m is None:
        raise FileNotFoundError(f"cannot read mask {path}")
    return cv2.cvtColor(m, cv2.COLOR_BGR2RGB)


def decode_mask(mask_rgb):
    r = mask_rgb[..., 0]
    b = mask_rgb[..., 2]
    is_crack = r > 127
    is_spalling = b > 127
    label = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    label[is_crack] = 1
    label[is_spalling] = 2
    return label


def tier_from_rel(rel: str) -> str:
    return rel.split("/", 1)[0]


# =========================================================================
# Morphology features (inline, no torch)
# =========================================================================

def compute_morph_features(mask):
    from scipy.ndimage import distance_transform_edt, label
    from skimage.morphology import skeletonize
    from scipy.signal import fftconvolve

    h, w = mask.shape
    total_px = h * w
    crack_mask = (mask == 1)
    spalling_mask = (mask == 2)
    crack_px = crack_mask.sum()
    spalling_px = spalling_mask.sum()

    feat = {
        "crack_ratio": float(crack_px) / total_px,
        "spalling_ratio": float(spalling_px) / total_px,
        "has_crack": bool(crack_px > 0),
        "has_spalling": bool(spalling_px > 0),
        "crack_mean_width": 0.0,
        "crack_width_cv": 0.0,
        "junction_density": 0.0,
        "crack_components": 0,
        "crack_spalling_proximity": 0.0,
    }

    if crack_px < 10:
        return feat

    skel = skeletonize(crack_mask)
    skel_px = skel.sum()
    if skel_px < 2:
        return feat

    dt = distance_transform_edt(crack_mask)
    widths = dt[skel] * 2.0
    feat["crack_mean_width"] = float(widths.mean())
    if feat["crack_mean_width"] > 0:
        feat["crack_width_cv"] = float(widths.std() / (widths.mean() + 1e-6))

    skel_u8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = fftconvolve(skel_u8, kernel, mode='same')
    junctions = skel & (neighbor_count >= 4)
    feat["junction_density"] = float(junctions.sum()) / float(skel_px)

    labeled, n_comp = label(crack_mask)
    feat["crack_components"] = int(n_comp)

    if spalling_px > 0 and crack_px > 0:
        dt_crack = distance_transform_edt(~crack_mask)
        near_crack = dt_crack[spalling_mask] < 10.0
        feat["crack_spalling_proximity"] = float(near_crack.sum()) / (float(spalling_px) + 1e-6)

    return feat


# =========================================================================
# Part 1: Per-tier segmentation analysis
# =========================================================================

def per_tier_class_ratios(train_files):
    tier_stats = defaultdict(lambda: {"total_px": 0, "crack_px": 0, "spalling_px": 0, "bg_px": 0, "n": 0})
    for rel in train_files:
        tier = tier_from_rel(rel)
        m_rgb = read_mask_rgb(mask_path(DATA_ROOT, rel))
        label = decode_mask(m_rgb)
        h, w = label.shape
        s = tier_stats[tier]
        s["total_px"] += h * w
        s["crack_px"] += int((label == 1).sum())
        s["spalling_px"] += int((label == 2).sum())
        s["bg_px"] += int((label == 0).sum())
        s["n"] += 1

    print("\n" + "=" * 70)
    print("PART 1A: Per-tier class pixel ratios")
    print("=" * 70)
    for tier in ["Easy", "Medium", "Hard"]:
        s = tier_stats[tier]
        total = max(s["total_px"], 1)
        print(f"\n{tier} ({s['n']} samples):")
        print(f"  Background: {s['bg_px']/total:.4f}")
        print(f"  Crack:      {s['crack_px']/total:.4f}")
        print(f"  Spalling:   {s['spalling_px']/total:.4f}")
    return tier_stats


def per_tier_morph_stats(train_files):
    tier_morph = defaultdict(list)

    print("\n" + "=" * 70)
    print("PART 1B: Per-tier morphology statistics")
    print("=" * 70)
    print("Computing morphology features...")

    for i, rel in enumerate(train_files):
        tier = tier_from_rel(rel)
        m_rgb = read_mask_rgb(mask_path(DATA_ROOT, rel))
        label = decode_mask(m_rgb)
        feat = compute_morph_features(label)
        tier_morph[tier].append(feat)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(train_files)}]")

    for tier in ["Easy", "Medium", "Hard"]:
        feats = tier_morph[tier]
        has_crack = [f for f in feats if f["has_crack"]]
        print(f"\n{tier} ({len(feats)} samples, {len(has_crack)} with crack):")
        if has_crack:
            widths = [f["crack_mean_width"] for f in has_crack]
            jdens = [f["junction_density"] for f in has_crack]
            comps = [f["crack_components"] for f in has_crack]
            width_cvs = [f["crack_width_cv"] for f in has_crack]
            crack_ratios = [f["crack_ratio"] for f in has_crack]
            prox = [f["crack_spalling_proximity"] for f in has_crack if f["has_spalling"]]
            print(f"  crack_mean_width:  mean={np.mean(widths):.2f}, median={np.median(widths):.2f}, "
                  f"std={np.std(widths):.2f}")
            print(f"  crack_width_cv:    mean={np.mean(width_cvs):.3f}")
            print(f"  junction_density:  mean={np.mean(jdens):.4f}, std={np.std(jdens):.4f}")
            print(f"  crack_components:  mean={np.mean(comps):.1f}, median={np.median(comps):.0f}")
            print(f"  crack_ratio:       mean={np.mean(crack_ratios):.5f}")
            if prox:
                print(f"  crack_spalling_proximity: mean={np.mean(prox):.4f} ({len(prox)} samples)")
        spalling = [f for f in feats if f["has_spalling"]]
        print(f"  has_spalling: {len(spalling)}/{len(feats)}")
    return tier_morph


# =========================================================================
# Part 2: Extra data inventory
# =========================================================================

def classification_data_analysis():
    print("\n" + "=" * 70)
    print("PART 2A: Classification data inventory")
    print("=" * 70)

    crack_dir = CLASSIF_ROOT / "Crack"
    noncrack_dir = CLASSIF_ROOT / "Non-Crack"
    crack_files = sorted(crack_dir.glob("*.*"))
    noncrack_files = sorted(noncrack_dir.glob("*.*"))
    print(f"  Crack images: {len(crack_files)}")
    print(f"  Non-Crack images: {len(noncrack_files)}")

    crack_exts = defaultdict(int)
    for f in crack_files:
        crack_exts[f.suffix.lower()] += 1
    print(f"  Crack extensions: {dict(crack_exts)}")

    noncrack_exts = defaultdict(int)
    for f in noncrack_files:
        noncrack_exts[f.suffix.lower()] += 1
    print(f"  Non-Crack extensions: {dict(noncrack_exts)}")

    # Sample sizes
    print("\n  Sampling image sizes (first 10 Crack):")
    sizes = []
    for f in crack_files[:10]:
        img = cv2.imread(str(f))
        if img is not None:
            sizes.append(img.shape[:2])
    print(f"    Unique sizes: {set(sizes)}")

    print("  Sampling image sizes (first 10 Non-Crack):")
    sizes = []
    for f in noncrack_files[:10]:
        img = cv2.imread(str(f))
        if img is not None:
            sizes.append(img.shape[:2])
    print(f"    Unique sizes: {set(sizes)}")

    # Filename overlap with segmentation
    seg_stems = set()
    for diff in ["Easy", "Medium", "Hard"]:
        img_dir = DATA_ROOT / diff / "Images"
        if img_dir.exists():
            seg_stems.update(f.stem for f in img_dir.glob("*"))

    crack_stems = set(f.stem for f in crack_files)
    noncrack_stems = set(f.stem for f in noncrack_files)

    print(f"\n  Filename overlap with segmentation:")
    print(f"    Crack ∩ Seg: {len(crack_stems & seg_stems)} / {len(crack_stems)}")
    print(f"    NonCrack ∩ Seg: {len(noncrack_stems & seg_stems)} / {len(noncrack_stems)}")


def detection_data_analysis():
    print("\n" + "=" * 70)
    print("PART 2B: Detection data inventory")
    print("=" * 70)

    detect_imgs = sorted((DETECT_ROOT / "Images").glob("*.*"))
    print(f"  Detection images: {len(detect_imgs)}")

    yolo_dir = DETECT_ROOT / "Labels" / "Yolo"
    yolo_files = sorted(yolo_dir.glob("*.txt"))
    print(f"  YOLO label files: {len(yolo_files)}")

    bbox_counts = defaultdict(int)
    class_counts = defaultdict(int)
    bbox_sizes = []

    for yf in yolo_files:
        lines = yf.read_text().strip().split("\n")
        n_boxes = 0
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            cls = int(parts[0])
            class_counts[cls] += 1
            n_boxes += 1
            if len(parts) >= 5:
                w, h = float(parts[3]), float(parts[4])
                bbox_sizes.append((cls, w, h))
        bbox_counts[n_boxes] += 1

    print(f"\n  Bboxes per image distribution:")
    for k in sorted(bbox_counts.keys()):
        print(f"    {k} boxes: {bbox_counts[k]} images")
    print(f"\n  Class counts:")
    print(f"    class 0 (crack): {class_counts.get(0, 0)}")
    print(f"    class 1 (spalling): {class_counts.get(1, 0)}")

    if bbox_sizes:
        crack_boxes = [(w, h) for c, w, h in bbox_sizes if c == 0]
        spalling_boxes = [(w, h) for c, w, h in bbox_sizes if c == 1]
        if crack_boxes:
            cw, ch = zip(*crack_boxes)
            print(f"\n  Crack bbox size (relative):")
            print(f"    width:  mean={np.mean(cw):.3f}, std={np.std(cw):.3f}")
            print(f"    height: mean={np.mean(ch):.3f}, std={np.std(ch):.3f}")
        if spalling_boxes:
            sw, sh = zip(*spalling_boxes)
            print(f"  Spalling bbox size (relative):")
            print(f"    width:  mean={np.mean(sw):.3f}, std={np.std(sw):.3f}")
            print(f"    height: mean={np.mean(sh):.3f}, std={np.std(sh):.3f}")

    # Check overlap with segmentation
    detect_stems = set(f.stem for f in detect_imgs)
    seg_stems = set()
    for diff in ["Easy", "Medium", "Hard"]:
        img_dir = DATA_ROOT / diff / "Images"
        if img_dir.exists():
            seg_stems.update(f.stem for f in img_dir.glob("*"))

    overlap = detect_stems & seg_stems
    print(f"\n  Detection ∩ Segmentation (by filename stem): {len(overlap)} / {len(detect_stems)}")

    det_prefixes = defaultdict(int)
    for f in detect_imgs:
        prefix = f.stem.split(" ")[0] if " " in f.stem else f.stem[0]
        det_prefixes[prefix] += 1
    print(f"  Detection filename prefixes: {dict(det_prefixes)}")

    return class_counts


def detection_bbox_seg_mask_alignment(all_files):
    print("\n" + "=" * 70)
    print("PART 2C: Detection bbox <-> Segmentation mask alignment")
    print("=" * 70)

    voc_dir = DETECT_ROOT / "Labels" / "Pascal VOC"

    seg_stem_to_rel = {}
    for rel in all_files:
        diff, name = rel.split("/", 1)
        stem = Path(name).stem
        seg_stem_to_rel[stem] = rel

    checked = 0
    recall_scores = []
    for voc_file in sorted(voc_dir.glob("*.json")):
        stem = voc_file.stem
        if stem not in seg_stem_to_rel:
            continue

        rel = seg_stem_to_rel[stem]
        m_rgb = read_mask_rgb(mask_path(DATA_ROOT, rel))
        label = decode_mask(m_rgb)
        h, w = label.shape

        with open(voc_file) as f:
            det = json.load(f)

        img_w = det["image"]["width"]
        img_h = det["image"]["height"]

        bbox_mask = np.zeros((img_h, img_w), dtype=bool)
        for ann in det["annotations"]:
            x, y, bw, bh = ann["bbox"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img_w, int(x + bw)), min(img_h, int(y + bh))
            bbox_mask[y1:y2, x1:x2] = True

        if (img_h, img_w) != (h, w):
            bbox_mask = cv2.resize(bbox_mask.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

        fg_mask = (label > 0)
        if fg_mask.sum() == 0:
            continue

        fg_in_bbox = (fg_mask & bbox_mask).sum()
        fg_total = fg_mask.sum()
        recall = fg_in_bbox / fg_total
        recall_scores.append(recall)
        checked += 1

        if checked <= 5:
            print(f"  {rel}: fg_recall_in_bbox={recall:.3f} "
                  f"(fg={fg_total}, in_bbox={fg_in_bbox})")

    if recall_scores:
        print(f"\n  Checked {checked} overlapping images")
        print(f"  FG recall in bboxes: mean={np.mean(recall_scores):.3f}, "
              f"median={np.median(recall_scores):.3f}, min={np.min(recall_scores):.3f}")
        print(f"  100% coverage: {sum(1 for s in recall_scores if s > 0.99)}/{checked}")
        print(f"  >90% coverage: {sum(1 for s in recall_scores if s > 0.90)}/{checked}")
        print(f"  >80% coverage: {sum(1 for s in recall_scores if s > 0.80)}/{checked}")
    else:
        print("  No overlapping images found for alignment check.")


# =========================================================================
# Main
# =========================================================================

def main():
    train_txt = SPLITS_DIR / "train.txt"
    train_files = read_split_file(train_txt)
    print(f"Loaded {len(train_files)} training files")

    tier_counts = defaultdict(int)
    for rel in train_files:
        tier_counts[tier_from_rel(rel)] += 1
    print(f"Per-tier: {dict(tier_counts)}")

    all_files = list(train_files)
    for split in ["val", "test"]:
        split_file = SPLITS_DIR / f"{split}.txt"
        if split_file.exists():
            all_files.extend(read_split_file(split_file))

    # Part 1
    per_tier_class_ratios(train_files)
    per_tier_morph_stats(train_files)

    # Part 2
    classification_data_analysis()
    detection_data_analysis()
    detection_bbox_seg_mask_alignment(all_files)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

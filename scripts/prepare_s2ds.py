"""Prepare the S2DS dataset for cross-dataset evaluation.

S2DS (Structural Defects Dataset) contains 743 images (1024x1024) of concrete
surfaces with 7-class pixel-level annotations (RGBA PNGs). We map to our
3-class scheme:
    S2DS 1 (Crack)    -> our class 1 (crack)
    S2DS 2 (Spalling) -> our class 2 (spalling)
    S2DS 0,3,4,5,6    -> our class 0 (background)

Expected input layout (already downloaded at Dataset/s2ds/):
    Dataset/s2ds/train/{id}.png + {id}_lab.png   (563 pairs)
    Dataset/s2ds/val/{id}.png   + {id}_lab.png   (87 pairs)
    Dataset/s2ds/test/{id}.png  + {id}_lab.png   (93 pairs)

Usage:
    python scripts/prepare_s2ds.py
    python scripts/prepare_s2ds.py --split test          # test only
    python scripts/prepare_s2ds.py --raw-dir /other/path

Output:
    Dataset/S2DS/images/{split}_{id}.png   (resized to 512x512)
    Dataset/S2DS/masks/{split}_{id}.png    (3-class index mask, 512x512)
    Dataset/S2DS/test_files.txt            (test split file list)
    Dataset/S2DS/all_files.txt             (all splits)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CODES_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW = CODES_DIR / "Dataset" / "s2ds"
OUTPUT_DIR = CODES_DIR / "Dataset" / "S2DS"
TARGET_SIZE = 512

# S2DS ground-truth label colors (RGB, after lab[lab<200]=0 in original code)
# From s2ds/utils/utils.py lab2class():
#   black   (0,0,0)       -> 0: background
#   white   (255,255,255) -> 1: crack
#   red     (255,0,0)     -> 2: spalling
#   yellow  (255,255,0)   -> 3: corrosion      -> bg
#   cyan    (0,255,255)   -> 4: efflorescence   -> bg
#   green   (0,255,0)     -> 5: vegetation      -> bg
#   blue    (0,0,255)     -> 6: control point   -> bg

S2DS_COLORS_RGB = np.array([
    [0, 0, 0],          # 0 bg
    [255, 255, 255],    # 1 crack
    [255, 0, 0],        # 2 spalling
    [255, 255, 0],      # 3 corrosion
    [0, 255, 255],      # 4 efflorescence
    [0, 255, 0],        # 5 vegetation
    [0, 0, 255],        # 6 control point
], dtype=np.uint8)

# Map S2DS class -> our 3-class
S2DS_TO_OURS = np.array([0, 1, 2, 0, 0, 0, 0], dtype=np.uint8)


def s2ds_lab2class(lab_rgb: np.ndarray) -> np.ndarray:
    """Replicate S2DS lab2class: RGB label -> integer class map."""
    lab = lab_rgb.copy()
    lab[lab < 200] = 0
    out = np.zeros(lab.shape[:2], dtype=np.uint8)
    for cls_id in range(1, 7):
        color = S2DS_COLORS_RGB[cls_id]
        out[(lab == color).all(axis=2)] = cls_id
    return out


def remap_to_3class(s2ds_mask: np.ndarray) -> np.ndarray:
    """Map S2DS 7-class -> our 3-class (bg=0, crack=1, spalling=2)."""
    return S2DS_TO_OURS[s2ds_mask]


def process_split(raw_dir: Path, split: str, output_dir: Path, target_size: int):
    """Process one split (train/val/test)."""
    split_dir = raw_dir / split
    if not split_dir.exists():
        print(f"  [WARN] Split dir not found: {split_dir}")
        return []

    img_out = output_dir / "images"
    mask_out = output_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    lab_paths = sorted(split_dir.glob("*_lab.png"))
    print(f"  [{split}] Found {len(lab_paths)} labels")

    file_list = []
    stats = {"n": 0, "crack": 0, "spalling": 0, "crack_px": 0, "spalling_px": 0, "total_px": 0}

    for lab_path in lab_paths:
        stem = lab_path.stem.replace("_lab", "")
        img_path = split_dir / f"{stem}.png"
        if not img_path.exists():
            img_path = split_dir / f"{stem}.jpg"
        if not img_path.exists():
            print(f"    [WARN] No image for {lab_path.name}")
            continue

        # Read image
        img = Image.open(img_path).convert("RGB")
        # Read label (may be RGBA)
        lab = Image.open(lab_path).convert("RGB")
        lab_np = np.array(lab)

        # Decode: RGB -> 7-class -> 3-class
        s2ds_mask = s2ds_lab2class(lab_np)
        our_mask = remap_to_3class(s2ds_mask)

        # Resize
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        # Mask: use nearest neighbor
        mask_pil = Image.fromarray(our_mask)
        mask_resized = mask_pil.resize((target_size, target_size), Image.NEAREST)

        # Save with split prefix to avoid id collisions across splits
        out_name = f"{split}_{stem}"
        img_resized.save(str(img_out / f"{out_name}.png"))
        mask_resized.save(str(mask_out / f"{out_name}.png"))

        file_list.append(out_name)

        # Stats
        mask_arr = np.array(mask_resized)
        stats["n"] += 1
        stats["crack"] += int((mask_arr == 1).any())
        stats["spalling"] += int((mask_arr == 2).any())
        stats["crack_px"] += int((mask_arr == 1).sum())
        stats["spalling_px"] += int((mask_arr == 2).sum())
        stats["total_px"] += mask_arr.size

    print(f"    Processed: {stats['n']}")
    print(f"    With crack: {stats['crack']}, with spalling: {stats['spalling']}")
    if stats["total_px"] > 0:
        print(f"    Crack px: {stats['crack_px']/stats['total_px']*100:.2f}%, "
              f"Spalling px: {stats['spalling_px']/stats['total_px']*100:.2f}%")

    return file_list


def main():
    parser = argparse.ArgumentParser(description="Prepare S2DS for cross-dataset eval")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW),
                        help="Path to extracted s2ds/ with train/val/test subdirs")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Process only one split (default: all)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not raw_dir.exists():
        print(f"[ERROR] Raw dir not found: {raw_dir}")
        return

    splits = [args.split] if args.split else ["train", "val", "test"]

    print(f"[prepare_s2ds] Raw: {raw_dir}")
    print(f"[prepare_s2ds] Output: {output_dir}")
    print(f"[prepare_s2ds] Target size: {args.target_size}")

    all_files = []
    test_files = []

    for split in splits:
        files = process_split(raw_dir, split, output_dir, args.target_size)
        all_files.extend(files)
        if split == "test":
            test_files = files

    # Write file lists
    if test_files:
        p = output_dir / "test_files.txt"
        with open(p, "w") as f:
            for name in sorted(test_files):
                f.write(f"{name}\n")
        print(f"\n[prepare_s2ds] Test file list ({len(test_files)}): {p}")

    if all_files:
        p = output_dir / "all_files.txt"
        with open(p, "w") as f:
            for name in sorted(all_files):
                f.write(f"{name}\n")
        print(f"[prepare_s2ds] All file list ({len(all_files)}): {p}")

    print(f"\n[prepare_s2ds] Done. Total: {len(all_files)} images")


if __name__ == "__main__":
    main()

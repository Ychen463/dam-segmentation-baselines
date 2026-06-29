#!/usr/bin/env python
"""Fig 3: Qualitative comparison — input / GT / model predictions side by side.

Usage (on RunPod where checkpoints exist):
    python scripts/fig3_qualitative.py
    python scripts/fig3_qualitative.py --models segformer_b2_plain_512 dataopt_baseline_N0
    python scripts/fig3_qualitative.py --samples "Hard/H (184).jpg" "Easy/E (255).jpg"

Produces a grid:  rows = selected images,  cols = [Image, GT, Model1, Model2, ...]
Each prediction cell shows the overlay + per-class IoU for that image.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

# ---- paths ----
CODES_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = CODES_DIR / "Dataset" / "DamSegment" / "Damage Segmentaion"
SPLITS_DIR = CODES_DIR / "baseline_unet" / "splits"

# ---- helpers ----
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
    label[r > 127] = 1
    label[b > 127] = 2
    return label

PALETTE = np.array([
    [0, 0, 0],        # background — black
    [255, 60, 60],     # crack — red
    [60, 200, 60],     # spalling — green
], dtype=np.uint8)

def colorize(label):
    """Label (H,W) -> RGB (H,W,3)."""
    return PALETTE[label]

def overlay(img, label, alpha=0.5):
    vis = img.copy()
    for cid, color in [(1, [255, 60, 60]), (2, [60, 200, 60])]:
        mask = label == cid
        if mask.any():
            vis[mask] = (np.array(color) * alpha + vis[mask] * (1 - alpha)).astype(np.uint8)
    return vis

def per_image_iou(pred, gt, cid):
    p = pred == cid
    g = gt == cid
    inter = (p & g).sum()
    union = (p | g).sum()
    if union == 0:
        return float('nan')
    return inter / union

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def preprocess(img_rgb, img_size):
    """Resize + normalize + to tensor (1,3,H,W)."""
    h, w = img_size, img_size
    resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
    return x


# ---- display name mapping ----
DISPLAY_NAMES = {
    "unet_r34_320": "U-Net",
    "deeplabv3p_r50_512": "DeepLabV3+",
    "deeplabv3p_r34_320": "DeepLabV3+ (R34)",
    "segformer_b2_plain_512": "SegFormer",
    "segformer_b2_staticcurr_512": "SegFormer+Curr",
    "mask2former_swin_small_512": "Mask2Former",
    "dscformer_plain_G0": "DSConv",
    "dscformer_srl_G1": "DSConv+SRL",
    "dkd10_no_srl": "TopoDistill",
    "dataopt_baseline_N0": "DSConv+SRL (T1)",
}


def pick_diverse_samples(split_file, n=4, seed=42):
    """Pick samples from different tiers that have both crack+spalling."""
    with open(split_file) as f:
        files = [l.strip() for l in f if l.strip()]
    rng = random.Random(seed)

    picks = []
    for tier in ["Easy", "Medium", "Hard"]:
        tier_files = [f for f in files if f.startswith(tier + "/")]
        good = []
        for rel in tier_files:
            m = read_rgb(mask_path(DATA_ROOT, rel))
            label = decode_mask(m)
            if (label == 1).any() and (label == 2).any():
                good.append(rel)
        rng.shuffle(good)
        # Take 1-2 from each tier
        take = 2 if tier == "Hard" else 1
        picks.extend(good[:take])

    return picks[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["segformer_b2_plain_512",
                                 "dscformer_srl_G1",
                                 "dkd10_no_srl"])
    parser.add_argument("--samples", nargs="+", default=None,
                        help="Specific sample rel paths (e.g. 'Hard/H (184).jpg')")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--output", default="figures/fig3_qualitative.pdf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[fig3] device: {device}")

    # Pick samples
    if args.samples:
        samples = args.samples
    else:
        split_file = SPLITS_DIR / "test.txt"
        samples = pick_diverse_samples(split_file, n=args.n_samples, seed=args.seed)
    print(f"[fig3] samples: {samples}")

    # Load models
    import sys
    sys.path.insert(0, str(CODES_DIR))
    from shared_eval.model_registry import get as get_entry, load_model

    models = {}
    for mname in args.models:
        try:
            models[mname] = load_model(mname, device=device)
            print(f"[fig3] loaded: {mname}")
        except Exception as e:
            print(f"[fig3] SKIP {mname}: {e}")

    n_models = len(models)
    n_rows = len(samples)
    n_cols = 2 + n_models  # Image, GT, model1, model2, ...

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers (row 0 only)
    col_headers = ["Input", "Ground Truth"] + [
        DISPLAY_NAMES.get(m, m) for m in models.keys()
    ]

    for i, rel in enumerate(samples):
        # Read raw image and GT
        img_raw = read_rgb(image_path(DATA_ROOT, rel))
        mask_rgb = read_rgb(mask_path(DATA_ROOT, rel))
        gt = decode_mask(mask_rgb)

        tier = rel.split("/")[0]
        row_label = f"{tier}: {Path(rel).stem}"

        # Col 0: Image with row label
        axes[i, 0].imshow(img_raw)
        if i == 0:
            axes[i, 0].set_title(f"{col_headers[0]}\n{row_label}",
                                  fontsize=11, fontweight="bold", pad=6)
        else:
            axes[i, 0].set_title(row_label, fontsize=9, pad=6)
        axes[i, 0].axis("off")

        # Col 1: GT overlay
        axes[i, 1].imshow(overlay(img_raw, gt))
        if i == 0:
            axes[i, 1].set_title(col_headers[1], fontsize=11,
                                  fontweight="bold", pad=10)
        axes[i, 1].axis("off")

        # Cols 2+: Model predictions
        for j, (mname, model) in enumerate(models.items()):
            entry = get_entry(mname)
            img_size = entry.img_size
            x = preprocess(img_raw, img_size).to(device)

            with torch.no_grad():
                logits = model(x)

            # Resize logits to original image size for visualization
            orig_h, orig_w = img_raw.shape[:2]
            logits_up = F.interpolate(logits, (orig_h, orig_w),
                                      mode="bilinear", align_corners=False)
            pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()

            # Compute per-image IoU
            iou_cr = per_image_iou(pred, gt, 1)
            iou_sp = per_image_iou(pred, gt, 2)

            ax = axes[i, 2 + j]
            ax.imshow(overlay(img_raw, pred))
            iou_str = f"Cr:{iou_cr:.2f} Sp:{iou_sp:.2f}" if not (
                np.isnan(iou_cr) and np.isnan(iou_sp)) else ""
            # Column header on first row, IoU scores on all rows
            if i == 0:
                ax.set_title(f"{col_headers[2 + j]}\n{iou_str}",
                             fontsize=11, fontweight="bold", pad=6)
            else:
                ax.set_title(iou_str, fontsize=9, pad=6)
            ax.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=(1, 0.24, 0.24), label="Crack"),
        mpatches.Patch(color=(0.24, 0.78, 0.24), label="Spalling"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    png_path = out_path.with_suffix(".png")
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

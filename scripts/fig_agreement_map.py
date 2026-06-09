#!/usr/bin/env python
"""Fig: Teacher agreement map visualization.

Shows for 2-3 samples:
  Input | GT | T1 pred | T2 pred | Disagreement heatmap | Student pred

Illustrates WHERE teachers disagree and how confidence-aware KD uses that signal.

Usage (on RunPod):
    python scripts/fig_agreement_map.py
    python scripts/fig_agreement_map.py --samples "Hard/H (184).jpg" "Hard/H (332).jpg"
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F

CODES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODES_DIR))

DATA_ROOT = CODES_DIR / "Dataset" / "DamSegment" / "Damage Segmentaion"
SPLITS_DIR = CODES_DIR / "baseline_unet" / "splits"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


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


def overlay(img, label, alpha=0.45):
    vis = img.copy()
    cr = label == 1
    sp = label == 2
    if cr.any():
        vis[cr] = (np.array([255, 60, 60]) * alpha +
                   vis[cr] * (1 - alpha)).astype(np.uint8)
    if sp.any():
        vis[sp] = (np.array([60, 200, 60]) * alpha +
                   vis[sp] * (1 - alpha)).astype(np.uint8)
    return vis


def preprocess(img_rgb, img_size=512):
    h, w = img_size, img_size
    resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()


def pick_hard_samples(split_file, n=3, seed=42):
    with open(split_file) as f:
        files = [l.strip() for l in f if l.strip()]
    hard = [f for f in files if f.startswith("Hard/")]
    both = []
    for rel in hard:
        m = read_rgb(mask_path(DATA_ROOT, rel))
        label = decode_mask(m)
        if (label == 1).any() and (label == 2).any():
            both.append(rel)
    rng = random.Random(seed)
    rng.shuffle(both)
    return both[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--output", default="figures/fig_agreement_map.pdf")
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=4.0)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fig_agreement] device: {device}")

    # Pick samples
    if args.samples:
        samples = args.samples
    else:
        split_file = SPLITS_DIR / "test.txt"
        samples = pick_hard_samples(split_file, n=args.n_samples)
    print(f"[fig_agreement] samples: {samples}")

    # Load models
    from full_method import config as C
    from full_method.config import RunCfg, apply_preset
    from full_method.model import DSCformerDam
    from full_method.sam_model import TopoLoRASAM

    cfg = RunCfg()
    apply_preset(cfg, "DKD10")

    # Teacher 1
    t1_path = (C.PKG_DIR / cfg.kd_teacher_checkpoint).resolve()
    print(f"[fig_agreement] loading T1 from {t1_path}")
    t1_state = torch.load(t1_path, map_location=device, weights_only=False)
    t1 = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)
    t1.load_state_dict(t1_state["model"])
    t1.eval()

    # Teacher 2
    t2_path = (C.PKG_DIR / cfg.kd_teacher2_checkpoint).resolve()
    print(f"[fig_agreement] loading T2 from {t2_path}")
    t2_state = torch.load(t2_path, map_location=device, weights_only=False)
    t2 = TopoLoRASAM(
        sam_checkpoint=cfg.sam_checkpoint,
        num_classes=C.NUM_CLASSES,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        fpn_dim=cfg.sam_fpn_dim,
        sam_img_size=cfg.sam_img_size,
    ).to(device)
    t2.load_state_dict(t2_state["model"])
    t2.eval()

    # Student (DKD10)
    student_path = C.PKG_DIR / "runs" / "dkd10_no_srl" / "best.pt"
    print(f"[fig_agreement] loading student from {student_path}")
    s_state = torch.load(student_path, map_location=device, weights_only=False)
    student = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)
    student.load_state_dict(s_state["model"])
    student.eval()

    tau = args.temperature
    n_rows = len(samples)
    col_titles = ["Input", "Ground Truth", "Teacher 1\n(TopoDistill)",
                  "Teacher 2\n(SAM-LoRA)", "Disagreement\nMap", "Student\n(DTKD)"]
    n_cols = len(col_titles)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, rel in enumerate(samples):
        img_raw = read_rgb(image_path(DATA_ROOT, rel))
        mask_rgb = read_rgb(mask_path(DATA_ROOT, rel))
        gt = decode_mask(mask_rgb)
        orig_h, orig_w = img_raw.shape[:2]

        x = preprocess(img_raw, 512).to(device)

        with torch.no_grad():
            t1_out = t1(x)
            t1_logits = F.interpolate(t1_out["seg_logits"].float(),
                                       (orig_h, orig_w), mode="bilinear",
                                       align_corners=False)
            t2_out = t2(x)
            t2_logits = F.interpolate(t2_out["seg_logits"].float(),
                                       (orig_h, orig_w), mode="bilinear",
                                       align_corners=False)
            s_out = student(x)
            s_logits = F.interpolate(s_out["seg_logits"].float(),
                                      (orig_h, orig_w), mode="bilinear",
                                      align_corners=False)

            # Teacher disagreement: KL(T1 || T2) per pixel
            t1_prob = F.softmax(t1_logits / tau, dim=1)
            t2_prob = F.softmax(t2_logits / tau, dim=1)
            kl = (t1_prob * (t1_prob.clamp_min(1e-8).log()
                             - t2_prob.clamp_min(1e-8).log())).sum(1)  # (1,H,W)
            kl_map = kl.squeeze(0).cpu().numpy()  # (H,W)

        t1_pred = t1_logits.argmax(1).squeeze(0).cpu().numpy()
        t2_pred = t2_logits.argmax(1).squeeze(0).cpu().numpy()
        s_pred = s_logits.argmax(1).squeeze(0).cpu().numpy()

        # Col 0: Input
        axes[i, 0].imshow(img_raw)
        tier = rel.split("/")[0]
        axes[i, 0].set_title(f"{tier}: {Path(rel).stem}" if i == 0
                              else Path(rel).stem, fontsize=9)

        # Col 1: GT
        axes[i, 1].imshow(overlay(img_raw, gt))

        # Col 2: T1 prediction
        axes[i, 2].imshow(overlay(img_raw, t1_pred))

        # Col 3: T2 prediction
        axes[i, 3].imshow(overlay(img_raw, t2_pred))

        # Col 4: Disagreement heatmap
        # Normalize for visualization
        vmax = max(kl_map.max(), 0.01)
        axes[i, 4].imshow(img_raw, alpha=0.3)
        im = axes[i, 4].imshow(kl_map, cmap="hot", alpha=0.7,
                                vmin=0, vmax=vmax)
        plt.colorbar(im, ax=axes[i, 4], fraction=0.046, pad=0.04)

        # Col 5: Student prediction
        axes[i, 5].imshow(overlay(img_raw, s_pred))

        for j in range(n_cols):
            axes[i, j].axis("off")

    # Column titles
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color=(1, 0.24, 0.24), label="Crack"),
        mpatches.Patch(color=(0.24, 0.78, 0.24), label="Spalling"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=10, frameon=True, bbox_to_anchor=(0.35, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = CODES_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    png_path = out_path.with_suffix(".png")
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

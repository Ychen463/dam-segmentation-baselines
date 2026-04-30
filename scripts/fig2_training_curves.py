#!/usr/bin/env python
"""Fig 2: Training curves — val mIoU_fg and train loss vs epoch for main models.

Usage:
    python scripts/fig2_training_curves.py [--output figures/fig2_training_curves.pdf]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CODES_DIR = Path(__file__).resolve().parent.parent

# Models to plot: (display_name, csv_path_relative_to_CODES_DIR, color, linestyle)
MODELS = [
    ("U-Net (R34)",
     "baseline_unet/runs/unet_r34_ce_dice/metrics.csv",
     "#e07b39", "--"),
    ("DeepLabV3+ (R50)",
     "baseline_deeplab/runs/deeplabv3p_r50_512/metrics.csv",
     "#d62728", "--"),
    ("Mask2Former (Swin-S)",
     "baseline_mask2former/runs/mask2former_plain_M0/metrics.csv",
     "#9467bd", "--"),
    ("SegFormer-B2",
     "baseline_segformer/runs/segformer_b2_plain_512/metrics.csv",
     "#2ca02c", "-."),
    ("+ DSConv (G0)",
     "full_method/runs/dscformer_plain_G0/metrics.csv",
     "#8c564b", "-."),
    ("Ours: + DSConv + SRL (G1)",
     "full_method/runs/dscformer_srl_G1/metrics.csv",
     "#1f77b4", "-"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/fig2_training_curves.pdf")
    args = parser.parse_args()

    out_path = CODES_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, csv_rel, color, ls in MODELS:
        csv_path = CODES_DIR / csv_rel
        if not csv_path.exists():
            print(f"  [skip] {name}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # Filter validation rows (some CSVs lack 'split' column)
        if "split" in df.columns:
            df_val = df[df["split"] == "val"].copy()
        else:
            df_val = df.copy()
        df_val = df_val.sort_values("epoch")

        # Left: val mIoU_fg
        ax1.plot(df_val["epoch"], df_val["mIoU_fg"],
                 label=name, color=color, linestyle=ls, linewidth=1.8)

        # Right: train loss (skip Mask2Former due to different loss scale)
        if "Mask2Former" not in name:
            ax2.plot(df_val["epoch"], df_val["train_loss"],
                     label=name, color=color, linestyle=ls, linewidth=1.8)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Validation mIoU (foreground)", fontsize=12)
    ax1.set_title("(a) Validation mIoU$_{fg}$", fontsize=13)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.35, 0.75)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Training Loss", fontsize=12)
    ax2.set_title("(b) Training Loss", fontsize=13)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    png_path = out_path.with_suffix(".png")
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

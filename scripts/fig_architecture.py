#!/usr/bin/env python
"""Method overview architecture diagram using matplotlib.

Usage:
    python scripts/fig_architecture.py [--output figures/fig_architecture.pdf]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CODES_DIR = Path(__file__).resolve().parent.parent

# Colors
C_ENC = "#4A90D9"
C_DEC = "#27AE60"
C_SEG = "#E74C3C"
C_BD  = "#E67E22"
C_LOSS = "#F1C40F"
C_DIFF = "#8E44AD"
C_SAMP = "#16A085"
C_SCHED = "#7F8C8D"
C_INPUT = "#95A5A6"
C_BG1 = "#D6EAF8"
C_BG2 = "#E8DAEF"


def draw_box(ax, x, y, w, h, text, color, fontsize=8, alpha=0.3, lw=1.2,
             textcolor="black", fontstyle="normal", fontweight="normal"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor="black",
                          alpha=alpha, linewidth=lw, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight=fontweight, fontstyle=fontstyle,
            zorder=3, family="sans-serif")
    return box


def draw_arrow(ax, x1, y1, x2, y2, color="black", lw=1.0, style="-",
               alpha=0.7, connectionstyle="arc3,rad=0"):
    ls = "-" if style == "-" else "--"
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle="->,head_width=3,head_length=3",
                           connectionstyle=connectionstyle,
                           color=color, linewidth=lw, linestyle=ls,
                           alpha=alpha, zorder=1)
    ax.add_patch(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/fig_architecture.pdf")
    args = parser.parse_args()

    out_path = CODES_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # ============================================================
    # Background regions
    # ============================================================
    # Network architecture bg
    bg1 = FancyBboxPatch((-0.3, 3.2), 14.8, 6.5, boxstyle="round,pad=0.2",
                          facecolor=C_BG1, edgecolor=C_ENC, alpha=0.25,
                          linewidth=1.5, linestyle="--", zorder=0)
    ax.add_patch(bg1)
    ax.text(0.2, 9.4, "Network Architecture", fontsize=11, fontweight="bold",
            color=C_ENC, family="sans-serif", alpha=0.8)

    # Training dynamics bg
    bg2 = FancyBboxPatch((-0.3, -0.8), 16.5, 3.8, boxstyle="round,pad=0.2",
                          facecolor=C_BG2, edgecolor=C_DIFF, alpha=0.2,
                          linewidth=1.5, linestyle="--", zorder=0)
    ax.add_patch(bg2)
    ax.text(13.5, -0.5, "Training Dynamics", fontsize=11, fontweight="bold",
            color=C_DIFF, family="sans-serif", alpha=0.8)

    # ============================================================
    # INPUT
    # ============================================================
    draw_box(ax, 0.8, 6.5, 1.4, 2.4, "Input\nImage\n$H{\\times}W$",
             C_INPUT, fontsize=9, alpha=0.35)

    # ============================================================
    # ENCODER STAGES
    # ============================================================
    enc_x = 3.5
    enc_stages = [
        ("Stage 1\nH/4, 64ch", 8.0, 0.20),
        ("Stage 2\nH/8, 128ch", 6.9, 0.30),
        ("Stage 3\nH/16, 320ch", 5.8, 0.40),
        ("Stage 4\nH/32, 512ch", 4.7, 0.50),
    ]
    ax.text(enc_x, 9.0, "SegFormer Encoder", fontsize=10, fontweight="bold",
            ha="center", color=C_ENC, family="sans-serif")

    for text, y, a in enc_stages:
        draw_box(ax, enc_x, y, 1.8, 0.7, text, C_ENC, fontsize=7, alpha=a)

    # Input → Encoder arrows
    for _, y, _ in enc_stages:
        draw_arrow(ax, 1.5, 6.5, enc_x - 0.9, y, color=C_INPUT, lw=0.8)

    # ============================================================
    # DECODER
    # ============================================================
    dec_x = 6.8
    dec_y = 6.5
    draw_box(ax, dec_x, dec_y, 2.0, 1.2, "MLP Decoder\nLinear Fuse\n768ch",
             C_DEC, fontsize=8, alpha=0.35)
    ax.text(dec_x, 7.5, "Decoder", fontsize=10, fontweight="bold",
            ha="center", color=C_DEC, family="sans-serif")

    # Encoder → Decoder
    for _, y, _ in enc_stages:
        draw_arrow(ax, enc_x + 0.9, y, dec_x - 1.0, dec_y, color=C_ENC, lw=0.8)

    # ============================================================
    # SEGMENTATION HEAD
    # ============================================================
    seg_x, seg_y = 10.2, 7.4
    draw_box(ax, seg_x, seg_y, 2.0, 0.8, "Segmentation Head",
             C_SEG, fontsize=8, alpha=0.30)

    # BOUNDARY HEAD
    bd_x, bd_y = 10.2, 5.6
    draw_box(ax, bd_x, bd_y, 2.0, 0.8, "Boundary Head",
             C_BD, fontsize=8, alpha=0.30)

    # Decoder → Heads
    draw_arrow(ax, dec_x + 1.0, dec_y + 0.3, seg_x - 1.0, seg_y, color=C_DEC, lw=0.9)
    draw_arrow(ax, dec_x + 1.0, dec_y - 0.3, bd_x - 1.0, bd_y, color=C_DEC, lw=0.9)

    # ============================================================
    # OUTPUTS
    # ============================================================
    out_x = 13.0
    draw_box(ax, out_x, seg_y, 1.8, 0.7, "Seg Logits\n3 classes",
             C_SEG, fontsize=7, alpha=0.20)
    draw_box(ax, out_x, bd_y, 1.8, 0.7, "Boundary\nMap",
             C_BD, fontsize=7, alpha=0.20)

    draw_arrow(ax, seg_x + 1.0, seg_y, out_x - 0.9, seg_y, color=C_SEG, lw=0.9)
    draw_arrow(ax, bd_x + 1.0, bd_y, out_x - 0.9, bd_y, color=C_BD, lw=0.9)

    # ============================================================
    # GROUND TRUTH
    # ============================================================
    draw_box(ax, 0.8, 3.8, 1.4, 0.8, "Ground\nTruth",
             C_INPUT, fontsize=8, alpha=0.35)

    # ============================================================
    # COMPOSITE LOSS
    # ============================================================
    loss_x, loss_y = 7.5, 1.8
    loss_w, loss_h = 9.0, 1.2
    draw_box(ax, loss_x, loss_y, loss_w, loss_h, "",
             C_LOSS, fontsize=8, alpha=0.25)
    ax.text(loss_x, 2.6, "Composite Loss  $\\mathcal{L}$", fontsize=10,
            fontweight="bold", ha="center", color="#7D6608", family="sans-serif")

    # Loss sub-components
    loss_names = ["$\\mathcal{L}_{CE}$", "$\\mathcal{L}_{Dice}$",
                  "$\\mathcal{L}_{Tversky}$", "$\\mathcal{L}_{Bd}$",
                  "$\\mathcal{L}_{SRL}$"]
    loss_colors = [C_LOSS]*5
    for i, (name, lc) in enumerate(zip(loss_names, loss_colors)):
        lx = loss_x - 3.2 + i * 1.6
        draw_box(ax, lx, loss_y - 0.05, 1.2, 0.55, name, lc,
                 fontsize=9, alpha=0.50)

    # Formula below
    ax.text(loss_x, 0.9,
            "$\\mathcal{L} = \\lambda_1\\mathcal{L}_{CE} + \\lambda_2\\mathcal{L}_{Dice}"
            " + \\lambda_3\\mathcal{L}_{Tversky} + \\lambda_4\\mathcal{L}_{Bd}"
            " + \\lambda_5\\mathcal{L}_{SRL}$",
            ha="center", va="center", fontsize=8, color="black",
            family="sans-serif", alpha=0.7)

    # Seg out → Loss
    draw_arrow(ax, out_x, seg_y - 0.35, out_x, loss_y + 0.6,
               color=C_SEG, lw=0.9)
    # Bd out → Loss
    draw_arrow(ax, out_x, bd_y - 0.35, out_x, loss_y + 0.6,
               color=C_BD, lw=0.9)
    # GT → Loss
    draw_arrow(ax, 1.5, 3.8, loss_x - loss_w/2 + 0.2, loss_y,
               color=C_INPUT, lw=0.9)

    # ============================================================
    # DIFFICULTY ESTIMATOR (below loss, left)
    # ============================================================
    diff_x, diff_y = 4.5, -0.1
    draw_box(ax, diff_x, diff_y, 5.0, 1.0, "", C_DIFF, fontsize=8, alpha=0.20)
    ax.text(diff_x, 0.55, "Online Difficulty Estimator", fontsize=9,
            fontweight="bold", ha="center", color=C_DIFF, family="sans-serif")

    # Sub-components
    diff_labels = ["Loss $\\ell_i$", "Uncert $u_i$",
                   "Boundary $b_i$", "Sparsity $s_i$"]
    for i, dl in enumerate(diff_labels):
        dx = diff_x - 1.8 + i * 1.2
        draw_box(ax, dx, diff_y - 0.05, 1.0, 0.40, dl, C_DIFF,
                 fontsize=6, alpha=0.40)

    # Formula
    ax.text(diff_x, -0.75,
            "$d_i = \\alpha\\hat{\\ell}_i + \\beta\\hat{u}_i"
            " + \\gamma\\hat{b}_i + \\delta\\hat{s}_i$  (EMA + z-score)",
            ha="center", va="center", fontsize=7, color=C_DIFF,
            family="sans-serif", alpha=0.8)

    # Loss → Difficulty (dashed)
    draw_arrow(ax, loss_x - 2.0, loss_y - 0.6, diff_x, diff_y + 0.5,
               color=C_DIFF, lw=0.8, style="--")

    # ============================================================
    # DYNAMIC CURRICULUM SAMPLER (below loss, right)
    # ============================================================
    samp_x, samp_y = 11.0, -0.1
    draw_box(ax, samp_x, samp_y, 4.0, 1.0, "", C_SAMP, fontsize=8, alpha=0.20)
    ax.text(samp_x, 0.55, "Dynamic Curriculum Sampler", fontsize=9,
            fontweight="bold", ha="center", color=C_SAMP, family="sans-serif")

    tier_labels = ["Easy", "Medium", "Hard"]
    tier_alphas = [0.30, 0.45, 0.60]
    for i, (tl, ta) in enumerate(zip(tier_labels, tier_alphas)):
        tx = samp_x - 1.2 + i * 1.2
        draw_box(ax, tx, samp_y - 0.05, 0.9, 0.40, tl, C_SAMP,
                 fontsize=7, alpha=ta)

    ax.text(samp_x, -0.75, "Softmax-weighted tier-gated sampling",
            ha="center", va="center", fontsize=7, color=C_SAMP,
            family="sans-serif", alpha=0.8)

    # Difficulty → Sampler
    draw_arrow(ax, diff_x + 2.5, diff_y, samp_x - 2.0, samp_y,
               color=C_DIFF, lw=1.0)
    ax.text(7.8, 0.2, "$d_i$ scores", fontsize=7, ha="center",
            color=C_DIFF, family="sans-serif", alpha=0.7)

    # ============================================================
    # STAGE SCHEDULER
    # ============================================================
    sched_x, sched_y = 15.0, -0.1
    draw_box(ax, sched_x, sched_y, 1.5, 1.0, "Stage\nScheduler\n0-30-70%",
             C_SCHED, fontsize=7, alpha=0.25)
    draw_arrow(ax, sched_x - 0.75, sched_y, samp_x + 2.0, samp_y,
               color=C_SCHED, lw=0.8)

    # ============================================================
    # FEEDBACK LOOP: Sampler → Input (next epoch)
    # ============================================================
    draw_arrow(ax, samp_x - 1.5, samp_y - 0.5, 0.8, 3.4,
               color=C_SAMP, lw=0.8, style="--",
               connectionstyle="arc3,rad=-0.3")
    ax.text(4.5, 2.0, "Next epoch\nsample selection", fontsize=7,
            ha="center", color=C_SAMP, family="sans-serif", alpha=0.6,
            rotation=0)

    # ============================================================
    # TITLE
    # ============================================================
    ax.text(7.5, 9.8,
            "Dynamic Difficulty-Aware Curriculum Learning with Boundary Refinement",
            ha="center", va="center", fontsize=13, fontweight="bold",
            family="sans-serif", color="black")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=250, bbox_inches="tight")
    print(f"Saved: {out_path}")
    png_path = out_path.with_suffix(".png")
    fig.savefig(str(png_path), dpi=250, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

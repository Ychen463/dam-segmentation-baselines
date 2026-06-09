#!/usr/bin/env python
"""Method overview architecture diagram — vertical (portrait) layout.

Matches the paper's final method:
  (a) TopoDistill: SegFormer-B2 + parallel DSConv branch
  (b) DTKD: class-conditional dual-teacher KD with confidence-aware distillation

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

# ── Colour palette ──────────────────────────────────────────────────
C_ENC   = "#4A90D9"   # encoder
C_DSC   = "#E67E22"   # DSConv branch
C_DEC   = "#27AE60"   # decoder
C_SEG   = "#E74C3C"   # seg head / logits
C_T1    = "#2980B9"   # teacher 1
C_T2    = "#8E44AD"   # teacher 2
C_ENS   = "#16A085"   # ensemble / DTKD
C_CONF  = "#D4AC0D"   # confidence-aware KD
C_LOSS  = "#F1C40F"   # loss
C_INPUT = "#95A5A6"   # input / GT
C_BG_A  = "#D6EAF8"   # bg region (a)
C_BG_B  = "#E8DAEF"   # bg region (b)


def draw_box(ax, x, y, w, h, text, color, fontsize=8, alpha=0.30, lw=1.2,
             textcolor="black", fontweight="normal", fontstyle="normal",
             zorder=2):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.06",
                          facecolor=color, edgecolor="black",
                          alpha=alpha, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight=fontweight, fontstyle=fontstyle,
            zorder=zorder + 1, family="sans-serif")
    return box


def draw_arrow(ax, x1, y1, x2, y2, color="black", lw=1.0, style="-",
               alpha=0.7, connectionstyle="arc3,rad=0", zorder=1):
    ls = "-" if style == "-" else "--"
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle="->,head_width=3,head_length=3",
                           connectionstyle=connectionstyle,
                           color=color, linewidth=lw, linestyle=ls,
                           alpha=alpha, zorder=zorder)
    ax.add_patch(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/fig_architecture.pdf")
    args = parser.parse_args()

    out_path = CODES_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Portrait canvas
    fig, ax = plt.subplots(1, 1, figsize=(10, 16))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.0, 17.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # ==================================================================
    # Background region (a): Network Architecture  (upper half)
    # ==================================================================
    bg_a = FancyBboxPatch((0.0, 7.5), 10.0, 8.8,
                           boxstyle="round,pad=0.25",
                           facecolor=C_BG_A, edgecolor=C_ENC, alpha=0.20,
                           linewidth=1.5, linestyle="--", zorder=0)
    ax.add_patch(bg_a)
    ax.text(0.6, 16.0, "(a) TopoDistill Architecture", fontsize=12,
            fontweight="bold", color=C_ENC, family="sans-serif", alpha=0.85)

    # Background region (b): Training / DTKD  (lower half)
    bg_b = FancyBboxPatch((0.0, -0.6), 10.0, 7.8,
                           boxstyle="round,pad=0.25",
                           facecolor=C_BG_B, edgecolor=C_T2, alpha=0.18,
                           linewidth=1.5, linestyle="--", zorder=0)
    ax.add_patch(bg_b)
    ax.text(0.6, 6.9, "(b) Class-Conditional Dual-Teacher KD (DTKD)",
            fontsize=12, fontweight="bold", color=C_T2,
            family="sans-serif", alpha=0.85)

    # ==================================================================
    # INPUT IMAGE (top)
    # ==================================================================
    inp_x, inp_y = 2.5, 15.2
    draw_box(ax, inp_x, inp_y, 1.6, 1.2, "Input\nImage\n$H{\\times}W$",
             C_INPUT, fontsize=9, alpha=0.35)

    # ==================================================================
    # ENCODER (left column)
    # ==================================================================
    enc_x = 2.5
    ax.text(enc_x, 14.2, "SegFormer-B2 Encoder", fontsize=10,
            fontweight="bold", ha="center", color=C_ENC, family="sans-serif")

    enc_stages = [
        ("Stage 1\n$H/4$, 64ch",  13.4, 0.25),
        ("Stage 2\n$H/8$, 128ch", 12.4, 0.35),
        ("Stage 3\n$H/16$, 320ch", 11.4, 0.45),
        ("Stage 4\n$H/32$, 512ch", 10.4, 0.55),
    ]
    for text, y, a in enc_stages:
        draw_box(ax, enc_x, y, 2.2, 0.7, text, C_ENC, fontsize=7, alpha=a)

    # Input → each encoder stage
    draw_arrow(ax, inp_x, inp_y - 0.6, enc_x, enc_stages[0][1] + 0.35,
               color=C_INPUT, lw=0.9)
    # Vertical chain between stages
    for i in range(len(enc_stages) - 1):
        draw_arrow(ax, enc_x, enc_stages[i][1] - 0.35,
                   enc_x, enc_stages[i + 1][1] + 0.35,
                   color=C_ENC, lw=0.8)

    # ==================================================================
    # DSConv BRANCH (right of encoder, parallel)
    # ==================================================================
    dsc_x = 6.5
    dsc_y = 12.9
    ax.text(dsc_x, 14.2, "CrackSnake Branch", fontsize=10,
            fontweight="bold", ha="center", color=C_DSC, family="sans-serif")

    draw_box(ax, dsc_x, dsc_y, 2.6, 1.8, "", C_DSC, fontsize=8, alpha=0.18)
    # Sub-boxes inside DSConv
    draw_box(ax, dsc_x - 0.6, dsc_y + 0.4, 1.2, 0.55,
             "DSConv-$x$\n$K{=}9$", C_DSC, fontsize=6.5, alpha=0.40)
    draw_box(ax, dsc_x + 0.6, dsc_y + 0.4, 1.2, 0.55,
             "DSConv-$y$\n$K{=}9$", C_DSC, fontsize=6.5, alpha=0.40)
    draw_box(ax, dsc_x, dsc_y - 0.45, 1.6, 0.45,
             "Proj → $\\mathbf{E}_{\\mathrm{crack}}$",
             C_DSC, fontsize=7, alpha=0.45)

    # Encoder Stage 1,2 → DSConv branch
    draw_arrow(ax, enc_x + 1.1, enc_stages[0][1],
               dsc_x - 1.3, dsc_y + 0.4,
               color=C_DSC, lw=0.9)
    draw_arrow(ax, enc_x + 1.1, enc_stages[1][1],
               dsc_x - 1.3, dsc_y + 0.4,
               color=C_DSC, lw=0.9)
    ax.text(4.5, 13.5, "Stage 1, 2", fontsize=6.5, ha="center",
            color=C_DSC, alpha=0.7, family="sans-serif")

    # ==================================================================
    # DECODER
    # ==================================================================
    dec_x, dec_y = 2.5, 9.2
    draw_box(ax, dec_x, dec_y, 2.4, 0.8, "MLP Decoder\nLinear Fuse",
             C_DEC, fontsize=8, alpha=0.35)
    # Encoder stages → Decoder (all four stages feed into MLP decoder)
    draw_arrow(ax, enc_x, enc_stages[-1][1] - 0.35,
               dec_x, dec_y + 0.4, color=C_ENC, lw=0.9)
    # Small annotation
    ax.text(enc_x + 1.15, (enc_stages[-1][1] + dec_y) / 2 + 0.15,
            "All 4\nstages", fontsize=5.5, color=C_ENC, alpha=0.55,
            family="sans-serif", ha="center", va="center")

    # ==================================================================
    # SEGMENTATION HEAD + CRACK ENHANCEMENT
    # ==================================================================
    seg_x, seg_y = 5.5, 9.2
    draw_box(ax, seg_x, seg_y, 2.0, 0.8, "Seg Head\n3-class logits",
             C_SEG, fontsize=8, alpha=0.30)
    draw_arrow(ax, dec_x + 1.2, dec_y, seg_x - 1.0, seg_y,
               color=C_DEC, lw=0.9)

    # DSConv → additive enhancement on crack channel
    draw_arrow(ax, dsc_x, dsc_y - 0.9, seg_x + 0.6, seg_y + 0.4,
               color=C_DSC, lw=1.0, style="--",
               connectionstyle="arc3,rad=0.2")
    ax.text(6.8, 10.3, "$\\hat{y}_{\\mathrm{cr}} = z_{\\mathrm{cr}} "
            "+ \\mathbf{E}_{\\mathrm{crack}}$",
            fontsize=7, ha="center", color=C_DSC, family="sans-serif",
            alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=C_DSC, alpha=0.5, linewidth=0.8))

    # Student logits output
    slog_x, slog_y = 8.5, 9.2
    draw_box(ax, slog_x, slog_y, 1.6, 0.7, "Student\nLogits $\\mathbf{z}_s$",
             C_SEG, fontsize=7.5, alpha=0.22)
    draw_arrow(ax, seg_x + 1.0, seg_y, slog_x - 0.8, slog_y,
               color=C_SEG, lw=0.9)

    # ==================================================================
    # SEPARATOR — dashed line between (a) and (b)
    # ==================================================================

    # ==================================================================
    # TEACHER 1  (left)
    # ==================================================================
    t1_x, t1_y = 1.8, 5.8
    draw_box(ax, t1_x, t1_y, 2.4, 1.2, "Teacher 1\nTopoDistill\n(frozen)",
             C_T1, fontsize=8, alpha=0.30, fontweight="bold")
    ax.text(t1_x, 5.0, "Strong on spalling", fontsize=6.5,
            ha="center", color=C_T1, fontstyle="italic",
            family="sans-serif", alpha=0.7)

    # TEACHER 2  (right)
    t2_x, t2_y = 6.0, 5.8
    draw_box(ax, t2_x, t2_y, 2.4, 1.2, "Teacher 2\nSAM-LoRA\n(frozen)",
             C_T2, fontsize=8, alpha=0.30, fontweight="bold")
    ax.text(t2_x, 5.0, "Strong on crack connectivity", fontsize=6.5,
            ha="center", color=C_T2, fontstyle="italic",
            family="sans-serif", alpha=0.7)

    # ==================================================================
    # CLASS-CONDITIONAL ENSEMBLE
    # ==================================================================
    ens_x, ens_y = 3.9, 3.8
    draw_box(ax, ens_x, ens_y, 3.8, 1.0, "", C_ENS, fontsize=8, alpha=0.20)
    ax.text(ens_x, 4.5, "Class-Conditional Ensemble", fontsize=9,
            fontweight="bold", ha="center", color=C_ENS, family="sans-serif")

    # Weight boxes inside
    w_labels = [("bg\n$w_2{=}0.5$", ens_x - 1.2),
                ("cr\n$w_2{=}0.6$", ens_x),
                ("sp\n$w_2{=}0.3$", ens_x + 1.2)]
    for label, wx in w_labels:
        draw_box(ax, wx, ens_y, 0.95, 0.6, label, C_ENS,
                 fontsize=6.5, alpha=0.45)

    # T1, T2 → Ensemble
    draw_arrow(ax, t1_x, t1_y - 0.6, ens_x - 1.0, ens_y + 0.5,
               color=C_T1, lw=0.9)
    draw_arrow(ax, t2_x, t2_y - 0.6, ens_x + 1.0, ens_y + 0.5,
               color=C_T2, lw=0.9)

    # Ensemble output label
    ax.text(ens_x, 3.1,
            "$\\tilde{p}_c = w_{1,c}\\,p_{1,c} + w_{2,c}\\,p_{2,c}$",
            fontsize=8, ha="center", color=C_ENS, family="sans-serif",
            alpha=0.8)

    # ==================================================================
    # TEACHER DISAGREEMENT → AGREEMENT MAP
    # ==================================================================
    dis_x, dis_y = 8.2, 5.1
    draw_box(ax, dis_x, dis_y, 2.2, 1.6, "", C_CONF, fontsize=7, alpha=0.18)
    ax.text(dis_x, 5.7, "Teacher\nDisagreement", fontsize=7.5,
            fontweight="bold", ha="center", color=C_CONF, family="sans-serif")
    ax.text(dis_x, 4.8, "$d(i) = \\mathrm{KL}(p_1^{(i)} \\| p_2^{(i)})$",
            fontsize=6.5, ha="center", color=C_CONF, family="sans-serif")
    ax.text(dis_x, 4.4, "$a(i) = 1 - d(i)/\\max d$",
            fontsize=6.5, ha="center", color=C_CONF, family="sans-serif")

    # T1, T2 → Disagreement
    draw_arrow(ax, t1_x + 1.2, t1_y - 0.3, dis_x - 1.1, dis_y + 0.3,
               color=C_CONF, lw=0.8, style="--",
               connectionstyle="arc3,rad=-0.15")
    draw_arrow(ax, t2_x + 1.2, t2_y - 0.3, dis_x - 1.1, dis_y + 0.1,
               color=C_CONF, lw=0.8, style="--")

    # ==================================================================
    # CONFIDENCE-AWARE KD LOSS
    # ==================================================================
    kd_x, kd_y = 5.5, 1.8
    draw_box(ax, kd_x, kd_y, 5.5, 1.0, "", C_CONF, fontsize=8, alpha=0.22)
    ax.text(kd_x, 2.5, "Confidence-Aware KD Loss", fontsize=9.5,
            fontweight="bold", ha="center", color="#7D6608",
            family="sans-serif")
    ax.text(kd_x, kd_y,
            "$\\mathcal{L}_{\\mathrm{KD}}^{\\mathrm{conf}} = "
            "\\tau^2 \\cdot \\frac{1}{N}\\sum_i a(i)\\,"
            "\\mathrm{KL}(\\tilde{\\mathbf{p}}^{(i)} \\| "
            "\\mathrm{softmax}(\\mathbf{z}_s^{(i)}/\\tau))$",
            fontsize=7, ha="center", color="black",
            family="sans-serif", alpha=0.8)

    # Ensemble → KD loss
    draw_arrow(ax, ens_x, ens_y - 0.5, kd_x - 1.5, kd_y + 0.5,
               color=C_ENS, lw=0.9)
    # Agreement map → KD loss
    draw_arrow(ax, dis_x, dis_y - 0.8, kd_x + 1.5, kd_y + 0.5,
               color=C_CONF, lw=0.9)
    ax.text(7.5, 3.1, "$a(i)$", fontsize=7.5, ha="center",
            color=C_CONF, fontweight="bold", family="sans-serif")
    # Student logits → KD loss (route along right side)
    draw_arrow(ax, slog_x + 0.3, slog_y - 0.35, kd_x + 2.5, kd_y + 0.5,
               color=C_SEG, lw=0.9, connectionstyle="arc3,rad=0.25")

    # ==================================================================
    # TOTAL LOSS
    # ==================================================================
    tot_x, tot_y = 5.5, 0.2
    draw_box(ax, tot_x, tot_y, 7.0, 0.7, "", C_LOSS, fontsize=8, alpha=0.25)
    ax.text(tot_x, tot_y,
            "$\\mathcal{L} = (1{-}\\alpha)"
            "[\\lambda_1\\mathcal{L}_{\\mathrm{CE}}"
            " + \\lambda_2\\mathcal{L}_{\\mathrm{Dice}}]"
            " + \\alpha\\,\\mathcal{L}_{\\mathrm{KD}}^{\\mathrm{conf}}$"
            "      ($\\alpha{=}0.5,\\;\\tau{=}4$)",
            fontsize=7.5, ha="center", color="black",
            family="sans-serif", fontweight="bold")
    ax.text(tot_x, tot_y + 0.6, "Total Loss $\\mathcal{L}$", fontsize=9.5,
            fontweight="bold", ha="center", color="#7D6608",
            family="sans-serif")

    # GT → Total loss
    gt_x, gt_y = 0.8, 2.5
    draw_box(ax, gt_x, gt_y, 1.4, 0.8, "Ground\nTruth",
             C_INPUT, fontsize=8, alpha=0.35)
    draw_arrow(ax, gt_x + 0.7, gt_y - 0.2, tot_x - 3.5, tot_y + 0.35,
               color=C_INPUT, lw=0.9)
    # Student logits → Total loss (CE+Dice part) — route along right edge
    draw_arrow(ax, slog_x + 0.4, slog_y - 0.35, tot_x + 3.0, tot_y + 0.35,
               color=C_SEG, lw=0.7, alpha=0.45,
               connectionstyle="arc3,rad=0.4")
    ax.text(9.7, 5.5, "$\\mathbf{z}_s$", fontsize=6.5, color=C_SEG,
            alpha=0.55, family="sans-serif")
    # KD loss → Total loss
    draw_arrow(ax, kd_x, kd_y - 0.5, tot_x, tot_y + 0.35,
               color=C_CONF, lw=0.9)

    # ==================================================================
    # LEGEND (bottom-right)
    # ==================================================================
    legend_items = [
        (C_ENC, "SegFormer Encoder"),
        (C_DSC, "DSConv Branch"),
        (C_DEC, "Decoder"),
        (C_T1,  "Teacher 1 (task-specific)"),
        (C_T2,  "Teacher 2 (foundation model)"),
        (C_ENS, "Ensemble / DTKD"),
        (C_CONF, "Confidence-aware KD"),
    ]
    handles = [mpatches.Patch(facecolor=c, edgecolor="black", alpha=0.4,
                               label=l) for c, l in legend_items]
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              framealpha=0.5, edgecolor="gray",
              bbox_to_anchor=(1.0, -0.05))

    # ==================================================================
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    png_path = out_path.with_suffix(".png")
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

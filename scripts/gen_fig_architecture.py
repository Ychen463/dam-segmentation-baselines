"""Generate Figure 1: HeteroDistill architecture diagram.

Part (a): SegFormer-B2 + DSConv branch
Part (b): DTKD with equal-weight logit averaging (no agreement-aware components)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colours ──────────────────────────────────────────────────────────────
C_ENC   = "#A8C8E8"  # SegFormer encoder (blue)
C_BRANCH= "#FDDCB5"  # DSConv branch (orange)
C_DEC   = "#C8E6C9"  # Decoder (green)
C_T1    = "#B3CCE6"  # Teacher 1 (light blue)
C_T2    = "#D5B8E8"  # Teacher 2 (purple)
C_ENS   = "#B2DFDB"  # Ensemble / DTKD (teal)
C_LOSS  = "#FFF9C4"  # Loss (yellow)
C_STUD  = "#F8BBD0"  # Student logits (pink)
C_GT    = "#E0E0E0"  # Ground truth (grey)
C_BG    = "#FAFAFA"  # Panel background

def rounded_box(ax, x, y, w, h, text, color, fontsize=9, bold=False,
                text2=None, fontsize2=7):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor="#666666", linewidth=1.0)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.08 if text2 else 0), text,
            ha="center", va="center", fontsize=fontsize, fontweight=weight)
    if text2:
        ax.text(x, y - 0.12, text2, ha="center", va="center",
                fontsize=fontsize2, fontstyle="italic", color="#555555")

def arrow(ax, x1, y1, x2, y2, color="#555555", style="-|>", lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def dashed_arrow(ax, x1, y1, x2, y2, color="#888888", lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle="dashed"))

# ── Figure setup ─────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10, 14),
                                  gridspec_kw={"height_ratios": [1.1, 0.9]})

for ax in (ax_a, ax_b):
    ax.set_xlim(-0.5, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")

# ======================================================================
# Part (a): SegFormer-DSConv Architecture
# ======================================================================
ax_a.set_ylim(-1.0, 9.5)

# Panel background
ax_a.add_patch(FancyBboxPatch((-0.3, -0.8), 10.6, 10.1,
               boxstyle="round,pad=0.15", facecolor=C_BG,
               edgecolor="#CCCCCC", linewidth=1.5))
ax_a.text(0.3, 9.0, "(a) SegFormer-DSConv Architecture",
          fontsize=13, fontweight="bold", color="#D4760A")

# Input
rounded_box(ax_a, 2.5, 8.2, 1.8, 0.55, "Input Image", C_GT,
            fontsize=9, text2="H x W")

# SegFormer Encoder
ax_a.text(2.5, 7.45, "SegFormer-B2 Encoder", ha="center",
          fontsize=10, fontweight="bold", color="#1565C0")

stages = [
    (6.8, "Stage 1", "H/4, 64ch"),
    (6.0, "Stage 2", "H/8, 128ch"),
    (5.2, "Stage 3", "H/16, 320ch"),
    (4.4, "Stage 4", "H/32, 512ch"),
]
for yy, name, info in stages:
    rounded_box(ax_a, 2.5, yy, 1.8, 0.5, name, C_ENC, fontsize=8, text2=info)

# Arrows between stages
arrow(ax_a, 2.5, 7.9, 2.5, 7.1)
for i in range(len(stages) - 1):
    arrow(ax_a, 2.5, stages[i][0] - 0.3, 2.5, stages[i+1][0] + 0.3)

# DSConv Branch
ax_a.text(7.0, 7.45, "CrackSnake Branch", ha="center",
          fontsize=10, fontweight="bold", color="#E65100")

rounded_box(ax_a, 6.3, 6.6, 1.3, 0.5, "DSConv-x", C_BRANCH, fontsize=8,
            text2="K = 9")
rounded_box(ax_a, 7.7, 6.6, 1.3, 0.5, "DSConv-y", C_BRANCH, fontsize=8,
            text2="K = 9")
rounded_box(ax_a, 7.0, 5.7, 1.8, 0.45, "Proj -> E_crack", C_BRANCH, fontsize=8)

# Stage 1,2 -> Branch
ax_a.annotate("Stage 1, 2", xy=(5.6, 6.6), xytext=(3.5, 6.8),
              fontsize=7, color="#E65100",
              arrowprops=dict(arrowstyle="-|>", color="#E65100", lw=1.0))

# Branch internal arrows
arrow(ax_a, 6.3, 6.3, 6.65, 5.95, color="#E65100")
arrow(ax_a, 7.7, 6.3, 7.35, 5.95, color="#E65100")

# Crack channel addition
rounded_box(ax_a, 7.0, 4.6, 2.2, 0.4,
            "y_cr = z_cr + E_crack", C_BRANCH, fontsize=8)
arrow(ax_a, 7.0, 5.45, 7.0, 4.85, color="#E65100")

# MLP Decoder
rounded_box(ax_a, 2.5, 3.2, 2.0, 0.55, "MLP Decoder", C_DEC,
            fontsize=9, text2="Linear Fuse")
arrow(ax_a, 2.5, 4.1, 2.5, 3.5)

# Seg Head
rounded_box(ax_a, 5.2, 3.2, 2.0, 0.55, "Seg Head", C_DEC,
            fontsize=9, text2="3-class logits")
arrow(ax_a, 3.5, 3.2, 4.15, 3.2)
arrow(ax_a, 7.0, 4.35, 7.0, 3.55)
# Connect seg head to crack addition
dashed_arrow(ax_a, 6.2, 3.2, 6.5, 4.4, color="#E65100")

# Student Logits
rounded_box(ax_a, 8.5, 3.2, 1.8, 0.55, "Student Logits z_s", C_STUD,
            fontsize=9)
arrow(ax_a, 6.2, 3.2, 7.55, 3.2)

# "4 stages" annotation
ax_a.text(1.3, 4.5, "4 stages", fontsize=7, color="#888888", rotation=90)

# ======================================================================
# Part (b): Dual-Teacher KD (DTKD)
# ======================================================================
ax_b.set_ylim(-0.5, 8.0)

# Panel background
ax_b.add_patch(FancyBboxPatch((-0.3, -0.3), 10.6, 8.1,
               boxstyle="round,pad=0.15", facecolor=C_BG,
               edgecolor="#CCCCCC", linewidth=1.5))
ax_b.text(0.3, 7.3, "(b) Dual-Teacher KD (DTKD)",
          fontsize=13, fontweight="bold", color="#D4760A")

# Teacher 1
rounded_box(ax_b, 2.0, 6.4, 2.2, 0.7, "Teacher 1", C_T1,
            fontsize=10, bold=True, text2="DSConv+SRL (frozen)")
ax_b.text(2.0, 5.85, "Strong on pixel accuracy", ha="center",
          fontsize=7, fontstyle="italic", color="#1565C0")

# Teacher 2
rounded_box(ax_b, 6.5, 6.4, 2.2, 0.7, "Teacher 2", C_T2,
            fontsize=10, bold=True, text2="SAM-LoRA (frozen)")
ax_b.text(6.5, 5.85, "Strong on spalling connectivity", ha="center",
          fontsize=7, fontstyle="italic", color="#7B1FA2")

# T1 logits
rounded_box(ax_b, 2.0, 5.1, 1.6, 0.45, "T1 Logits z_1", C_T1, fontsize=8)
arrow(ax_b, 2.0, 6.0, 2.0, 5.35)

# T2 logits
rounded_box(ax_b, 6.5, 5.1, 1.6, 0.45, "T2 Logits z_2", C_T2, fontsize=8)
arrow(ax_b, 6.5, 6.0, 6.5, 5.35)

# Equal-weight averaging
rounded_box(ax_b, 4.25, 4.0, 3.0, 0.6, "Equal-Weight Logit Averaging", C_ENS,
            fontsize=9, bold=True)
ax_b.text(4.25, 3.55, r"$\bar{z} = 0.5 \cdot z_1 + 0.5 \cdot z_2$",
          ha="center", va="center", fontsize=9, color="#00695C")

# Arrows from logits to averaging
arrow(ax_b, 2.0, 4.85, 3.2, 4.35)
arrow(ax_b, 6.5, 4.85, 5.3, 4.35)

# Ensemble soft target
rounded_box(ax_b, 4.25, 2.8, 2.6, 0.5, "Ensemble Soft Target", C_ENS,
            fontsize=9)
ax_b.text(4.25, 2.4, r"$\bar{p} = \mathrm{softmax}(\bar{z}/\tau)$",
          ha="center", va="center", fontsize=9, color="#00695C")
arrow(ax_b, 4.25, 3.25, 4.25, 3.1)

# Student logits (from part a)
rounded_box(ax_b, 8.5, 4.0, 1.8, 0.55, "Student Logits z_s", C_STUD,
            fontsize=9)
ax_b.text(9.6, 4.0, "(from a)", fontsize=7, color="#888888")

# KL Distillation Loss
rounded_box(ax_b, 6.5, 1.5, 2.8, 0.6, "KL Distillation Loss", C_LOSS,
            fontsize=9, bold=True)
ax_b.text(6.5, 1.0, r"$\mathcal{L}_{\mathrm{KD}} = \tau^2 \cdot \mathrm{KL}(\bar{p} \| \mathrm{softmax}(z_s/\tau))$",
          ha="center", va="center", fontsize=8, color="#555555")

# Arrows to KL loss
arrow(ax_b, 4.25, 2.15, 5.5, 1.7)
arrow(ax_b, 8.5, 3.7, 7.5, 1.85)

# Ground Truth
rounded_box(ax_b, 1.5, 1.5, 1.6, 0.55, "Ground Truth", C_GT,
            fontsize=9)

# CE + Dice supervised loss
rounded_box(ax_b, 3.5, 1.5, 2.2, 0.6, "CE + Dice Loss", C_LOSS,
            fontsize=9, bold=True)
ax_b.text(3.5, 1.0, r"$\mathcal{L}_{\mathrm{sup}} = \lambda_1 \mathcal{L}_{\mathrm{CE}} + \lambda_2 \mathcal{L}_{\mathrm{Dice}}$",
          ha="center", va="center", fontsize=8, color="#555555")
arrow(ax_b, 2.3, 1.5, 2.35, 1.5)
dashed_arrow(ax_b, 8.5, 3.4, 4.5, 1.85, color="#888888")

# Total loss
rounded_box(ax_b, 5.0, 0.1, 5.0, 0.55, "Total Loss", C_LOSS,
            fontsize=9, bold=True)
ax_b.text(5.0, -0.25,
          r"$\mathcal{L} = (1-\alpha)[\lambda_1\mathcal{L}_{\mathrm{CE}} + \lambda_2\mathcal{L}_{\mathrm{Dice}}] + \alpha\,\mathcal{L}_{\mathrm{KD}}$"
          r"$\qquad (\alpha{=}0.5,\;\tau{=}4)$",
          ha="center", va="center", fontsize=9, color="#333333")
arrow(ax_b, 3.5, 0.85, 4.0, 0.45)
arrow(ax_b, 6.5, 0.85, 5.5, 0.45)

# Legend
legend_items = [
    (C_ENC, "SegFormer Encoder"),
    (C_BRANCH, "DSConv Branch"),
    (C_DEC, "Decoder"),
    (C_T1, "Teacher 1 (task-specific)"),
    (C_T2, "Teacher 2 (foundation model)"),
    (C_ENS, "Ensemble / DTKD"),
    (C_LOSS, "Loss"),
]
for i, (c, label) in enumerate(legend_items):
    yy = 7.0 - i * 0.35
    ax_b.add_patch(FancyBboxPatch((8.8, yy - 0.12), 0.3, 0.24,
                   boxstyle="round,pad=0.02", facecolor=c,
                   edgecolor="#666666", linewidth=0.5))
    ax_b.text(9.2, yy, label, fontsize=7, va="center")

plt.tight_layout(pad=1.0)

# Save
out_dir = "/Users/lynnchen/Documents/Research/Dynamic Difficulty-Aware Curriculum Learning with Boundary Refinement for Fine-Grained Crack and Spalling Segmentation in Concrete Dams/Codes/workspace/final/figures"
fig.savefig(f"{out_dir}/fig_architecture.png", dpi=200, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{out_dir}/fig_architecture.pdf", bbox_inches="tight",
            facecolor="white")
print("Saved fig_architecture.png and .pdf")

#!/usr/bin/env python
"""Fig 4: Curriculum learning dynamics — tier sampling distribution over epochs.

Shows how the proposed dynamic curriculum progressively introduces harder
samples, contrasted with the static curriculum baseline.

Usage:
    python scripts/fig4_curriculum_sampling.py [--output figures/fig4_curriculum.pdf]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODES_DIR = Path(__file__).resolve().parent.parent

TIER_COLORS = {"Easy": "#2ecc71", "Medium": "#f39c12", "Hard": "#e74c3c"}


def load_sampling(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df_val = df[df["split"] == "val"].sort_values("epoch")
    # Compute fractions
    total = df_val["sampled_t0"] + df_val["sampled_t1"] + df_val["sampled_t2"]
    total = total.replace(0, 1)  # avoid division by zero
    df_val = df_val.copy()
    df_val["frac_easy"] = df_val["sampled_t0"] / total
    df_val["frac_medium"] = df_val["sampled_t1"] / total
    df_val["frac_hard"] = df_val["sampled_t2"] / total
    return df_val


def plot_stacked(ax, df, title):
    epochs = df["epoch"].values
    easy = df["frac_easy"].values
    medium = df["frac_medium"].values
    hard = df["frac_hard"].values

    ax.stackplot(epochs, easy, medium, hard,
                 labels=["Easy", "Medium", "Hard"],
                 colors=[TIER_COLORS["Easy"], TIER_COLORS["Medium"],
                         TIER_COLORS["Hard"]],
                 alpha=0.85)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Sampling Fraction", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.grid(True, alpha=0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/fig4_curriculum.pdf")
    args = parser.parse_args()

    out_path = CODES_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Dynamic curriculum (proposed - full method G2)
    dynamic_csv = CODES_DIR / "full_method/runs/dscformer_full_G2/metrics.csv"
    # Static curriculum (baseline)
    static_csv = CODES_DIR / "baseline_segformer/runs/segformer_b2_staticcurr_512/metrics.csv"

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Check if static curriculum has sampling columns
    df_static = pd.read_csv(static_csv)
    has_sampling = "sampled_t0" in df_static.columns and df_static["sampled_t0"].sum() > 0

    if has_sampling:
        df_s = load_sampling(static_csv)
        plot_stacked(axes[0], df_s, "(a) Static Curriculum")
    else:
        # Static: show uniform distribution
        df_s = pd.read_csv(static_csv)
        df_s = df_s[df_s["split"] == "val"].sort_values("epoch")
        epochs = df_s["epoch"].values
        n = len(epochs)
        # Static curriculum uses fixed schedule: epochs 1-33 easy, 34-66 easy+med, 67-100 all
        easy = np.ones(n)
        medium = np.zeros(n)
        hard = np.zeros(n)
        for i, e in enumerate(epochs):
            pct = e / epochs[-1]
            if pct < 0.33:
                easy[i], medium[i], hard[i] = 1.0, 0.0, 0.0
            elif pct < 0.66:
                easy[i], medium[i], hard[i] = 0.5, 0.5, 0.0
            else:
                easy[i], medium[i], hard[i] = 0.33, 0.33, 0.34
        axes[0].stackplot(epochs, easy, medium, hard,
                          labels=["Easy", "Medium", "Hard"],
                          colors=[TIER_COLORS["Easy"], TIER_COLORS["Medium"],
                                  TIER_COLORS["Hard"]], alpha=0.85)
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Sampling Fraction", fontsize=11)
        axes[0].set_title("(a) Static Curriculum", fontsize=12)
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.2)

    df_d = load_sampling(dynamic_csv)
    plot_stacked(axes[1], df_d, "(b) Dynamic Difficulty-Aware Curriculum (Ours)")

    # Shared legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    png_path = out_path.with_suffix(".png")
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

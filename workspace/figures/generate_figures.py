"""Generate publication-quality figures for the paper."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import os

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Color palette
COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

# =============================================================================
# Figure 1: Main comparison bar chart (3:2 aspect)
# =============================================================================
def fig_main_comparison_bar():
    methods = ['U-Net\n(R34)', 'DeepLabV3+\n(R50)', 'Mask2Former\n(Swin-S)', 'SegFormer\n-B2', 'Ours\n(DSCFormer)']
    metrics = ['mIoU$_{fg}$', 'BF1$_{fg}$', 'clDice$_{fg}$']
    data = np.array([
        [59.6, 64.6, 73.7],
        [63.4, 59.2, 75.1],
        [67.9, 72.5, 86.6],
        [70.4, 66.8, 81.4],
        [71.4, 72.1, 86.2],
    ])

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(methods))
    width = 0.25
    metric_colors = ['#4C72B0', '#55A868', '#C44E52']

    for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
        bars = ax.bar(x + (i - 1) * width, data[:, i], width, label=metric, color=color, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(50, 95)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_main_comparison_bar.png'), dpi=200, bbox_inches='tight')
    plt.close()

# =============================================================================
# Figure 2: Ablation component contribution (4:3 aspect)
# =============================================================================
def fig_ablation_component_contribution():
    configs = ['+DSConv', '+DSConv+SRL\n(Ours)']
    metrics = ['$\\Delta$mIoU$_{fg}$', '$\\Delta$BF1$_{fg}$', '$\\Delta$clDice$_{fg}$']
    data = np.array([
        [0.0, 2.6, 4.1],
        [1.0, 5.3, 4.8],
    ])

    fig, ax = plt.subplots(figsize=(5.33, 4))
    x = np.arange(len(configs))
    width = 0.22
    metric_colors = ['#4C72B0', '#55A868', '#C44E52']

    for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
        bars = ax.bar(x + (i - 1) * width, data[:, i], width, label=metric, color=color, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, data[:, i]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'+{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Improvement over SegFormer-B2 (%)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylim(0, 7)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_ablation_component_contribution.png'), dpi=200, bbox_inches='tight')
    plt.close()

# =============================================================================
# Figure 3: Per-tier radar chart (1:1 aspect)
# =============================================================================
def fig_per_tier_radar():
    metrics = ['mIoU', 'IoU$_{cr}$', 'IoU$_{sp}$', 'BF1', 'clDice']
    tiers = ['Easy', 'Medium', 'Hard']

    ours = {
        'Easy':   [76.0, 65.2, 86.9, 68.3, 87.0],
        'Medium': [72.7, 61.0, 84.3, 77.9, 89.6],
        'Hard':   [66.1, 53.2, 79.0, 75.5, 84.4],
    }
    segformer = {
        'Easy':   [74.8, 63.4, 86.1, 65.1, 84.9],
        'Medium': [71.1, 60.0, 82.2, 66.6, 77.6],
        'Hard':   [64.4, 53.0, 75.8, 69.4, 79.4],
    }

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), subplot_kw=dict(polar=True))

    for idx, tier in enumerate(tiers):
        ax = axes[idx]
        vals_ours = ours[tier] + [ours[tier][0]]
        vals_seg = segformer[tier] + [segformer[tier][0]]

        ax.plot(angles, vals_ours, 'o-', color='#C44E52', linewidth=1.5, markersize=4, label='Ours')
        ax.fill(angles, vals_ours, alpha=0.1, color='#C44E52')
        ax.plot(angles, vals_seg, 's--', color='#4C72B0', linewidth=1.5, markersize=4, label='SegFormer-B2')
        ax.fill(angles, vals_seg, alpha=0.1, color='#4C72B0')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_ylim(50, 95)
        ax.set_yticks([60, 70, 80, 90])
        ax.set_yticklabels(['60', '70', '80', '90'], fontsize=7)
        ax.set_title(f'{tier} Tier', fontsize=11, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc='lower left', bbox_to_anchor=(-0.1, -0.15), fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_per_tier_radar.png'), dpi=200, bbox_inches='tight')
    plt.close()

# =============================================================================
# Generate all and write captions
# =============================================================================
if __name__ == '__main__':
    fig_main_comparison_bar()
    fig_ablation_component_contribution()
    fig_per_tier_radar()

    captions = {
        "fig_main_comparison_bar": "Comparison of segmentation methods on foreground-averaged metrics (mIoU_fg, BF1_fg, clDice_fg). Our DSCFormer achieves the highest mIoU while maintaining competitive boundary and connectivity scores.",
        "fig_ablation_component_contribution": "Incremental contribution of each proposed component over the SegFormer-B2 baseline. DSConv improves boundary and connectivity metrics, while adding SRL further boosts all three metrics.",
        "fig_per_tier_radar": "Per-difficulty-tier radar comparison between our method and SegFormer-B2 across five metrics. Our approach shows consistent gains, particularly on boundary (BF1) and connectivity (clDice) metrics for medium and hard samples."
    }

    with open(os.path.join(OUT_DIR, 'captions.json'), 'w') as f:
        json.dump(captions, f, indent=2)

    print("All figures generated successfully.")

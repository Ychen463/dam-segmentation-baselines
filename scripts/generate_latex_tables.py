#!/usr/bin/env python
"""Generate LaTeX tables for the paper from evaluation result JSONs.

Produces:
  - Table 1: Main comparison (all models, overall metrics)
  - Table 2: Per-tier performance breakdown (proposed method vs baselines)
  - Table 3: Ablation study

Usage:
    python scripts/generate_latex_tables.py [--output-dir figures/]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CODES_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = CODES_DIR / "results"

# ============================================================
# Model definitions for each table
# ============================================================

# (display_name, json_filename, is_ours)
MAIN_MODELS = [
    ("U-Net (ResNet-34)", "unet_r34_320_test.json", False),
    ("DeepLabV3+ (R34, 320)", "deeplabv3p_r34_320_test.json", False),
    ("DeepLabV3+ (R50, 512)", "deeplabv3p_r50_512_test.json", False),
    ("Mask2Former (Swin-S)", "mask2former_swin_small_512_test.json", False),
    ("SegFormer-B2", "segformer_b2_plain_512_test.json", False),
    ("Ours", "dscformer_srl_G1_test.json", True),
    ("Ours + TTA", "dscformer_srl_G1_test_tta.json", True),
]

# For tier breakdown
TIER_MODELS = [
    ("U-Net (R34)", "unet_r34_320_test.json", False),
    ("DeepLabV3+ (R50)", "deeplabv3p_r50_512_test.json", False),
    ("Mask2Former (Swin-S)", "mask2former_swin_small_512_test.json", False),
    ("SegFormer-B2", "segformer_b2_plain_512_test.json", False),
    ("Ours", "dscformer_srl_G1_test.json", True),
]

# Ablation: proposed method with components removed
# Check which result files actually exist
ABLATION_CANDIDATES = [
    ("SegFormer-B2 (baseline)", "segformer_b2_plain_512_test.json"),
    ("+ DSConv", "dscformer_plain_G0_test.json"),
    ("+ DSConv + SRL (Ours)", "dscformer_srl_G1_test.json"),
    ("+ DSConv + Aux Loss", "dscformer_aux_H0_test.json"),
    ("+ Curriculum Learning", "dscformer_full_G2_test.json"),
]


def load_results(filename: str) -> Optional[Dict]:
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val, bold=False, decimals=1):
    """Format a metric value as percentage string."""
    if val is None:
        return "---"
    s = f"{val * 100:.{decimals}f}"
    if bold:
        return f"\\textbf{{{s}}}"
    return s


def find_best(models_data: List[Tuple[str, Dict, bool]], metric: str,
              tier: str = "overall") -> int:
    """Find index of best model for a metric."""
    best_idx, best_val = -1, -1
    for i, (_, data, _) in enumerate(models_data):
        if data is None:
            continue
        val = data.get(tier, {}).get(metric)
        if val is not None and val > best_val:
            best_val = val
            best_idx = i
    return best_idx


# ============================================================
# Table 1: Main comparison
# ============================================================
def generate_table1(output_dir: Path):
    metrics = [
        ("mIoU$_{fg}$", "mIoU_fg"),
        ("IoU$_{cr}$", "IoU_crack"),
        ("IoU$_{sp}$", "IoU_spalling"),
        ("BF1$_{fg}$", "BF1_fg_mean"),
        ("clDice$_{fg}$", "clDice_fg_mean"),
        ("ConnR$_{fg}$", "ConnR_fg_mean"),
    ]

    models_data = []
    for name, fname, is_ours in MAIN_MODELS:
        data = load_results(fname)
        models_data.append((name, data, is_ours))

    # Find best per metric
    best_indices = {}
    for _, key in metrics:
        best_indices[key] = find_best(models_data, key)

    lines = []
    lines.append("% Table 1: Main quantitative comparison on DamSegment test set")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Quantitative comparison on the DamSegment test set. "
                 "Best results in \\textbf{bold}. "
                 "Ours = SegFormer-B2 + DSConv + SRL.}")
    lines.append("\\label{tab:main_comparison}")
    cols = "l" + "c" * len(metrics)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")

    # Header
    header = "Method & " + " & ".join(h for h, _ in metrics) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Rows
    for i, (name, data, is_ours) in enumerate(models_data):
        if i == len(MAIN_MODELS) - 2:  # before "Ours" rows
            lines.append("\\midrule")
        vals = []
        for _, key in metrics:
            if data is None:
                vals.append("---")
            else:
                v = data.get("overall", {}).get(key)
                is_best = (best_indices[key] == i)
                vals.append(fmt(v, bold=is_best))
        row = f"{name} & " + " & ".join(vals) + " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    tex = "\n".join(lines)
    out = output_dir / "table1_main_comparison.tex"
    out.write_text(tex)
    print(f"Saved: {out}")
    print(tex)
    print()


# ============================================================
# Table 2: Per-tier breakdown
# ============================================================
def generate_table2(output_dir: Path):
    tiers = ["Easy", "Medium", "Hard"]
    metrics = [
        ("mIoU$_{fg}$", "mIoU_fg"),
        ("IoU$_{cr}$", "IoU_crack"),
        ("IoU$_{sp}$", "IoU_spalling"),
        ("BF1$_{fg}$", "BF1_fg_mean"),
        ("clDice$_{fg}$", "clDice_fg_mean"),
    ]

    models_data = []
    for name, fname, is_ours in TIER_MODELS:
        data = load_results(fname)
        models_data.append((name, data, is_ours))

    lines = []
    lines.append("% Table 2: Per-difficulty-tier performance breakdown")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Per-difficulty-tier performance on the DamSegment test set.}")
    lines.append("\\label{tab:tier_breakdown}")
    cols = "ll" + "c" * len(metrics)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")

    header = "Tier & Method & " + " & ".join(h for h, _ in metrics) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for tier in tiers:
        # Find best per metric in this tier
        best_idx = {}
        for _, key in metrics:
            best_idx[key] = find_best(models_data, key, tier)

        for i, (name, data, is_ours) in enumerate(models_data):
            tier_prefix = tier if i == 0 else ""
            vals = []
            for _, key in metrics:
                if data is None:
                    vals.append("---")
                else:
                    v = data.get(tier, {}).get(key)
                    is_best = (best_idx[key] == i)
                    vals.append(fmt(v, bold=is_best))
            row = f"{tier_prefix} & {name} & " + " & ".join(vals) + " \\\\"
            lines.append(row)
        if tier != "Hard":
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    tex = "\n".join(lines)
    out = output_dir / "table2_tier_breakdown.tex"
    out.write_text(tex)
    print(f"Saved: {out}")
    print(tex)
    print()


# ============================================================
# Table 3: Ablation study
# ============================================================
def generate_table3(output_dir: Path):
    metrics = [
        ("mIoU$_{fg}$", "mIoU_fg"),
        ("IoU$_{cr}$", "IoU_crack"),
        ("IoU$_{sp}$", "IoU_spalling"),
        ("BF1$_{fg}$", "BF1_fg_mean"),
        ("clDice$_{fg}$", "clDice_fg_mean"),
        ("ConnR$_{fg}$", "ConnR_fg_mean"),
    ]

    # Filter to existing results
    ablation_models = []
    for name, fname in ABLATION_CANDIDATES:
        data = load_results(fname)
        if data is not None:
            ablation_models.append((name, data))

    if not ablation_models:
        print("[table3] No ablation results found, skipping.")
        return

    lines = []
    lines.append("% Table 3: Ablation study")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study on the DamSegment test set. "
                 "Each row adds a component to the baseline.}")
    lines.append("\\label{tab:ablation}")
    cols = "l" + "c" * len(metrics)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")

    header = "Configuration & " + " & ".join(h for h, _ in metrics) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Find best
    best_idx = {}
    for _, key in metrics:
        best_i, best_v = -1, -1
        for i, (_, data) in enumerate(ablation_models):
            v = data.get("overall", {}).get(key)
            if v is not None and v > best_v:
                best_v = v
                best_i = i
        best_idx[key] = best_i

    for i, (name, data) in enumerate(ablation_models):
        vals = []
        for _, key in metrics:
            v = data.get("overall", {}).get(key)
            is_best = (best_idx[key] == i)
            vals.append(fmt(v, bold=is_best))
        row = f"{name} & " + " & ".join(vals) + " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    tex = "\n".join(lines)
    out = output_dir / "table3_ablation.tex"
    out.write_text(tex)
    print(f"Saved: {out}")
    print(tex)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    output_dir = CODES_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_table1(output_dir)
    generate_table2(output_dir)
    generate_table3(output_dir)


if __name__ == "__main__":
    main()

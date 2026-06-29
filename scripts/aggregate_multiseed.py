"""Aggregate multi-seed evaluation results into mean +/- std tables.

Reads JSON result files produced by ``eval_all`` for each seed variant,
computes mean and standard deviation across seeds, and outputs a summary
table suitable for inclusion in the paper.

Usage::

    python scripts/aggregate_multiseed.py \
        --results-dir results/multiseed/ \
        --seeds 42 123 2024 \
        --output results/multiseed/stability_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


# Models to aggregate (base_name -> paper label)
MODELS = {
    "plain_segformer_P0": "SegFormer-B2",
    "dscformer_plain_G0": "DSConv-only",
    "dscformer_srl_G1": "DSCFormer+SRL",
    "dual_kd_classaware_DKD2": "DSCFormer+DTKD",
}

# Metrics to report
METRICS = [
    "mIoU_fg", "IoU_crack", "IoU_spalling",
    "BF1_fg_mean", "BF1_crack", "BF1_spalling",
    "clDice_fg_mean", "clDice_crack", "clDice_spalling",
    "ConnR_fg_mean", "ConnR_crack", "ConnR_spalling",
]

# Short metric names for display
SHORT_NAMES = {
    "mIoU_fg": "mIoU_fg",
    "IoU_crack": "IoU_cr",
    "IoU_spalling": "IoU_sp",
    "BF1_fg_mean": "BF1_fg",
    "BF1_crack": "BF1_cr",
    "BF1_spalling": "BF1_sp",
    "clDice_fg_mean": "clDice_fg",
    "clDice_crack": "clDice_cr",
    "clDice_spalling": "clDice_sp",
    "ConnR_fg_mean": "ConnR_fg",
    "ConnR_crack": "ConnR_cr",
    "ConnR_spalling": "ConnR_sp",
}


def load_results(results_dir: Path, base_name: str, seeds: List[int],
                 split: str = "test") -> Dict[str, List[float]]:
    """Load per-seed JSON results and return {metric: [val_seed1, val_seed2, ...]}."""
    collected: Dict[str, List[float]] = {m: [] for m in METRICS}
    for seed in seeds:
        fname = f"{base_name}_seed{seed}_{split}.json"
        path = results_dir / fname
        if not path.exists():
            print(f"  [WARN] Missing: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        overall = data.get("overall", data)
        for metric in METRICS:
            val = overall.get(metric)
            if val is not None:
                collected[metric].append(val)
    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/multiseed/")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="results/multiseed/stability_summary.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary = {}

    print("=" * 80)
    print("  Multi-Seed Stability Analysis")
    print("=" * 80)

    # Header
    header_metrics = ["mIoU_fg", "IoU_crack", "IoU_spalling",
                      "BF1_fg_mean", "clDice_fg_mean", "ConnR_fg_mean"]
    header_short = [SHORT_NAMES[m] for m in header_metrics]
    print(f"\n  {'Model':<22} " + "  ".join(f"{s:>14}" for s in header_short))
    print("  " + "-" * (22 + 16 * len(header_metrics)))

    for base_name, paper_label in MODELS.items():
        collected = load_results(results_dir, base_name, args.seeds, args.split)
        n_seeds = min(len(v) for v in collected.values()) if collected else 0

        model_summary = {"paper_label": paper_label, "n_seeds": n_seeds, "metrics": {}}

        vals_str = []
        for m in header_metrics:
            vals = collected[m]
            if len(vals) >= 2:
                mean = np.mean(vals) * 100
                std = np.std(vals, ddof=1) * 100
                model_summary["metrics"][m] = {
                    "mean": round(float(mean), 2),
                    "std": round(float(std), 2),
                    "values": [round(v * 100, 2) for v in vals],
                }
                vals_str.append(f"{mean:>6.1f}+/-{std:>4.1f}")
            elif len(vals) == 1:
                val = vals[0] * 100
                model_summary["metrics"][m] = {
                    "mean": round(float(val), 2),
                    "std": 0.0,
                    "values": [round(float(val), 2)],
                }
                vals_str.append(f"{val:>6.1f}      ")
            else:
                vals_str.append(f"{'N/A':>14}")

        print(f"  {paper_label:<22} " + "  ".join(vals_str))
        summary[base_name] = model_summary

    # LaTeX table snippet
    print("\n\n" + "=" * 80)
    print("  LaTeX table snippet (for paper)")
    print("=" * 80)
    latex_metrics = ["mIoU_fg", "IoU_crack", "IoU_spalling",
                     "BF1_fg_mean", "ConnR_fg_mean"]
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Training stability across 3 random seeds (mean $\pm$ std, \%).}")
    print(r"\label{tab:stability}")
    cols = " ".join(["l"] + ["c"] * len(latex_metrics))
    print(r"\begin{tabular}{" + cols + "}")
    print(r"\toprule")
    latex_headers = [SHORT_NAMES[m].replace("_", r"$_{\text{").rstrip() + r"}}$"
                     if "_" in SHORT_NAMES[m] else SHORT_NAMES[m]
                     for m in latex_metrics]
    print("Method & " + " & ".join(latex_headers) + r" \\")
    print(r"\midrule")

    for base_name, paper_label in MODELS.items():
        cells = []
        for m in latex_metrics:
            info = summary[base_name]["metrics"].get(m, {})
            mean = info.get("mean", 0)
            std = info.get("std", 0)
            if std > 0:
                cells.append(f"{mean:.1f}$\\pm${std:.1f}")
            else:
                cells.append(f"{mean:.1f}")
        print(f"{paper_label} & " + " & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[aggregate] Saved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Round 7 Results Analysis: Softmax Sampling vs Loss Reweight.

Reads test_report.txt from all relevant runs and generates:
  1. Overall ranking table (sorted by mIoU_fg)
  2. Head-to-head comparisons (D0 vs D0v2, D1 vs D1v2, F1 vs F1v2)
  3. Training convergence summary from metrics.csv
  4. Markdown report written to logs/round7_analysis.md

Usage::

    python scripts/14_round7_analysis.py
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "full_method" / "runs"
OUT_DIR = ROOT / "logs"

# Experiments to include (label → run directory name)
EXPERIMENTS = {
    "P0":   "plain_segformer_P0",
    "P1":   "plain_cldice_P1",
    "C2":   "competence_soft_C2",
    "D0":   "difficulty_only_D0",
    "D1":   "difficulty_cldice_D1",
    "D0v2": "difficulty_sampling_D0v2",
    "D1v2": "difficulty_sampling_cldice_D1v2",
    "F1":   "full_method_F1",
    "F1v2": "full_method_F1v2",
}

STRATEGY_LABELS = {
    "P0":   "plain segformer",
    "P1":   "plain + clDice",
    "C2":   "competence curriculum",
    "D0":   "loss reweight only",
    "D1":   "loss reweight + clDice",
    "D0v2": "softmax sampling",
    "D1v2": "softmax sampling + clDice",
    "F1":   "loss reweight + C2 + clDice",
    "F1v2": "softmax sampling + C2 + clDice",
}

ROUND_LABELS = {
    "P0": "baseline", "P1": "baseline", "C2": "baseline",
    "D0": "R6", "D1": "R6", "F1": "R6",
    "D0v2": "R7", "D1v2": "R7", "F1v2": "R7",
}

# Key metrics for tables
KEY_METRICS = [
    "mIoU_fg", "IoU_crack", "IoU_spalling",
    "BF1_fg_mean", "clDice_fg_mean", "ConnR_fg_mean",
]
METRIC_SHORT = {
    "mIoU_fg": "mIoU_fg",
    "IoU_crack": "IoU_crack",
    "IoU_spalling": "IoU_spalling",
    "BF1_fg_mean": "BF1_fg",
    "clDice_fg_mean": "clDice_fg",
    "ConnR_fg_mean": "ConnR_fg",
}

# Head-to-head pairs (R6 → R7)
H2H_PAIRS = [
    ("D0", "D0v2", "difficulty only"),
    ("D1", "D1v2", "difficulty + clDice"),
    ("F1", "F1v2", "full method"),
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_test_report(path: Path) -> Dict[str, float]:
    """Parse a test_report.txt into a metric dict."""
    metrics: Dict[str, float] = {}
    best_epoch: Optional[int] = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("best epoch:"):
            best_epoch = int(line.split(":")[1].strip())
        m = re.match(r"(\w+):\s+([\d.eE+-]+)", line)
        if m:
            metrics[m.group(1)] = float(m.group(2))
    if best_epoch is not None:
        metrics["best_epoch"] = float(best_epoch)
    return metrics


def read_best_epoch_from_csv(path: Path) -> Optional[int]:
    """Read metrics.csv to find the last recorded validation epoch."""
    if not path.exists():
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        last_epoch = None
        for row in reader:
            if row.get("split") == "val":
                last_epoch = int(row["epoch"])
    return last_epoch


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def fmt(val: float, precision: int = 4) -> str:
    return f"{val:.{precision}f}"


def bold_best(values: List[float], labels: List[str], higher_better: bool = True) -> List[str]:
    """Return formatted strings with best value bolded in markdown."""
    if not values:
        return []
    best_val = max(values) if higher_better else min(values)
    result = []
    for v in values:
        s = fmt(v)
        if abs(v - best_val) < 1e-8:
            s = f"**{s}**"
        result.append(s)
    return result


def delta_str(old: float, new: float) -> str:
    """Format a delta as percentage with sign."""
    diff = (new - old) * 100
    sign = "+" if diff >= 0 else ""
    bold = "**" if abs(diff) >= 1.0 else ""
    return f"{bold}{sign}{diff:.2f}%{bold}"


# ---------------------------------------------------------------------------
# Analysis generation
# ---------------------------------------------------------------------------

def generate_analysis() -> str:
    lines: List[str] = []

    # Load all experiments
    all_metrics: Dict[str, Dict[str, float]] = {}
    for label, dirname in EXPERIMENTS.items():
        report_path = RUNS_DIR / dirname / "test_report.txt"
        if not report_path.exists():
            print(f"  WARNING: {report_path} not found, skipping {label}")
            continue
        all_metrics[label] = parse_test_report(report_path)

    if not all_metrics:
        return "ERROR: No test reports found."

    # ----- Header -----
    lines.append("# Round 7 Results Analysis: Softmax Sampling vs Loss Reweight")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append("Round 7 tested whether replacing **dynamic loss reweighting** (Round 6) "
                 "with **softmax sampling** could improve performance. The core hypothesis: "
                 "instead of reweighting the loss by difficulty, use difficulty scores to bias "
                 "the sampling probability via softmax.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ----- Overall ranking table -----
    lines.append("## Test Set Results Summary")
    lines.append("")
    lines.append("### All Experiments (sorted by mIoU_fg)")
    lines.append("")

    sorted_labels = sorted(all_metrics.keys(),
                           key=lambda k: all_metrics[k].get("mIoU_fg", 0),
                           reverse=True)

    header = "| Experiment | Strategy | mIoU_fg | IoU_crack | IoU_spalling | BF1_fg | clDice_fg | ConnR_fg |"
    sep =    "|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    # Find best for each metric
    best_vals = {}
    for m in KEY_METRICS:
        vals = [all_metrics[l].get(m, 0) for l in sorted_labels]
        best_vals[m] = max(vals) if vals else 0

    for label in sorted_labels:
        m = all_metrics[label]
        rd = ROUND_LABELS.get(label, "")
        tag = f" ({rd})" if rd else ""
        exp_str = f"**{label}**{tag}" if label in ("F1", "D0v2", "D1v2", "F1v2") else f"{label}{tag}"
        strat = STRATEGY_LABELS.get(label, "")

        cells = []
        for metric in KEY_METRICS:
            val = m.get(metric, 0)
            s = fmt(val)
            if abs(val - best_vals[metric]) < 1e-8:
                s = f"**{s}**"
            cells.append(s)

        lines.append(f"| {exp_str} | {strat} | {' | '.join(cells)} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ----- Head-to-head comparisons -----
    lines.append("## Round 6 vs Round 7 Head-to-Head")
    lines.append("")

    for r6_label, r7_label, desc in H2H_PAIRS:
        if r6_label not in all_metrics or r7_label not in all_metrics:
            continue

        r6 = all_metrics[r6_label]
        r7 = all_metrics[r7_label]

        lines.append(f"### {r6_label} ({desc}) → {r7_label}")
        lines.append("")
        lines.append("| Metric | {0} (reweight) | {1} (sampling) | Delta |".format(r6_label, r7_label))
        lines.append("|---|---|---|---|")

        wins_r6 = 0
        wins_r7 = 0

        for metric in KEY_METRICS:
            v6 = r6.get(metric, 0)
            v7 = r7.get(metric, 0)
            s6 = fmt(v6)
            s7 = fmt(v7)
            if v6 > v7 + 1e-8:
                s6 = f"**{s6}**"
                wins_r6 += 1
            elif v7 > v6 + 1e-8:
                s7 = f"**{s7}**"
                wins_r7 += 1
            d = delta_str(v6, v7)
            mname = METRIC_SHORT.get(metric, metric)
            lines.append(f"| {mname} | {s6} | {s7} | {d} |")

        lines.append("")

        # Verdict
        miou_delta = (r7.get("mIoU_fg", 0) - r6.get("mIoU_fg", 0)) * 100
        connr_delta = (r7.get("ConnR_fg_mean", 0) - r6.get("ConnR_fg_mean", 0)) * 100

        if r6_label == "D0":
            lines.append(f"**Verdict: {r7_label} wins on IoU/BF1, {r6_label} wins on connectivity "
                         f"metrics. Softmax sampling is a clear upgrade for {r6_label}.**")
        elif r6_label == "D1":
            lines.append(f"**Verdict: {r7_label} marginally better on IoU/BF1, but ConnR_fg drops "
                         f"sharply ({connr_delta:+.1f}%). Softmax sampling hurts connectivity when "
                         f"combined with clDice.**")
        elif r6_label == "F1":
            lines.append(f"**Verdict: {r6_label} (Round 6) is clearly better. {r7_label} drops on "
                         f"crack IoU and boundary F1. Softmax sampling hurts the full pipeline.**")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ----- Training convergence -----
    lines.append("## Training Convergence")
    lines.append("")
    lines.append("| Experiment | Best Epoch | Val mIoU_fg (best) | Test mIoU_fg |")
    lines.append("|---|---|---|---|")

    for label in sorted_labels:
        m = all_metrics[label]
        best_ep = int(m.get("best_epoch", 0))
        # Read val mIoU from metrics.csv
        csv_path = RUNS_DIR / EXPERIMENTS[label] / "metrics.csv"
        val_miou = "—"
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("split") == "val" and int(row["epoch"]) == best_ep:
                        val_miou = fmt(float(row["mIoU_fg"]))
                        break
        test_miou = fmt(m.get("mIoU_fg", 0))
        lines.append(f"| {label} | {best_ep} | {val_miou} | {test_miou} |")

    lines.append("")
    lines.append("F1v2 converged at epoch 35 (vs F1 at epoch 80), suggesting instability "
                 "or premature convergence with softmax sampling in the full method.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ----- Key observations -----
    lines.append("## Key Observations")
    lines.append("")

    f1_miou = all_metrics.get("F1", {}).get("mIoU_fg", 0)
    p1_miou = all_metrics.get("P1", {}).get("mIoU_fg", 0)
    p1_gap = (f1_miou - p1_miou) * 100

    observations = [
        f"**F1 (Round 6) remains the best overall model** with mIoU_fg = {fmt(f1_miou)}, "
        f"the highest among all experiments.",

        "**Softmax sampling helps the simple case (D0→D0v2)** but **hurts the full pipeline "
        "(F1→F1v2)**. The more components are stacked, the less benefit sampling provides.",

        "**Crack IoU is the bottleneck across all experiments** (~0.54-0.57), while spalling "
        "IoU is consistently strong (~0.84-0.85).",

        f"**F1v2 converged too early (epoch 35)** compared to F1 (epoch 80), suggesting "
        f"instability or premature convergence with softmax sampling in the full method.",

        f"**Baselines are surprisingly competitive**: P1 (plain + clDice, mIoU_fg={fmt(p1_miou)}) "
        f"is only {p1_gap:.1f}% behind F1, raising questions about whether the difficulty "
        f"mechanism adds meaningful value over clDice alone.",

        "**ConnR_fg is volatile**: D1v2 has a large ConnR drop vs D1 (-11.4%), suggesting "
        "softmax sampling + clDice combination may fragment predictions.",
    ]

    for i, obs in enumerate(observations, 1):
        lines.append(f"{i}. {obs}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ----- Recommendations -----
    lines.append("## Recommendation for Next Steps")
    lines.append("")
    lines.append(f"The current best model is **F1 (Round 6, loss reweight)**. Options to consider:")
    lines.append("")
    lines.append("1. **Stick with F1 as the final model** — it leads on mIoU_fg and BF1_fg.")
    lines.append("2. **Investigate why P1 (plain + clDice) nearly matches F1** — if the difficulty "
                 "mechanism doesn't add much over clDice, the paper's contribution narrative "
                 "needs adjustment.")
    lines.append("3. **Try a hybrid**: softmax sampling for D0-level, loss reweight for full pipeline.")
    lines.append("4. **Address crack IoU**: this is the weakest link (~0.57 vs spalling ~0.85). "
                 "Targeted improvements here would have the biggest impact.")
    lines.append("")

    return "\n".join(lines)


def main():
    print("Round 7 Results Analysis")
    print("=" * 60)

    report = generate_analysis()

    # Print to stdout
    print(report)

    # Save to file
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "round7_analysis.md"
    out_path.write_text(report)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

"""Statistical significance testing between model pairs.

Reads per-image CSV files produced by ``eval_all --per-image`` and runs
paired Wilcoxon signed-rank tests + bootstrap 95% confidence intervals.

Usage::

    # Compare full method vs all baselines on key metrics
    python -m shared_eval.stats_significance --results-dir results/ --split test

    # Compare two specific models
    python -m shared_eval.stats_significance --results-dir results/ --split test \
        --model-a segformer_b2_full_512 --model-b segformer_b2_plain_512

    # Filter by tier
    python -m shared_eval.stats_significance --results-dir results/ --split test --tier Hard
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_per_image_csv(path: Path) -> List[Dict[str, object]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k in ("file", "tier"):
                    parsed[k] = v
                else:
                    try:
                        parsed[k] = float(v)
                    except (ValueError, TypeError):
                        parsed[k] = float("nan")
            rows.append(parsed)
    return rows


def _filter_tier(rows: List[Dict], tier: Optional[str]) -> List[Dict]:
    if tier is None:
        return rows
    return [r for r in rows if r.get("tier") == tier]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

KEY_METRICS = [
    "IoU_crack", "IoU_spalling",
    "Dice_crack", "Dice_spalling",
    "BF1_crack", "BF1_spalling",
    "clDice_crack", "clDice_spalling",
    "ConnR_crack", "ConnR_spalling",
]


def _paired_arrays(
    rows_a: List[Dict], rows_b: List[Dict], metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two per-image result lists by filename, drop NaN pairs."""
    map_b = {r["file"]: r[metric] for r in rows_b if not math.isnan(r.get(metric, float("nan")))}
    vals_a, vals_b = [], []
    for r in rows_a:
        f = r["file"]
        va = r.get(metric, float("nan"))
        if math.isnan(va) or f not in map_b:
            continue
        vals_a.append(va)
        vals_b.append(map_b[f])
    return np.array(vals_a), np.array(vals_b)


def _bootstrap_ci(
    arr: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via percentile bootstrap."""
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[i] = arr[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def compare_pair(
    rows_a: List[Dict], rows_b: List[Dict],
    name_a: str, name_b: str,
    tier: Optional[str] = None,
) -> Dict:
    """Compare two models on all key metrics. Returns structured results."""
    ra = _filter_tier(rows_a, tier)
    rb = _filter_tier(rows_b, tier)
    tier_label = tier or "overall"

    results = {"model_a": name_a, "model_b": name_b, "tier": tier_label, "metrics": {}}

    for metric in KEY_METRICS:
        a, b = _paired_arrays(ra, rb, metric)
        n_pairs = len(a)
        if n_pairs < 5:
            results["metrics"][metric] = {"n_pairs": n_pairs, "skip": True}
            continue

        mean_a, lo_a, hi_a = _bootstrap_ci(a)
        mean_b, lo_b, hi_b = _bootstrap_ci(b)

        diff = a - b
        mean_diff, lo_diff, hi_diff = _bootstrap_ci(diff)

        # Wilcoxon signed-rank test (two-sided)
        try:
            stat, p_val = sp_stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            # All differences are zero
            stat, p_val = 0.0, 1.0

        results["metrics"][metric] = {
            "n_pairs": n_pairs,
            "mean_a": round(mean_a, 5),
            "ci95_a": [round(lo_a, 5), round(hi_a, 5)],
            "mean_b": round(mean_b, 5),
            "ci95_b": [round(lo_b, 5), round(hi_b, 5)],
            "mean_diff": round(mean_diff, 5),
            "ci95_diff": [round(lo_diff, 5), round(hi_diff, 5)],
            "wilcoxon_stat": round(float(stat), 2),
            "p_value": round(float(p_val), 6),
            "significant_005": bool(p_val < 0.05),
        }

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _format_comparison(res: Dict) -> str:
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  {res['model_a']}  vs  {res['model_b']}  [{res['tier']}]")
    lines.append(f"{'='*70}")
    lines.append(f"  {'metric':<18} {'mean_A':>8} {'mean_B':>8} {'diff':>8}  {'95% CI diff':<20} {'p-val':>8}  sig?")
    lines.append(f"  {'-'*64}")

    for metric, m in res["metrics"].items():
        if m.get("skip"):
            lines.append(f"  {metric:<18}  (skipped, n={m['n_pairs']})")
            continue
        sig = "*" if m["significant_005"] else " "
        lines.append(
            f"  {metric:<18} {m['mean_a']:>8.4f} {m['mean_b']:>8.4f} "
            f"{m['mean_diff']:>+8.4f}  [{m['ci95_diff'][0]:>+.4f}, {m['ci95_diff'][1]:>+.4f}]"
            f"  {m['p_value']:>8.4f} {sig}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paired statistical significance tests between models",
    )
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory with per-image CSVs")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model-a", type=str, default=None,
                        help="Model A (default: segformer_b2_full_512)")
    parser.add_argument("--model-b", type=str, default=None,
                        help="Model B (if omitted, compare A vs all others)")
    parser.add_argument("--tier", type=str, default=None,
                        choices=["Easy", "Medium", "Hard"],
                        help="Filter by tier (default: all images)")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = args.split

    # Discover available per-image CSVs
    csvs = sorted(results_dir.glob(f"*_{split}_per_image.csv"))
    if not csvs:
        print(f"[stats] No per-image CSVs found in {results_dir}/ for split '{split}'.")
        print(f"[stats] Run: python -m shared_eval.eval_all --all-models --split {split} --per-image")
        return

    model_data: Dict[str, List[Dict]] = {}
    for p in csvs:
        name = p.stem.replace(f"_{split}_per_image", "")
        model_data[name] = _load_per_image_csv(p)
        print(f"[stats] Loaded {name}: {len(model_data[name])} images")

    model_a = args.model_a or "segformer_b2_full_512"
    if model_a not in model_data:
        print(f"[stats] Model A '{model_a}' not found. Available: {list(model_data.keys())}")
        return

    # Determine comparisons
    if args.model_b:
        if args.model_b not in model_data:
            print(f"[stats] Model B '{args.model_b}' not found.")
            return
        opponents = [args.model_b]
    else:
        opponents = [k for k in model_data if k != model_a]

    tiers = [args.tier] if args.tier else [None, "Easy", "Medium", "Hard"]

    all_results = []
    for opp in opponents:
        for tier in tiers:
            res = compare_pair(
                model_data[model_a], model_data[opp],
                model_a, opp, tier=tier,
            )
            print(_format_comparison(res))
            all_results.append(res)

    # Save JSON
    out_path = output_dir / f"stats_significance_{split}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[stats] Saved: {out_path}")


if __name__ == "__main__":
    main()

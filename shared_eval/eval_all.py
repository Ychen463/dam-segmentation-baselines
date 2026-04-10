"""Unified evaluation script for all baselines.

Usage examples::

    # Single model, overall test metrics
    python -m shared_eval.eval_all --model deeplabv3p_r50_512 --split test

    # Single model, per-tier breakdown
    python -m shared_eval.eval_all --model deeplabv3p_r50_512 --split test --per-tier

    # All registered models
    python -m shared_eval.eval_all --all-models --split test --per-tier

    # Custom output directory
    python -m shared_eval.eval_all --model unet_r34_320 --split test --output-dir results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from baseline_unet import config as C
from baseline_unet.dataset import (
    DamSegmentDataset,
    build_transforms,
    read_split_file,
)
from baseline_unet.splits import SPLIT_FILES

from .metrics_full import SegMetricsFull
from .model_registry import get as get_entry, list_models, load_model

BF1_TOLERANCE_PX = 2
BATCH_SIZE = 4
NUM_WORKERS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _filter_by_tier(files: List[str], tier: str) -> List[str]:
    """Keep only files whose path starts with the given tier prefix (Easy/Medium/Hard)."""
    return [f for f in files if f.startswith(tier + "/")]


@torch.no_grad()
def _evaluate(model, loader: DataLoader, device: str) -> Dict[str, float]:
    metrics = SegMetricsFull(C.NUM_CLASSES, BF1_TOLERANCE_PX)
    metrics.reset()
    for images, masks, _rels in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        metrics.update(logits, masks)
    return metrics.compute()


def _run_eval(
    model_name: str, split: str, per_tier: bool, device: str,
) -> Dict[str, Dict[str, float]]:
    """Run evaluation, return dict with 'overall' and optionally tier keys."""
    entry = get_entry(model_name)
    img_size = entry.img_size

    print(f"[eval] Loading model: {model_name}")
    model = load_model(model_name, device=device)

    transform = build_transforms(img_size, train=False)

    # Read split files
    split_path = SPLIT_FILES[split]
    all_files = read_split_file(split_path)
    print(f"[eval] Split '{split}': {len(all_files)} images")

    results: Dict[str, Dict[str, float]] = {}

    # Overall
    ds = DamSegmentDataset(C.DATA_ROOT, all_files, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    print(f"[eval] Running overall evaluation ...")
    results["overall"] = _evaluate(model, loader, device)

    # Per-tier
    if per_tier:
        for tier in C.DIFFICULTIES:
            tier_files = _filter_by_tier(all_files, tier)
            if not tier_files:
                print(f"[eval] Tier '{tier}': 0 images, skipping")
                continue
            print(f"[eval] Tier '{tier}': {len(tier_files)} images")
            ds_t = DamSegmentDataset(C.DATA_ROOT, tier_files, transform=transform)
            loader_t = DataLoader(ds_t, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=(device == "cuda"))
            results[tier] = _evaluate(model, loader_t, device)

    return results


def _format_results(results: Dict[str, Dict[str, float]]) -> str:
    """Pretty-print results."""
    lines = []
    for group_name, m in results.items():
        lines.append(f"\n=== {group_name} ===")
        lines.append(
            f"  IoU     bg={m['IoU_background']:.4f}  crack={m['IoU_crack']:.4f}"
            f"  spalling={m['IoU_spalling']:.4f}  | mIoU_fg={m['mIoU_fg']:.4f}"
        )
        lines.append(
            f"  Dice    bg={m['Dice_background']:.4f}  crack={m['Dice_crack']:.4f}"
            f"  spalling={m['Dice_spalling']:.4f}"
        )
        lines.append(
            f"  BF1     crack={m['BF1_crack']:.4f}  spalling={m['BF1_spalling']:.4f}"
            f"  | BF1_fg_mean={m['BF1_fg_mean']:.4f}"
        )
        lines.append(
            f"  clDice  crack={m['clDice_crack']:.4f}  spalling={m['clDice_spalling']:.4f}"
            f"  | clDice_fg_mean={m['clDice_fg_mean']:.4f}"
        )
        lines.append(
            f"  ConnR   crack={m['ConnR_crack']:.4f}  spalling={m['ConnR_spalling']:.4f}"
            f"  | ConnR_fg_mean={m['ConnR_fg_mean']:.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for all baselines",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Model name from registry")
    group.add_argument("--all-models", action="store_true",
                       help="Evaluate all registered models")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--per-tier", action="store_true",
                        help="Also report per-difficulty-tier metrics")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for JSON output (default: results/)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (cuda/mps/cpu)")
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"[eval] Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list_models() if args.all_models else [args.model]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        results = _run_eval(model_name, args.split, args.per_tier, device)

        # Print
        print(_format_results(results))

        # Save JSON
        out_path = output_dir / f"{model_name}_{args.split}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[eval] Saved: {out_path}")


if __name__ == "__main__":
    main()

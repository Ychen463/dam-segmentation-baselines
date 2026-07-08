#!/usr/bin/env python3
"""Evaluate topology metrics (CompP, CompF1, PathCont) on all key models.

Runs the extended metric suite on saved checkpoints without retraining.
Outputs a summary table and per-model JSON files.

Usage:
    python scripts/eval_topology.py [--per-image] [--per-tier]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_unet import config as C
from baseline_unet.dataset import DamSegmentDataset, build_transforms, read_split_file
from baseline_unet.splits import SPLIT_FILES
from shared_eval.metrics_full import SegMetricsFull
from shared_eval.model_registry import get as get_entry, load_model

BF1_TOLERANCE_PX = 2
BATCH_SIZE = 4
NUM_WORKERS = 2

# Models to evaluate (must be in model_registry)
KEY_MODELS = [
    "plain_segformer_P0",          # (a) SegFormer-B2 baseline
    "dscformer_plain_G0",          # (b) DSConv-only
    "dscformer_srl_G1",            # (e) DSConv+SRL (Teacher 1)
    "dscformer_full_G2",           # (c) DTKD (final model)
    "mask2former_swin_small_512",  # Mask2Former baseline
    "segformer_b2_plain_512",      # SegFormer-B2 (baseline_segformer)
]


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _filter_by_tier(files, tier):
    return [f for f in files if f.startswith(tier + "/")]


@torch.no_grad()
def _evaluate_topology(model, loader, device):
    """Evaluate with topology metrics enabled."""
    metrics = SegMetricsFull(C.NUM_CLASSES, BF1_TOLERANCE_PX, compute_topology=True)
    metrics.reset()
    for images, masks, _rels in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        metrics.update(logits, masks)
    return metrics.compute()


def _format_topology_table(all_results):
    """Format a compact comparison table of topology metrics."""
    lines = []
    header = (f"{'Model':<25} {'CompR_cr':>8} {'CompP_cr':>8} {'CompF1_cr':>9} "
              f"{'PathC_cr':>8} {'CompR_sp':>8} {'CompP_sp':>8} {'CompF1_sp':>9} "
              f"{'PathC_sp':>8}")
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in all_results.items():
        lines.append(
            f"{name:<25} "
            f"{m.get('ConnR_crack', 0)*100:>7.1f}% "
            f"{m.get('CompP_crack', 0)*100:>7.1f}% "
            f"{m.get('CompF1_crack', 0)*100:>8.1f}% "
            f"{m.get('PathCont_crack', 0)*100:>7.1f}% "
            f"{m.get('ConnR_spalling', 0)*100:>7.1f}% "
            f"{m.get('CompP_spalling', 0)*100:>7.1f}% "
            f"{m.get('CompF1_spalling', 0)*100:>8.1f}% "
            f"{m.get('PathCont_spalling', 0)*100:>7.1f}%"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Topology metric evaluation")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--per-tier", action="store_true")
    parser.add_argument("--per-image", action="store_true")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all key models)")
    parser.add_argument("--output-dir", default="results/topology")
    args = parser.parse_args()

    device = _pick_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = args.models.split(",") if args.models else KEY_MODELS
    split_path = SPLIT_FILES[args.split]
    all_files = read_split_file(split_path)

    all_results = {}

    for model_name in models:
        try:
            entry = get_entry(model_name)
        except KeyError:
            print(f"[skip] {model_name} not in registry")
            continue

        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")

        model = load_model(model_name, device=device)
        transform = build_transforms(entry.img_size, train=False)
        ds = DamSegmentDataset(C.DATA_ROOT, all_files, transform=transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))

        m = _evaluate_topology(model, loader, device)
        all_results[model_name] = m

        # Print key topology metrics
        print(f"  CompR_cr={m.get('ConnR_crack',0)*100:.1f}%  "
              f"CompP_cr={m.get('CompP_crack',0)*100:.1f}%  "
              f"CompF1_cr={m.get('CompF1_crack',0)*100:.1f}%  "
              f"PathCont_cr={m.get('PathCont_crack',0)*100:.1f}%")
        print(f"  CompR_sp={m.get('ConnR_spalling',0)*100:.1f}%  "
              f"CompP_sp={m.get('CompP_spalling',0)*100:.1f}%  "
              f"CompF1_sp={m.get('CompF1_spalling',0)*100:.1f}%  "
              f"PathCont_sp={m.get('PathCont_spalling',0)*100:.1f}%")

        # Per-tier
        if args.per_tier:
            for tier in C.DIFFICULTIES:
                tier_files = _filter_by_tier(all_files, tier)
                if not tier_files:
                    continue
                ds_t = DamSegmentDataset(C.DATA_ROOT, tier_files, transform=transform)
                loader_t = DataLoader(ds_t, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=NUM_WORKERS)
                m_t = _evaluate_topology(model, loader_t, device)
                print(f"  [{tier}] CompR_cr={m_t.get('ConnR_crack',0)*100:.1f}%  "
                      f"CompP_cr={m_t.get('CompP_crack',0)*100:.1f}%  "
                      f"PathCont_cr={m_t.get('PathCont_crack',0)*100:.1f}%")

        # Save per-model JSON
        out_path = output_dir / f"{model_name}_topology.json"
        with open(out_path, "w") as f:
            json.dump(m, f, indent=2)

    # Summary table
    print(f"\n{'='*50}")
    print("  TOPOLOGY METRICS SUMMARY")
    print(f"{'='*50}")
    print(_format_topology_table(all_results))

    # Save summary
    summary_path = output_dir / "topology_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {summary_path}")


if __name__ == "__main__":
    main()

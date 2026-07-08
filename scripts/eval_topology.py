#!/usr/bin/env python3
"""Evaluate topology metrics (CompP, CompF1, PathCont) on all key models.

Runs the extended metric suite on saved checkpoints without retraining.
Outputs a summary table and per-model JSON files.

Usage:
    # Registry-based (needs model in registry):
    python scripts/eval_topology.py --models dscformer_plain_G0,dscformer_srl_G1

    # Direct checkpoint mode (any run directory):
    python scripts/eval_topology.py --run-dirs \
        dsconv_G0_s42,rerun_hetero_cc_s42,rerun_hetero_s42

    # With per-tier breakdown:
    python scripts/eval_topology.py --run-dirs dsconv_G0_s42 --per-tier
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_unet import config as C
from baseline_unet.dataset import DamSegmentDataset, build_transforms, read_split_file
from baseline_unet.splits import SPLIT_FILES
from shared_eval.metrics_full import SegMetricsFull

BF1_TOLERANCE_PX = 2
BATCH_SIZE = 4
NUM_WORKERS = 2

RUNS_DIR = Path(__file__).resolve().parent.parent / "full_method" / "runs"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _filter_by_tier(files, tier):
    return [f for f in files if f.startswith(tier + "/")]


def _load_model_from_run_dir(run_name: str, device: str):
    """Load a DSCformerDam model from a run directory (no registry needed)."""
    from full_method import config as fm_C
    from full_method.model import DSCformerDam

    ckpt_path = RUNS_DIR / run_name / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Try to load config from checkpoint
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Build model with default config (works for most ablation runs)
    cfg = fm_C.RunCfg()

    # Detect DCNv2 or standard conv from checkpoint keys
    model_keys = state.get("model", state).keys()
    key_str = " ".join(model_keys)
    if "dcn_x1" in key_str or "dcn_y1" in key_str:
        cfg.use_dcnv2_branch = True
    elif "expand" in key_str and "dsconv_x" not in key_str:
        cfg.use_standard_conv_branch = True

    model = DSCformerDam(cfg.pretrained, fm_C.NUM_CLASSES, cfg=cfg)
    model.load_state_dict(state["model"], strict=False)

    # Wrap to return (B, C, H, W) logits
    class _Wrapper(torch.nn.Module):
        def __init__(self, m, size=512):
            super().__init__()
            self.m = m
            self.size = size
        def forward(self, x):
            out = self.m(x)
            logits = out["seg_logits"]
            return F.interpolate(logits, size=(self.size, self.size),
                                 mode="bilinear", align_corners=False)

    wrapper = _Wrapper(model)
    wrapper.to(device).eval()
    return wrapper


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
    header = (f"{'Model':<30} {'mIoU_fg':>7} {'CompR_cr':>8} {'CompP_cr':>8} "
              f"{'CompF1_cr':>9} {'PathC_cr':>8} {'CompR_sp':>8} {'CompP_sp':>8} "
              f"{'CompF1_sp':>9} {'PathC_sp':>8}")
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in all_results.items():
        lines.append(
            f"{name:<30} "
            f"{m.get('mIoU_fg', 0)*100:>6.1f}% "
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
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated registry model names")
    parser.add_argument("--run-dirs", type=str, default=None,
                        help="Comma-separated run directory names under full_method/runs/")
    parser.add_argument("--output-dir", default="results/topology")
    args = parser.parse_args()

    if not args.models and not args.run_dirs:
        parser.error("Specify --models or --run-dirs")

    device = _pick_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = SPLIT_FILES[args.split]
    all_files = read_split_file(split_path)
    transform = build_transforms(512, train=False)
    ds = DamSegmentDataset(C.DATA_ROOT, all_files, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))

    all_results = {}

    # --- Registry-based models ---
    if args.models:
        from shared_eval.model_registry import get as get_entry, load_model
        for model_name in args.models.split(","):
            model_name = model_name.strip()
            try:
                entry = get_entry(model_name)
            except KeyError:
                print(f"[skip] {model_name} not in registry")
                continue
            print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
            model = load_model(model_name, device=device)
            m = _evaluate_topology(model, loader, device)
            all_results[model_name] = m
            _print_result(model_name, m)
            if args.per_tier:
                _print_per_tier(model, all_files, transform, device)
            _save_json(output_dir, model_name, m)

    # --- Direct run-dir mode ---
    if args.run_dirs:
        for run_name in args.run_dirs.split(","):
            run_name = run_name.strip()
            print(f"\n{'='*50}\n  {run_name}\n{'='*50}")
            try:
                model = _load_model_from_run_dir(run_name, device)
            except FileNotFoundError as e:
                print(f"[skip] {e}")
                continue
            m = _evaluate_topology(model, loader, device)
            all_results[run_name] = m
            _print_result(run_name, m)
            if args.per_tier:
                _print_per_tier(model, all_files, transform, device)
            _save_json(output_dir, run_name, m)

    # Summary table
    print(f"\n{'='*60}")
    print("  TOPOLOGY METRICS SUMMARY")
    print(f"{'='*60}")
    print(_format_topology_table(all_results))

    # Save summary
    summary_path = output_dir / "topology_summary.json"
    # Merge with existing summary if present
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {summary_path}")


def _print_result(name, m):
    print(f"  mIoU_fg={m.get('mIoU_fg',0)*100:.1f}%")
    print(f"  CompR_cr={m.get('ConnR_crack',0)*100:.1f}%  "
          f"CompP_cr={m.get('CompP_crack',0)*100:.1f}%  "
          f"CompF1_cr={m.get('CompF1_crack',0)*100:.1f}%  "
          f"PathCont_cr={m.get('PathCont_crack',0)*100:.1f}%")
    print(f"  CompR_sp={m.get('ConnR_spalling',0)*100:.1f}%  "
          f"CompP_sp={m.get('CompP_spalling',0)*100:.1f}%  "
          f"CompF1_sp={m.get('CompF1_spalling',0)*100:.1f}%  "
          f"PathCont_sp={m.get('PathCont_spalling',0)*100:.1f}%")


def _print_per_tier(model, all_files, transform, device):
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
              f"CompF1_cr={m_t.get('CompF1_crack',0)*100:.1f}%  "
              f"PathCont_cr={m_t.get('PathCont_crack',0)*100:.1f}%")


def _save_json(output_dir, name, m):
    out_path = output_dir / f"{name}_topology.json"
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()

"""Evaluate matched multi-seed DTKD mechanism isolation experiment.

Compares 5 conditions × 3 seeds to determine whether heterogeneous
teachers produce a net gain over the no-KD baseline.

Usage on RunPod:
    cd /workspace/dam-segmentation-baselines
    python scripts/eval_matched_multiseed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_unet.dataset import (
    DamSegmentDataset, build_transforms, read_split_file,
)
from baseline_unet import config as base_C
from full_method import config as C
from full_method.model import DSCformerDam
from shared_eval.metrics_full import SegMetricsFull

# Conditions and their run name patterns
CONDITIONS = [
    ("No-KD",           "rerun_nokd"),
    ("T1-only",         "rerun_t1only"),
    ("Duplicated T1",   "rerun_dup_t1"),
    ("Hetero equal-wt", "rerun_hetero"),
    ("Hetero class-cond", "rerun_hetero_cc"),
]
SEEDS = [42, 123, 2024]

METRIC_KEYS = ["mIoU_fg", "IoU_crack", "IoU_spalling", "BF1_fg_mean",
               "clDice_fg_mean", "ConnR_fg_mean"]


def evaluate_run(run_dir: Path, device, test_loader):
    """Evaluate a single run, return metrics dict or None if not found."""
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        return None

    cfg = C.RunCfg()
    cfg.model_type = "dscformer"
    model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    key = "ema_model" if "ema_model" in state else "model"
    model.load_state_dict(state[key], strict=False)
    model.eval()

    metrics = SegMetricsFull(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    metrics.reset()
    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs = imgs.to(device).float()
            masks = masks.to(device).long()
            out = model(imgs)
            seg_logits = F.interpolate(out["seg_logits"], masks.shape[-2:],
                                       mode="bilinear", align_corners=False)
            metrics.update(seg_logits, masks)

    return metrics.compute()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = C.RunCfg()
    test_files = read_split_file(C.SPLIT_FILES["test"])
    test_ds = DamSegmentDataset(base_C.DATA_ROOT, test_files,
                                 build_transforms(cfg.img_size, train=False))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                              num_workers=2, pin_memory=(device == "cuda"))
    print(f"Test set: {len(test_files)} images")

    # Evaluate all runs
    all_results = {}  # {condition: {seed: metrics}}
    for cond_name, run_prefix in CONDITIONS:
        all_results[cond_name] = {}
        for seed in SEEDS:
            run_name = f"{run_prefix}_s{seed}"
            run_dir = C.RUNS_DIR / run_name
            print(f"\n  Evaluating: {run_name}...", end=" ")
            m = evaluate_run(run_dir, device, test_loader)
            if m is None:
                print("SKIPPED (not found)")
                continue
            all_results[cond_name][seed] = m
            print(f"mIoU_fg={m['mIoU_fg']:.4f}")

            del m
            if device == "cuda":
                torch.cuda.empty_cache()

    # Compute statistics
    print(f"\n{'='*100}")
    print("  RESULTS: Matched Multi-Seed DTKD Mechanism Isolation")
    print(f"{'='*100}")
    print(f"{'Condition':<22s} {'Seeds':>5s} {'mIoU_fg':>10s} {'IoU_cr':>10s} "
          f"{'IoU_sp':>10s} {'BF1_fg':>10s} {'clDice_fg':>10s}")
    print("-" * 100)

    stats = {}
    for cond_name, _ in CONDITIONS:
        seed_results = all_results[cond_name]
        if not seed_results:
            print(f"{cond_name:<22s} {'0':>5s}   — no results —")
            continue

        n = len(seed_results)
        # Compute mean ± std for each metric
        cond_stats = {}
        row_parts = [f"{cond_name:<22s} {n:>5d}"]
        for k in METRIC_KEYS[:5]:  # first 5 metrics for display
            vals = [seed_results[s][k] for s in seed_results]
            mean = np.mean(vals)
            std = np.std(vals, ddof=1) if len(vals) > 1 else 0
            cond_stats[k] = {"mean": mean, "std": std, "values": vals}
            row_parts.append(f"{mean*100:>5.2f}±{std*100:.2f}")
        print("  ".join(row_parts))
        stats[cond_name] = cond_stats

    # Key comparisons
    print(f"\n{'='*80}")
    print("  KEY COMPARISONS (mIoU_fg)")
    print(f"{'='*80}")

    if "No-KD" in stats and "Hetero equal-wt" in stats:
        nokd = stats["No-KD"]["mIoU_fg"]
        hetero = stats["Hetero equal-wt"]["mIoU_fg"]
        delta = hetero["mean"] - nokd["mean"]
        print(f"\n  Hetero equal-wt vs No-KD:")
        print(f"    No-KD:        {nokd['mean']*100:.2f} ± {nokd['std']*100:.2f}")
        print(f"    Hetero eq-wt: {hetero['mean']*100:.2f} ± {hetero['std']*100:.2f}")
        print(f"    Delta:        {delta*100:+.2f}")
        print(f"    → {'NET GAIN' if delta > 0 else 'NO NET GAIN'} "
              f"({'significant' if abs(delta) > max(nokd['std'], hetero['std']) else 'within noise'})")

    if "No-KD" in stats and "Hetero class-cond" in stats:
        nokd = stats["No-KD"]["mIoU_fg"]
        hetero_cc = stats["Hetero class-cond"]["mIoU_fg"]
        delta = hetero_cc["mean"] - nokd["mean"]
        print(f"\n  Hetero class-cond vs No-KD:")
        print(f"    No-KD:           {nokd['mean']*100:.2f} ± {nokd['std']*100:.2f}")
        print(f"    Hetero class-cd: {hetero_cc['mean']*100:.2f} ± {hetero_cc['std']*100:.2f}")
        print(f"    Delta:           {delta*100:+.2f}")
        print(f"    → {'NET GAIN' if delta > 0 else 'NO NET GAIN'} "
              f"({'significant' if abs(delta) > max(nokd['std'], hetero_cc['std']) else 'within noise'})")

    if "No-KD" in stats and "T1-only" in stats:
        nokd = stats["No-KD"]["mIoU_fg"]
        t1only = stats["T1-only"]["mIoU_fg"]
        delta = t1only["mean"] - nokd["mean"]
        print(f"\n  T1-only vs No-KD:")
        print(f"    No-KD:   {nokd['mean']*100:.2f} ± {nokd['std']*100:.2f}")
        print(f"    T1-only: {t1only['mean']*100:.2f} ± {t1only['std']*100:.2f}")
        print(f"    Delta:   {delta*100:+.2f}")

    # Save full results
    save_data = {
        "conditions": {},
        "per_seed": {},
    }
    for cond_name, _ in CONDITIONS:
        if cond_name in stats:
            save_data["conditions"][cond_name] = {
                k: {"mean": round(v["mean"], 6), "std": round(v["std"], 6),
                    "values": [round(x, 6) for x in v["values"]]}
                for k, v in stats[cond_name].items()
            }
        for seed, m in all_results.get(cond_name, {}).items():
            key = f"{cond_name}_s{seed}"
            save_data["per_seed"][key] = {
                k: round(v, 6) for k, v in m.items() if isinstance(v, (int, float))
            }

    out_path = ROOT / "results" / "matched_multiseed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

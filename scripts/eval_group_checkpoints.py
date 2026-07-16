#!/usr/bin/env python3
"""Evaluate existing GR checkpoints for per-group metrics at multiple epochs.

Loads last.pt or epoch-specific checkpoints, runs val evaluation with
PerGroupEvaluator, and outputs a comparison table.

Usage:
    python scripts/eval_group_checkpoints.py

Outputs:
    full_method/runs/<run>/group_eval_epochs.csv  (per-run)
    stdout: comparison table across runs
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline_unet.dataset import build_transforms, read_split_file
from full_method import config as C
from full_method.dataset import FullMethodDataset, dict_collate
from full_method.group_eval import PerGroupEvaluator
from full_method.group_sampler import load_group_assignments
from full_method.model import DSCformerDam


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUNS = {
    "G0": "gr_g0_baseline",
    "G0R": "gr_g0r_replace",
    "G2": "gr_g2_inverse_sqrt",
}

EVAL_EPOCHS = [1, 5, 10, 15, 25, 50]

SPLIT_DIR = Path("baseline_unet/splits/balanced_group_split")
GA_PATH = SPLIT_DIR / "group_assignments.json"

IMG_SIZE = 512
BATCH_SIZE = 8
NUM_CLASSES = 3


def build_eval_loader(val_files, group_map, device):
    records = [{"id": f, "rel": f, "tier": 0, "has_spalling": False,
                "group_id": group_map.get(f, -1)} for f in val_files]
    ds = FullMethodDataset(C.DATA_ROOT, records,
                           build_transforms(IMG_SIZE, train=False))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=(device == "cuda"),
                      drop_last=False, collate_fn=dict_collate)


def find_checkpoint_for_epoch(run_dir: Path, target_epoch: int):
    """Find the right checkpoint for a given epoch.

    Strategy:
    - last.pt always contains the final epoch's model
    - best.pt contains the best val mIoU checkpoint
    - For intermediate epochs, we need to check metrics.csv to find
      which checkpoint was best at that epoch. Since we only have last.pt
      and best.pt, we can only evaluate:
      - The "best" checkpoint (best.pt)
      - The "last" checkpoint (last.pt = epoch 50)

    But the metrics.csv has per-epoch val metrics already recorded during
    training. We can just read those instead of re-evaluating.
    """
    # For group metrics we DO need to re-evaluate (they weren't all saved)
    if target_epoch == 50:
        p = run_dir / "last.pt"
        if p.exists():
            return p, "last"
    p = run_dir / "best.pt"
    if p.exists():
        return p, "best"
    return None, None


def load_epoch_metrics_from_csv(run_dir: Path):
    """Read per-epoch val metrics from metrics.csv."""
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        return {}
    rows = {}
    with open(csv_path) as f:
        # Skip comment lines
        lines = [l for l in f if not l.startswith("#")]
    if not lines:
        return {}
    reader = csv.DictReader(lines)
    for row in reader:
        try:
            ep = int(row["epoch"])
        except (ValueError, KeyError):
            continue
        rows[ep] = {
            "mIoU_fg": float(row.get("mIoU_fg") or "nan"),
            "IoU_crack": float(row.get("IoU_crack") or "nan"),
            "IoU_spalling": float(row.get("IoU_spalling") or "nan"),
            "BF1_crack": float(row.get("BF1_crack") or "nan"),
            "BF1_fg_mean": float(row.get("BF1_fg_mean") or "nan"),
        }
    return rows


@torch.no_grad()
def evaluate_with_groups(model, loader, group_eval, device):
    """Run val evaluation and return aggregate + group metrics."""
    model.eval()
    group_eval.reset()

    all_miou_fg = []
    total_correct = 0
    total_pixels = 0

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True).float()
        masks = batch["mask"].to(device, non_blocking=True).long()
        group_ids = batch["group_id"]

        outputs = model(imgs)
        seg_logits = F.interpolate(outputs["seg_logits"].float(),
                                   masks.shape[-2:], mode="bilinear",
                                   align_corners=False)
        group_eval.update(seg_logits, masks, group_ids)

        # Quick aggregate metrics
        preds = seg_logits.argmax(1)
        total_correct += (preds == masks).sum().item()
        total_pixels += masks.numel()

    gm = group_eval.compute_group_metrics()
    return gm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}")

    # Load group assignments and val files
    group_map = load_group_assignments(GA_PATH)
    val_files = read_split_file(SPLIT_DIR / "val.txt")
    print(f"[eval] {len(val_files)} val images, {len(set(group_map.values()))} groups")

    loader = build_eval_loader(val_files, group_map, device)

    # Build model template (will load weights per checkpoint)
    cfg = C.RunCfg()
    cfg.model_type = "dscformer"

    # Also read per-epoch metrics from CSV for aggregate numbers
    print(f"\n{'='*100}")
    print(f"{'Run':<6} {'Epoch':<7} {'Src':<5} {'mIoU_fg':>8} {'IoU_cr':>7} {'IoU_sp':>7} "
          f"{'CVaR20':>7} {'p10':>7} {'worst':>7} {'elig_w':>7} {'std':>7} {'gap':>7} "
          f"{'w_cr_R':>7}")
    print(f"{'-'*100}")

    all_results = []

    for run_label, run_name in RUNS.items():
        run_dir = C.RUNS_DIR / run_name
        if not run_dir.exists():
            print(f"[WARN] {run_dir} not found, skipping")
            continue

        csv_metrics = load_epoch_metrics_from_csv(run_dir)

        # Evaluate best.pt and last.pt with group metrics
        for ckpt_name in ["best", "last"]:
            ckpt_path = run_dir / f"{ckpt_name}.pt"
            if not ckpt_path.exists():
                continue

            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            epoch = state.get("epoch", "?")
            src_label = f"{ckpt_name}(e{epoch})"

            model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)
            weights = state.get("ema_model", state["model"])
            model.load_state_dict(weights, strict=True)

            group_eval = PerGroupEvaluator(NUM_CLASSES)
            gm = evaluate_with_groups(model, loader, group_eval, device)
            gs = gm['summary']

            # Get aggregate metrics from CSV for this epoch
            ep_int = int(epoch) if isinstance(epoch, int) else 0
            csv_row = csv_metrics.get(ep_int, {})
            miou_fg = csv_row.get("mIoU_fg", 0)
            iou_cr = csv_row.get("IoU_crack", 0)
            iou_sp = csv_row.get("IoU_spalling", 0)

            print(f"{run_label:<6} {src_label:<7} {'ckpt':<5} "
                  f"{miou_fg:>8.4f} {iou_cr:>7.4f} {iou_sp:>7.4f} "
                  f"{gs['cvar20_mIoU_fg']:>7.4f} {gs['p10_mIoU_fg']:>7.4f} "
                  f"{gs['worst_group_mIoU_fg']:>7.4f} "
                  f"{gs['eligible_worst_group_mIoU_fg']:>7.4f} "
                  f"{gs['std_mIoU_fg']:>7.4f} "
                  f"{gs['avg_worst_gap']:>7.4f} "
                  f"{gs['worst_crack_recall']:>7.4f}")

            result = {
                "run": run_label,
                "run_name": run_name,
                "checkpoint": ckpt_name,
                "epoch": ep_int,
                **{f"csv_{k}": v for k, v in csv_row.items()},
                **gs,
            }
            all_results.append(result)

            # Save per-group detail
            detail_path = run_dir / f"group_detail_{ckpt_name}.json"
            with open(detail_path, "w") as f:
                # Convert numpy types for JSON
                detail = {}
                for gid, gdata in gm['per_group'].items():
                    detail[str(gid)] = {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                                        for k, v in gdata.items()}
                json.dump({"summary": gs, "per_group": detail}, f, indent=2)

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        # Also print CSV-only rows for intermediate epochs
        for ep in EVAL_EPOCHS:
            if ep in csv_metrics and ep not in [r.get("epoch") for r in all_results
                                                 if r["run"] == run_label]:
                row = csv_metrics[ep]
                print(f"{run_label:<6} e{ep:<6} {'csv':<5} "
                      f"{row['mIoU_fg']:>8.4f} {row['IoU_crack']:>7.4f} "
                      f"{row['IoU_spalling']:>7.4f} "
                      f"{'--':>7} {'--':>7} {'--':>7} {'--':>7} {'--':>7} {'--':>7} {'--':>7}")

    print(f"{'='*100}")

    # Write combined CSV
    out_csv = C.RUNS_DIR / "gr_group_eval_comparison.csv"
    if all_results:
        keys = list(all_results[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_results:
                w.writerow(r)
        print(f"\n[eval] Wrote {out_csv}")

    # Decision summary
    print("\n--- Decision Summary ---")
    g0_results = [r for r in all_results if r["run"] == "G0"]
    g2_results = [r for r in all_results if r["run"] == "G2"]

    if g0_results and g2_results:
        # Compare best checkpoints
        g0_best = next((r for r in g0_results if r["checkpoint"] == "best"), g0_results[0])
        g2_best = next((r for r in g2_results if r["checkpoint"] == "best"), g2_results[0])
        g0_last = next((r for r in g0_results if r["checkpoint"] == "last"), None)
        g2_last = next((r for r in g2_results if r["checkpoint"] == "last"), None)

        print(f"Best checkpoint comparison (G2 vs G0):")
        print(f"  CVaR20: G0={g0_best['cvar20_mIoU_fg']:.4f} G2={g2_best['cvar20_mIoU_fg']:.4f} "
              f"Δ={g2_best['cvar20_mIoU_fg']-g0_best['cvar20_mIoU_fg']:+.4f}")
        print(f"  p10:    G0={g0_best['p10_mIoU_fg']:.4f} G2={g2_best['p10_mIoU_fg']:.4f} "
              f"Δ={g2_best['p10_mIoU_fg']-g0_best['p10_mIoU_fg']:+.4f}")

        if g0_last and g2_last:
            print(f"\nLast checkpoint comparison (G2 vs G0, epoch 50):")
            print(f"  CVaR20: G0={g0_last['cvar20_mIoU_fg']:.4f} G2={g2_last['cvar20_mIoU_fg']:.4f} "
                  f"Δ={g2_last['cvar20_mIoU_fg']-g0_last['cvar20_mIoU_fg']:+.4f}")
            print(f"  p10:    G0={g0_last['p10_mIoU_fg']:.4f} G2={g2_last['p10_mIoU_fg']:.4f} "
                  f"Δ={g2_last['p10_mIoU_fg']-g0_last['p10_mIoU_fg']:+.4f}")

        # Stop criteria
        best_cvar_delta = g2_best['cvar20_mIoU_fg'] - g0_best['cvar20_mIoU_fg']
        last_cvar_delta = (g2_last['cvar20_mIoU_fg'] - g0_last['cvar20_mIoU_fg']) if g0_last and g2_last else 0
        max_cvar_delta = max(best_cvar_delta, last_cvar_delta)

        print(f"\nStop criteria: max CVaR20 improvement = {max_cvar_delta:+.4f}")
        if max_cvar_delta < 1.0:
            print("  → CVaR20 improvement < 1.0: STOP group-aware fine-tuning.")
        else:
            print("  → CVaR20 improvement >= 1.0: consider continuing.")


if __name__ == "__main__":
    main()

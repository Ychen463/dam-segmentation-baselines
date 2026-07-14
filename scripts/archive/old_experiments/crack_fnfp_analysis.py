"""Crack FN vs FP analysis: diagnose why Crack IoU is the bottleneck.

For each test sample, compute:
  - TP, FP, FN pixel counts for crack class
  - Precision = TP/(TP+FP), Recall = TP/(TP+FN)
  - FN characterization: are missed cracks thin, boundary, or interior?
  - FP characterization: are false cracks near real cracks (boundary leakage)
    or isolated (hallucination)?
  - Width analysis: recall as a function of local crack width

Runs inference on test set using best G1 checkpoint.

Usage:
    python scripts/crack_fnfp_analysis.py
    python scripts/crack_fnfp_analysis.py --checkpoint runs/dkd10_no_srl/best.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False

from baseline_unet.dataset import build_transforms, read_split_file
from full_method import config as C
from full_method.config import RunCfg, apply_preset
from full_method.dataset import FullMethodDataset, build_records, dict_collate
from full_method.model import DSCformerDam


def width_map_from_mask(crack_mask: torch.Tensor) -> torch.Tensor:
    """Estimate local crack width via multi-scale erosion.

    Args:
        crack_mask: (B, H, W) binary crack mask.

    Returns:
        (B, H, W) estimated width in pixels (0 for non-crack).
    """
    crack_4d = crack_mask.float().unsqueeze(1)  # (B,1,H,W)
    width = torch.zeros_like(crack_mask, dtype=torch.float32)
    for r in [1, 2, 3, 5, 7, 10]:
        ks = 2 * r + 1
        eroded = -F.max_pool2d(-crack_4d, ks, stride=1, padding=r)
        width = width + eroded[:, 0] * 2.0
    width = width.clamp(min=0.0)
    # Only keep values where crack exists
    width = width * crack_mask.float()
    # Pixels that are crack but survived no erosion get width=1
    width = torch.where((crack_mask > 0) & (width < 1), torch.ones_like(width), width)
    return width


def boundary_mask(mask: torch.Tensor, cls: int, width: int = 3) -> torch.Tensor:
    """Binary boundary strip for a class. Returns (B,H,W) bool."""
    cls_mask = (mask == cls).float().unsqueeze(1)
    if cls_mask.sum() < 1:
        return mask.new_zeros(mask.shape, dtype=torch.bool)
    ks = 2 * width + 1
    dil = F.max_pool2d(cls_mask, ks, stride=1, padding=width)
    ero = -F.max_pool2d(-cls_mask, ks, stride=1, padding=width)
    return ((dil - ero) > 0.5)[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Model checkpoint (default: G1 best)")
    parser.add_argument("--preset", default="DKD10",
                        help="Config preset (default: DKD10)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fnfp] device={device}")

    cfg = RunCfg()
    apply_preset(cfg, args.preset)

    # Load test records
    test_files = read_split_file(C.SPLIT_FILES["test"])
    records = build_records(test_files, C.DATA_ROOT)
    print(f"[fnfp] {len(records)} test samples")

    tier_map = {r["id"]: r["tier"] for r in records}
    tier_names = {0: "Easy", 1: "Medium", 2: "Hard"}

    tfm = build_transforms(cfg.img_size, train=False)
    ds = FullMethodDataset(C.DATA_ROOT, records, transform=tfm)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2,
                        collate_fn=dict_collate, pin_memory=True)

    # Load model
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
    else:
        ckpt_path = (C.PKG_DIR / cfg.kd_teacher_checkpoint).resolve()
    print(f"[fnfp] loading model from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # Accumulators
    overall = {"tp": 0, "fp": 0, "fn": 0}
    by_tier = {t: {"tp": 0, "fp": 0, "fn": 0} for t in tier_names.values()}

    # FN characterization
    fn_thin = 0      # FN pixels with width <= 4
    fn_medium = 0    # FN pixels with width 5-10
    fn_wide = 0      # FN pixels with width > 10
    fn_boundary = 0  # FN pixels on crack boundary strip
    fn_interior = 0  # FN pixels in crack interior

    # FP characterization
    fp_near_crack = 0    # FP within 5px of real crack
    fp_near_spalling = 0 # FP within 5px of spalling
    fp_isolated = 0      # FP far from any FG

    # Width-binned recall
    width_bins = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 20), (20, 999)]
    width_tp = {b: 0 for b in width_bins}
    width_total = {b: 0 for b in width_bins}

    # Per-sample records
    per_sample = []

    print("[fnfp] running inference ...")
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device).float()
            masks = batch["mask"].to(device).long()
            sids = batch["sample_id"]
            H, W = masks.shape[-2:]

            outputs = model(imgs)
            logits = F.interpolate(outputs["seg_logits"].float(), (H, W),
                                   mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1)  # (B, H, W)

            gt_crack = (masks == 1)     # (B, H, W) bool
            pred_crack = (preds == 1)   # (B, H, W) bool

            tp_map = gt_crack & pred_crack
            fp_map = ~gt_crack & pred_crack
            fn_map = gt_crack & ~pred_crack

            # Width map of GT crack
            wmap = width_map_from_mask(gt_crack)

            # Crack boundary vs interior
            crack_bdry = boundary_mask(masks, cls=1, width=3)
            crack_interior = gt_crack & ~crack_bdry

            # For FP characterization: proximity to real crack and spalling
            crack_dilated = F.max_pool2d(gt_crack.float().unsqueeze(1), 11, 1, 5)[:, 0] > 0.5
            spalling_mask = (masks == 2)
            spalling_dilated = torch.zeros_like(spalling_mask)
            if spalling_mask.any():
                spalling_dilated = F.max_pool2d(spalling_mask.float().unsqueeze(1), 11, 1, 5)[:, 0] > 0.5

            for i in range(len(sids)):
                sid = sids[i]
                tier = tier_names.get(tier_map.get(sid, -1), "Unknown")

                tp_i = tp_map[i].sum().item()
                fp_i = fp_map[i].sum().item()
                fn_i = fn_map[i].sum().item()

                overall["tp"] += tp_i
                overall["fp"] += fp_i
                overall["fn"] += fn_i
                if tier in by_tier:
                    by_tier[tier]["tp"] += tp_i
                    by_tier[tier]["fp"] += fp_i
                    by_tier[tier]["fn"] += fn_i

                # FN characterization
                fn_pixels = fn_map[i]
                w_fn = wmap[i][fn_pixels]
                fn_thin += (w_fn <= 4).sum().item()
                fn_medium += ((w_fn > 4) & (w_fn <= 10)).sum().item()
                fn_wide += (w_fn > 10).sum().item()

                fn_on_bdry = (fn_pixels & crack_bdry[i]).sum().item()
                fn_on_int = (fn_pixels & crack_interior[i]).sum().item()
                fn_boundary += fn_on_bdry
                fn_interior += fn_on_int

                # FP characterization
                fp_pixels = fp_map[i]
                fp_near_cr = (fp_pixels & crack_dilated[i]).sum().item()
                fp_near_sp = (fp_pixels & spalling_dilated[i]).sum().item()
                fp_iso = (fp_pixels & ~crack_dilated[i] & ~spalling_dilated[i]).sum().item()
                fp_near_crack += fp_near_cr
                fp_near_spalling += fp_near_sp
                fp_isolated += fp_iso

                # Width-binned recall
                for (lo, hi) in width_bins:
                    bin_mask = gt_crack[i] & (wmap[i] > lo) & (wmap[i] <= hi)
                    n_total = bin_mask.sum().item()
                    n_tp = (bin_mask & pred_crack[i]).sum().item()
                    width_tp[(lo, hi)] += n_tp
                    width_total[(lo, hi)] += n_total

                # Per-sample
                prec = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0
                rec = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
                iou = tp_i / (tp_i + fp_i + fn_i) if (tp_i + fp_i + fn_i) > 0 else 0
                per_sample.append({
                    "file": sid, "tier": tier,
                    "tp": tp_i, "fp": fp_i, "fn": fn_i,
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "iou": round(iou, 4),
                })

    # Compute summaries
    def pr(d):
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        return p, r, iou

    print("\n" + "=" * 70)
    print("CRACK FN vs FP ANALYSIS")
    print("=" * 70)

    p, r, iou = pr(overall)
    print(f"\n--- Overall Crack ---")
    print(f"  Precision: {p:.4f}  (1-Precision = FP rate = {1-p:.4f})")
    print(f"  Recall:    {r:.4f}  (1-Recall = FN rate = {1-r:.4f})")
    print(f"  IoU:       {iou:.4f}")
    print(f"  TP={overall['tp']:,}  FP={overall['fp']:,}  FN={overall['fn']:,}")
    fn_ratio = overall["fn"] / (overall["fp"] + overall["fn"]) if (overall["fp"] + overall["fn"]) > 0 else 0
    print(f"  FN/(FN+FP) = {fn_ratio:.2%}  →  {'FN-dominated' if fn_ratio > 0.6 else 'FP-dominated' if fn_ratio < 0.4 else 'balanced'}")

    print(f"\n--- By Tier ---")
    for tier in ["Easy", "Medium", "Hard"]:
        p, r, iou = pr(by_tier[tier])
        d = by_tier[tier]
        fn_r = d["fn"] / (d["fp"] + d["fn"]) if (d["fp"] + d["fn"]) > 0 else 0
        print(f"  {tier:>6s}: P={p:.4f}  R={r:.4f}  IoU={iou:.4f}  "
              f"FP={d['fp']:>8,}  FN={d['fn']:>8,}  FN/(FN+FP)={fn_r:.1%}")

    print(f"\n--- FN Characterization (where are missed crack pixels?) ---")
    total_fn = fn_thin + fn_medium + fn_wide
    if total_fn > 0:
        print(f"  By width:")
        print(f"    Thin (<=4px):    {fn_thin:>10,}  ({fn_thin/total_fn:.1%})")
        print(f"    Medium (5-10px): {fn_medium:>10,}  ({fn_medium/total_fn:.1%})")
        print(f"    Wide (>10px):    {fn_wide:>10,}  ({fn_wide/total_fn:.1%})")
    total_fn2 = fn_boundary + fn_interior
    if total_fn2 > 0:
        print(f"  By location:")
        print(f"    Boundary strip:  {fn_boundary:>10,}  ({fn_boundary/total_fn2:.1%})")
        print(f"    Interior:        {fn_interior:>10,}  ({fn_interior/total_fn2:.1%})")

    print(f"\n--- FP Characterization (where are false crack pixels?) ---")
    total_fp = fp_near_crack + fp_near_spalling + fp_isolated
    if total_fp > 0:
        print(f"  Near real crack (<=5px): {fp_near_crack:>10,}  ({fp_near_crack/total_fp:.1%})")
        print(f"  Near spalling (<=5px):   {fp_near_spalling:>10,}  ({fp_near_spalling/total_fp:.1%})")
        print(f"  Isolated (halluc.):      {fp_isolated:>10,}  ({fp_isolated/total_fp:.1%})")

    print(f"\n--- Recall by Crack Width ---")
    for (lo, hi) in width_bins:
        t = width_total[(lo, hi)]
        tp = width_tp[(lo, hi)]
        rec = tp / t if t > 0 else 0
        label = f"({lo},{hi}]px" if hi < 999 else f">{lo}px"
        bar = "#" * int(rec * 40)
        print(f"  {label:>10s}: recall={rec:.4f}  n={t:>10,}  {bar}")

    # Per-sample worst
    per_sample.sort(key=lambda x: x["recall"])
    print(f"\n--- 10 Worst Recall Samples ---")
    for s in per_sample[:10]:
        print(f"  {s['file']:30s} tier={s['tier']:6s} "
              f"P={s['precision']:.3f} R={s['recall']:.3f} IoU={s['iou']:.3f} "
              f"FP={s['fp']:>6,} FN={s['fn']:>6,}")

    per_sample_prec = sorted(per_sample, key=lambda x: x["precision"])
    print(f"\n--- 10 Worst Precision Samples ---")
    for s in per_sample_prec[:10]:
        print(f"  {s['file']:30s} tier={s['tier']:6s} "
              f"P={s['precision']:.3f} R={s['recall']:.3f} IoU={s['iou']:.3f} "
              f"FP={s['fp']:>6,} FN={s['fn']:>6,}")

    # Save
    results = {
        "overall": {
            "precision": round(pr(overall)[0], 4),
            "recall": round(pr(overall)[1], 4),
            "iou": round(pr(overall)[2], 4),
            **overall,
        },
        "by_tier": {},
        "fn_by_width": {
            "thin_le4": fn_thin, "medium_5to10": fn_medium, "wide_gt10": fn_wide
        },
        "fn_by_location": {
            "boundary": fn_boundary, "interior": fn_interior
        },
        "fp_by_type": {
            "near_crack": fp_near_crack,
            "near_spalling": fp_near_spalling,
            "isolated": fp_isolated
        },
        "recall_by_width": {},
        "per_sample": per_sample,
    }
    for tier in ["Easy", "Medium", "Hard"]:
        p, r, iou = pr(by_tier[tier])
        results["by_tier"][tier] = {
            "precision": round(p, 4), "recall": round(r, 4), "iou": round(iou, 4),
            **by_tier[tier]
        }
    for (lo, hi) in width_bins:
        t = width_total[(lo, hi)]
        tp = width_tp[(lo, hi)]
        label = f"{lo}-{hi}" if hi < 999 else f"gt{lo}"
        results["recall_by_width"][label] = {
            "recall": round(tp / t, 4) if t > 0 else 0,
            "tp": tp, "total": t
        }

    out_dir = Path("results/dgacl")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "crack_fnfp_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[fnfp] saved to {out_path}")


if __name__ == "__main__":
    main()

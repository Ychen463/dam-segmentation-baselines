"""Per-sample IoU analysis to identify hard failures.

Computes per-sample Crack IoU and overall IoU, identifies worst-performing
samples, and characterizes their properties (tier, crack area, width, etc.).

Usage:
    python scripts/hard_sample_analysis.py
    python scripts/hard_sample_analysis.py --run fp1_tversky_precision
    python scripts/hard_sample_analysis.py --top 20
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from full_method import config as C
from full_method.eval_tta import load_model, plain_predict
from full_method.train import build_val_loader


def per_sample_iou(pred, gt, num_classes=3):
    """Compute per-class IoU for a single sample."""
    ious = {}
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (gt == c)
        inter = (pred_c & gt_c).sum().item()
        union = (pred_c | gt_c).sum().item()
        ious[c] = inter / union if union > 0 else float('nan')
    return ious


def crack_width_stats(gt_mask, crack_class=1):
    """Compute crack width statistics from GT mask using distance transform."""
    from scipy import ndimage
    crack = (gt_mask == crack_class).astype(np.uint8)
    if crack.sum() == 0:
        return {"area": 0, "mean_width": 0, "min_width": 0, "max_width": 0}

    # Distance transform gives half-width at each crack pixel
    dt = ndimage.distance_transform_edt(crack)
    # Approximate width = 2 * distance at skeleton pixels
    from skimage.morphology import skeletonize
    skel = skeletonize(crack > 0)
    if skel.sum() > 0:
        widths = 2 * dt[skel]
        return {
            "area": int(crack.sum()),
            "mean_width": float(np.mean(widths)),
            "min_width": float(np.min(widths)),
            "max_width": float(np.max(widths)),
        }
    return {"area": int(crack.sum()), "mean_width": 0, "min_width": 0, "max_width": 0}


def compute_fp_fn(pred, gt, crack_class=1):
    """Compute FP/FN pixel counts for crack."""
    gt_c = (gt == crack_class)
    pred_c = (pred == crack_class)
    tp = (gt_c & pred_c).sum().item()
    fp = (~gt_c & pred_c).sum().item()
    fn = (gt_c & ~pred_c).sum().item()
    return {"tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="dscformer_srl_G1",
                        help="Run directory name")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of worst samples to analyze in detail")
    parser.add_argument("--split", default="test",
                        choices=["test", "val", "train"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = C.RUNS_DIR / args.run
    if not run_dir.exists():
        print(f"[ERROR] {run_dir} not found")
        sys.exit(1)

    print(f"[hard-sample] Loading model from {run_dir}")
    model = load_model(run_dir, device)

    cfg = C.RunCfg()
    cfg.batch_size = 1
    with open(C.SPLIT_FILES[args.split]) as f:
        files = [line.strip() for line in f if line.strip()]

    loader = build_val_loader(files, cfg, device)
    use_amp = (device == "cuda")

    samples = []
    tier_stats = defaultdict(lambda: {"crack_ious": [], "spalling_ious": [], "miou_fgs": []})

    print(f"[hard-sample] Evaluating {len(files)} {args.split} samples...")
    for i, batch in enumerate(loader):
        imgs = batch["image"].to(device, non_blocking=True).float()
        masks = batch["mask"].to(device, non_blocking=True).long()
        target_size = masks.shape[-2:]

        logits = plain_predict(model, imgs, target_size, device, use_amp=use_amp)
        pred = logits.argmax(dim=1).cpu()
        gt = masks.cpu()

        file_id = files[i]
        tier = file_id.split("/")[0] if "/" in file_id else "Unknown"

        ious = per_sample_iou(pred[0], gt[0])
        crack_iou = ious.get(1, float('nan'))
        spalling_iou = ious.get(2, float('nan'))

        # mIoU_fg (only over classes present in GT)
        fg_ious = []
        if (gt[0] == 1).any():
            fg_ious.append(crack_iou)
        if (gt[0] == 2).any():
            fg_ious.append(spalling_iou)
        miou_fg = np.nanmean(fg_ious) if fg_ious else float('nan')

        fpfn = compute_fp_fn(pred[0], gt[0])

        gt_np = gt[0].numpy()
        has_crack = (gt_np == 1).any()
        has_spalling = (gt_np == 2).any()
        crack_area = int((gt_np == 1).sum())
        spalling_area = int((gt_np == 2).sum())

        sample_info = {
            "file": file_id,
            "tier": tier,
            "crack_iou": round(crack_iou, 4) if not np.isnan(crack_iou) else None,
            "spalling_iou": round(spalling_iou, 4) if not np.isnan(spalling_iou) else None,
            "miou_fg": round(miou_fg, 4) if not np.isnan(miou_fg) else None,
            "has_crack": has_crack,
            "has_spalling": has_spalling,
            "crack_area": crack_area,
            "spalling_area": spalling_area,
            "tp": fpfn["tp"],
            "fp": fpfn["fp"],
            "fn": fpfn["fn"],
        }
        samples.append(sample_info)

        # Tier aggregation
        if not np.isnan(crack_iou) and has_crack:
            tier_stats[tier]["crack_ious"].append(crack_iou)
        if not np.isnan(spalling_iou) and has_spalling:
            tier_stats[tier]["spalling_ious"].append(spalling_iou)
        if not np.isnan(miou_fg):
            tier_stats[tier]["miou_fgs"].append(miou_fg)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}]")

    # Sort by worst crack IoU (only samples that have crack)
    crack_samples = [s for s in samples if s["has_crack"]]
    crack_samples.sort(key=lambda s: s["crack_iou"] if s["crack_iou"] is not None else 999)

    # Compute width stats for worst samples
    print(f"\n[hard-sample] Computing width stats for top-{args.top} worst samples...")
    from full_method.dataset import FullMethodDataset
    from baseline_unet.dataset import build_transforms
    ds = FullMethodDataset(C.DATA_ROOT,
                           [{"id": s["file"], "rel": s["file"], "tier": 0, "has_spalling": False}
                            for s in crack_samples[:args.top]],
                           build_transforms(cfg.img_size, train=False))

    for idx in range(min(args.top, len(crack_samples))):
        sample_data = ds[idx]
        gt_np = sample_data["mask"].numpy()
        ws = crack_width_stats(gt_np)
        crack_samples[idx]["width_stats"] = ws

    # Print tier summary
    print(f"\n{'='*70}")
    print(f"Per-Tier Summary ({args.run}, {args.split} set)")
    print(f"{'='*70}")
    print(f"{'Tier':<10s} {'N':>5s} {'CrackIoU':>10s} {'SpIoU':>10s} {'mIoU_fg':>10s}")
    print("-" * 50)
    for tier in ["Easy", "Medium", "Hard"]:
        ts = tier_stats.get(tier, {"crack_ious": [], "spalling_ious": [], "miou_fgs": []})
        n = len(ts["miou_fgs"])
        ci = np.mean(ts["crack_ious"]) * 100 if ts["crack_ious"] else 0
        si = np.mean(ts["spalling_ious"]) * 100 if ts["spalling_ious"] else 0
        mf = np.mean(ts["miou_fgs"]) * 100 if ts["miou_fgs"] else 0
        print(f"{tier:<10s} {n:>5d} {ci:>10.2f} {si:>10.2f} {mf:>10.2f}")

    # Print worst samples
    print(f"\n{'='*70}")
    print(f"Top-{args.top} Worst Crack IoU Samples")
    print(f"{'='*70}")
    print(f"{'File':<35s} {'Tier':<8s} {'CrIoU':>7s} {'CrArea':>7s} {'FP':>7s} {'FN':>7s} {'Width':>6s}")
    print("-" * 80)

    zero_iou_count = 0
    zero_iou_by_tier = defaultdict(int)

    for s in crack_samples[:args.top]:
        ci = s["crack_iou"] * 100 if s["crack_iou"] is not None else 0
        ws = s.get("width_stats", {})
        mw = ws.get("mean_width", 0)
        fname = s["file"]
        if len(fname) > 34:
            fname = "..." + fname[-31:]
        print(f"{fname:<35s} {s['tier']:<8s} {ci:>7.2f} {s['crack_area']:>7d} "
              f"{s['fp']:>7d} {s['fn']:>7d} {mw:>6.1f}")

        if s["crack_iou"] is not None and s["crack_iou"] == 0:
            zero_iou_count += 1
            zero_iou_by_tier[s["tier"]] += 1

    # Summary stats
    all_crack_ious = [s["crack_iou"] for s in crack_samples if s["crack_iou"] is not None]
    low_iou = [x for x in all_crack_ious if x < 0.3]

    print(f"\n{'='*70}")
    print(f"Failure Analysis Summary")
    print(f"{'='*70}")
    print(f"Total samples with crack: {len(crack_samples)}")
    print(f"Samples with CrackIoU=0: {zero_iou_count} ({zero_iou_count/len(crack_samples)*100:.1f}%)")
    for tier in ["Easy", "Medium", "Hard"]:
        if tier in zero_iou_by_tier:
            print(f"  {tier}: {zero_iou_by_tier[tier]}")
    print(f"Samples with CrackIoU<0.3: {len(low_iou)} ({len(low_iou)/len(crack_samples)*100:.1f}%)")
    print(f"Mean CrackIoU: {np.mean(all_crack_ious)*100:.2f}%")
    print(f"Median CrackIoU: {np.median(all_crack_ious)*100:.2f}%")
    print(f"Std CrackIoU: {np.std(all_crack_ious)*100:.2f}%")

    # IoU distribution bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist, _ = np.histogram(all_crack_ious, bins=bins)
    print(f"\nCrack IoU Distribution:")
    for i in range(len(bins) - 1):
        bar = "#" * hist[i]
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:>4d} {bar}")

    # Save results
    out_path = Path(f"results/dgacl/hard_sample_analysis_{args.run}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "run": args.run,
        "split": args.split,
        "tier_summary": {},
        "worst_samples": crack_samples[:args.top],
        "iou_distribution": {
            f"[{bins[i]:.1f}-{bins[i+1]:.1f})": int(hist[i])
            for i in range(len(bins) - 1)
        },
        "stats": {
            "total_crack_samples": len(crack_samples),
            "zero_iou_count": zero_iou_count,
            "zero_iou_by_tier": dict(zero_iou_by_tier),
            "low_iou_count": len(low_iou),
            "mean_crack_iou": round(float(np.mean(all_crack_ious)), 4),
            "median_crack_iou": round(float(np.median(all_crack_ious)), 4),
        },
    }
    for tier in ["Easy", "Medium", "Hard"]:
        ts = tier_stats.get(tier, {"crack_ious": [], "spalling_ious": [], "miou_fgs": []})
        save_data["tier_summary"][tier] = {
            "n": len(ts["miou_fgs"]),
            "mean_crack_iou": round(float(np.mean(ts["crack_ious"])), 4) if ts["crack_ious"] else None,
            "mean_spalling_iou": round(float(np.mean(ts["spalling_ious"])), 4) if ts["spalling_ious"] else None,
            "mean_miou_fg": round(float(np.mean(ts["miou_fgs"])), 4) if ts["miou_fgs"] else None,
        }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[hard-sample] Saved to {out_path}")


if __name__ == "__main__":
    main()

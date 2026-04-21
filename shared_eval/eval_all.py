"""Unified evaluation script for all baselines.

Usage examples::

    # Single model, overall test metrics
    python -m shared_eval.eval_all --model deeplabv3p_r50_512 --split test

    # Single model, per-tier breakdown
    python -m shared_eval.eval_all --model deeplabv3p_r50_512 --split test --per-tier

    # All registered models
    python -m shared_eval.eval_all --all-models --split test --per-tier

    # Per-image CSV for statistical testing
    python -m shared_eval.eval_all --all-models --split test --per-image

    # Custom output directory
    python -m shared_eval.eval_all --model unet_r34_320 --split test --output-dir results/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_unet import config as C
from baseline_unet.dataset import (
    DamSegmentDataset,
    build_transforms,
    read_split_file,
)
from baseline_unet.splits import SPLIT_FILES

from .metrics_full import SegMetricsFull, _connectivity_ratio
from .cldice import cldice_single
from .model_registry import get as get_entry, list_models, load_model

BF1_TOLERANCE_PX = 2
BATCH_SIZE = 4
NUM_WORKERS = 2


# ---------------------------------------------------------------------------
# Morphological post-processing (N4)
# ---------------------------------------------------------------------------

def postprocess_prediction(pred: np.ndarray,
                           min_crack_area: int = 50,
                           min_spalling_area: int = 200,
                           crack_close_kernel: int = 5) -> np.ndarray:
    """Morphological cleanup of argmax prediction (H, W) with classes 0/1/2.

    Steps:
      1. Close small gaps in crack (bridge fractures) with elliptical kernel.
      2. Remove small crack connected components (< min_crack_area px).
      3. Remove small spalling connected components (< min_spalling_area px).
      4. Reassemble: crack (1) takes priority over spalling (2) at overlaps.
    """
    import cv2 as _cv2

    crack_mask = (pred == 1).astype(np.uint8)
    spalling_mask = (pred == 2).astype(np.uint8)

    # 1. Morphological closing on crack to bridge fractures
    kernel = _cv2.getStructuringElement(
        _cv2.MORPH_ELLIPSE, (crack_close_kernel, crack_close_kernel))
    crack_mask = _cv2.morphologyEx(crack_mask, _cv2.MORPH_CLOSE, kernel)

    # 2. Remove small crack components
    n_labels, labels, stats, _ = _cv2.connectedComponentsWithStats(
        crack_mask, connectivity=8)
    for i in range(1, n_labels):
        if stats[i, _cv2.CC_STAT_AREA] < min_crack_area:
            crack_mask[labels == i] = 0

    # 3. Remove small spalling components
    n_labels, labels, stats, _ = _cv2.connectedComponentsWithStats(
        spalling_mask, connectivity=8)
    for i in range(1, n_labels):
        if stats[i, _cv2.CC_STAT_AREA] < min_spalling_area:
            spalling_mask[labels == i] = 0

    # 4. Reassemble: background default, spalling first, then crack overrides
    out = np.zeros_like(pred)
    out[spalling_mask > 0] = 2
    out[crack_mask > 0] = 1
    return out


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


def _apply_crf_batch(images: torch.Tensor, logits: torch.Tensor) -> np.ndarray:
    """Apply CRF refinement to a batch. Returns (B, H, W) int predictions."""
    from .postprocess import crf_refine_from_logits
    imgs_np = (images.cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
    logits_np = logits.cpu().numpy()
    results = []
    for i in range(logits_np.shape[0]):
        refined = crf_refine_from_logits(imgs_np[i], logits_np[i],
                                         n_classes=C.NUM_CLASSES)
        results.append(refined)
    return np.stack(results)


def _postprocess_batch(logits: torch.Tensor, images: torch.Tensor,
                       postprocess: bool, use_crf: bool,
                       device: str):
    """Apply optional morphological PP and/or CRF. Returns fake logits tensor or None."""
    if not postprocess and not use_crf:
        return None
    if use_crf:
        pred_np = _apply_crf_batch(images, logits)
        if postprocess:
            for i in range(pred_np.shape[0]):
                pred_np[i] = postprocess_prediction(pred_np[i])
    else:
        pred_np = logits.argmax(dim=1).cpu().numpy()
        for i in range(pred_np.shape[0]):
            pred_np[i] = postprocess_prediction(pred_np[i])
    pred_t = torch.from_numpy(pred_np).to(device)
    fake_logits = torch.zeros_like(logits)
    fake_logits.scatter_(1, pred_t.unsqueeze(1), 10.0)
    return fake_logits


@torch.no_grad()
def _evaluate(model, loader: DataLoader, device: str,
              postprocess: bool = False, use_crf: bool = False) -> Dict[str, float]:
    metrics = SegMetricsFull(C.NUM_CLASSES, BF1_TOLERANCE_PX)
    metrics.reset()
    for images, masks, _rels in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        fake = _postprocess_batch(logits, images, postprocess, use_crf, device)
        metrics.update(fake if fake is not None else logits, masks)
    return metrics.compute()


def _forward_maybe_tta(model, images: torch.Tensor, use_tta: bool) -> torch.Tensor:
    """Forward pass with optional test-time augmentation (4x flip ensemble)."""
    logits = model(images)
    if not use_tta:
        return logits
    logits = logits + model(images.flip(-1)).flip(-1)
    logits = logits + model(images.flip(-2)).flip(-2)
    logits = logits + model(images.flip(-1, -2)).flip(-1, -2)
    return logits / 4.0


@torch.no_grad()
def _evaluate_tta(model, loader: DataLoader, device: str,
                  postprocess: bool = False, use_crf: bool = False) -> Dict[str, float]:
    """Evaluate with test-time augmentation (4x flip)."""
    metrics = SegMetricsFull(C.NUM_CLASSES, BF1_TOLERANCE_PX)
    metrics.reset()
    for images, masks, _rels in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = _forward_maybe_tta(model, images, use_tta=True)
        fake = _postprocess_batch(logits, images, postprocess, use_crf, device)
        metrics.update(fake if fake is not None else logits, masks)
    return metrics.compute()


@torch.no_grad()
def _evaluate_ensemble(
    model_names: List[str], loader: DataLoader, device: str,
    use_tta: bool = False,
) -> Dict[str, float]:
    """Evaluate an ensemble of models (average logits)."""
    models = [load_model(n, device=device) for n in model_names]
    metrics = SegMetricsFull(C.NUM_CLASSES, BF1_TOLERANCE_PX)
    metrics.reset()
    for images, masks, _rels in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits_sum = None
        for m in models:
            logits = _forward_maybe_tta(m, images, use_tta)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits_sum = logits_sum / len(models)
        metrics.update(logits_sum, masks)
    return metrics.compute()


# ---------------------------------------------------------------------------
# Per-image evaluation (for statistical testing)
# ---------------------------------------------------------------------------

# Lazy import to avoid hard dep on boundary_f1 at module level
_bf1_fn = None

def _get_bf1_fn():
    global _bf1_fn
    if _bf1_fn is None:
        from baseline_deeplab.boundary_f1 import boundary_f1_single
        _bf1_fn = boundary_f1_single
    return _bf1_fn


def _per_image_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute per-image metrics for a single (H, W) prediction/ground-truth pair."""
    bf1_fn = _get_bf1_fn()
    row: Dict[str, float] = {}

    for name, cid in (("crack", 1), ("spalling", 2)):
        pred_mask = (pred == cid)
        gt_mask = (gt == cid)
        # IoU
        inter = float((pred_mask & gt_mask).sum())
        union = float((pred_mask | gt_mask).sum())
        row[f"IoU_{name}"] = inter / max(union, 1e-9) if union > 0 else float("nan")
        # Dice
        denom = float(pred_mask.sum() + gt_mask.sum())
        row[f"Dice_{name}"] = 2.0 * inter / max(denom, 1e-9) if denom > 0 else float("nan")
        # BF1
        val = bf1_fn(pred, gt, cid, BF1_TOLERANCE_PX)
        row[f"BF1_{name}"] = val if val is not None else float("nan")
        # clDice
        val = cldice_single(pred, gt, cid)
        row[f"clDice_{name}"] = val if val is not None else float("nan")
        # Connectivity ratio
        val = _connectivity_ratio(pred, gt, cid)
        row[f"ConnR_{name}"] = val if val is not None else float("nan")

    return row


@torch.no_grad()
def _evaluate_per_image(
    model, loader: DataLoader, device: str, file_list: List[str],
    use_tta: bool = False, postprocess: bool = False, use_crf: bool = False,
) -> List[Dict[str, object]]:
    """Return a list of dicts, one per image, with per-image metrics + metadata."""
    rows: List[Dict[str, object]] = []
    idx = 0
    for images, masks, rels in loader:
        images = images.to(device, non_blocking=True)
        logits = _forward_maybe_tta(model, images, use_tta)
        if use_crf:
            pred_batch = _apply_crf_batch(images, logits)
            if postprocess:
                for i in range(pred_batch.shape[0]):
                    pred_batch[i] = postprocess_prediction(pred_batch[i])
        else:
            pred_batch = logits.argmax(dim=1).detach().cpu().numpy()
        gt_batch = masks.numpy()
        for b in range(pred_batch.shape[0]):
            pred_img = pred_batch[b]
            if not use_crf and postprocess:
                pred_img = postprocess_prediction(pred_img)
            fname = rels[b] if isinstance(rels, list) else rels[b]
            tier = fname.split("/")[0] if "/" in fname else "unknown"
            row = _per_image_metrics(pred_img, gt_batch[b])
            row["file"] = fname
            row["tier"] = tier
            rows.append(row)
            idx += 1
    return rows


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

def _run_eval(
    model_name: str, split: str, per_tier: bool, device: str,
    use_tta: bool = False, postprocess: bool = False, use_crf: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run evaluation, return dict with 'overall' and optionally tier keys."""
    entry = get_entry(model_name)
    img_size = entry.img_size

    print(f"[eval] Loading model: {model_name}")
    model = load_model(model_name, device=device)

    transform = build_transforms(img_size, train=False)
    eval_fn = _evaluate_tta if use_tta else _evaluate

    # Read split files
    split_path = SPLIT_FILES[split]
    all_files = read_split_file(split_path)
    pp_tag = " (PP)" if postprocess else ""
    crf_tag = " (CRF)" if use_crf else ""
    print(f"[eval] Split '{split}': {len(all_files)} images"
          f"{' (TTA)' if use_tta else ''}{pp_tag}{crf_tag}")

    results: Dict[str, Dict[str, float]] = {}

    # Overall
    ds = DamSegmentDataset(C.DATA_ROOT, all_files, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    print(f"[eval] Running overall evaluation ...")
    results["overall"] = eval_fn(model, loader, device,
                                 postprocess=postprocess, use_crf=use_crf)

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
            results[tier] = eval_fn(model, loader_t, device,
                                    postprocess=postprocess, use_crf=use_crf)

    return results


def _run_eval_per_image(
    model_name: str, split: str, device: str,
    use_tta: bool = False, postprocess: bool = False, use_crf: bool = False,
) -> List[Dict[str, object]]:
    """Run per-image evaluation, return list of per-image metric dicts."""
    entry = get_entry(model_name)
    img_size = entry.img_size

    print(f"[eval] Loading model: {model_name}")
    model = load_model(model_name, device=device)

    transform = build_transforms(img_size, train=False)
    split_path = SPLIT_FILES[split]
    all_files = read_split_file(split_path)
    print(f"[eval] Split '{split}': {len(all_files)} images (per-image mode)")

    ds = DamSegmentDataset(C.DATA_ROOT, all_files, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    return _evaluate_per_image(model, loader, device, all_files,
                               use_tta=use_tta, postprocess=postprocess,
                               use_crf=use_crf)


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
    group.add_argument("--ensemble", type=str, default=None,
                       help="Comma-separated model names for ensemble evaluation")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--per-tier", action="store_true",
                        help="Also report per-difficulty-tier metrics")
    parser.add_argument("--per-image", action="store_true",
                        help="Save per-image metrics CSV (for statistical tests)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (4x flip)")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply morphological post-processing to predictions")
    parser.add_argument("--crf", action="store_true",
                        help="Apply DenseCRF post-processing (requires pydensecrf)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for JSON/CSV output (default: results/)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (cuda/mps/cpu)")
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"[eval] Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensemble mode
    if args.ensemble:
        model_names = [n.strip() for n in args.ensemble.split(",")]
        ensemble_tag = "+".join(model_names)
        print(f"\n{'='*60}")
        print(f"  Ensemble: {ensemble_tag}")
        print(f"{'='*60}")

        # Build loader from first model's img_size
        entry = get_entry(model_names[0])
        transform = build_transforms(entry.img_size, train=False)
        split_path = SPLIT_FILES[args.split]
        all_files = read_split_file(split_path)
        ds = DamSegmentDataset(C.DATA_ROOT, all_files, transform=transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
        print(f"[eval] Split '{args.split}': {len(all_files)} images"
              f"{' (TTA)' if args.tta else ''}")

        results = {"overall": _evaluate_ensemble(model_names, loader, device,
                                                  use_tta=args.tta)}
        print(_format_results(results))
        suffix = "_tta" if args.tta else ""
        out_path = output_dir / f"ensemble_{ensemble_tag}_{args.split}{suffix}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[eval] Saved: {out_path}")
        return

    models = list_models() if args.all_models else [args.model]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        # Aggregated metrics (always)
        results = _run_eval(model_name, args.split, args.per_tier, device,
                            use_tta=args.tta, postprocess=args.postprocess,
                            use_crf=args.crf)
        print(_format_results(results))
        suffix = "_tta" if args.tta else ""
        if args.crf:
            suffix += "_crf"
        if args.postprocess:
            suffix += "_pp"
        out_path = output_dir / f"{model_name}_{args.split}{suffix}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[eval] Saved: {out_path}")

        # Per-image CSV (optional)
        if args.per_image:
            rows = _run_eval_per_image(model_name, args.split, device,
                                       use_tta=args.tta,
                                       postprocess=args.postprocess,
                                       use_crf=args.crf)
            csv_path = output_dir / f"{model_name}_{args.split}{suffix}_per_image.csv"
            if rows:
                fieldnames = ["file", "tier"] + [
                    k for k in rows[0] if k not in ("file", "tier")
                ]
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            print(f"[eval] Per-image CSV: {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

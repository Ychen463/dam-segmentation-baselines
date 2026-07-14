"""Cross-dataset evaluation: evaluate DamSegment-trained models on S2DS.

Zero-shot transfer: no fine-tuning, just load the trained model and run
inference on the S2DS test images (remapped to our 3-class scheme).

Usage::

    # Prepare S2DS first:
    python scripts/prepare_s2ds.py --raw-dir /path/to/s2ds/

    # Evaluate key models:
    python scripts/eval_cross_dataset.py

    # Evaluate a specific model:
    python scripts/eval_cross_dataset.py --model dscformer_srl_G1

    # With TTA:
    python scripts/eval_cross_dataset.py --tta

    # Per-image CSV for statistical testing:
    python scripts/eval_cross_dataset.py --per-image
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_unet import config as C
from baseline_unet.dataset import build_transforms
from shared_eval.metrics_full import SegMetricsFull
from shared_eval.cldice import cldice_single
from shared_eval.metrics_full import _connectivity_ratio
from shared_eval.model_registry import load_model

CODES_DIR = Path(__file__).resolve().parent.parent
S2DS_DIR = CODES_DIR / "Dataset" / "S2DS"
BF1_TOLERANCE_PX = 2

# Models to evaluate by default
DEFAULT_MODELS = [
    "segformer_b2_plain_512",
    "dscformer_srl_G1",
    "dual_kd_classaware_DKD2",
    "mask2former_swin_small_512",
]


class S2DSDataset(Dataset):
    """S2DS dataset loader (pre-processed 3-class masks at 512x512)."""

    def __init__(self, data_dir: Path, file_list: List[str], transform=None):
        self.data_dir = Path(data_dir)
        self.files = file_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        stem = self.files[idx]

        # Read image
        img_path = self.data_dir / "images" / f"{stem}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask (single-channel index)
        mask_path = self.data_dir / "masks" / f"{stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read: {mask_path}")
        mask = mask.astype(np.int64)

        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            image_t = out["image"]
            mask_t = out["mask"]
            if not torch.is_tensor(mask_t):
                mask_t = torch.from_numpy(mask_t)
            mask_t = mask_t.long()
            return image_t, mask_t, stem

        return img, mask, stem


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _forward_maybe_tta(model, images: torch.Tensor, use_tta: bool) -> torch.Tensor:
    """Forward with optional TTA (multi-scale + flip, 9 views)."""
    if not use_tta:
        return model(images)

    B, Ch, H, W = images.shape
    scales = (0.75, 1.0, 1.25)
    logits_sum = None
    for s in scales:
        sH, sW = int(H * s), int(W * s)
        x = F.interpolate(images, (sH, sW), mode="bilinear", align_corners=False)
        sl = model(x)
        # Add horizontal flip
        sl = sl + model(x.flip(-1)).flip(-1)
        # Add vertical flip
        sl = sl + model(x.flip(-2)).flip(-2)
        sl = sl / 3.0
        sl = F.interpolate(sl, (H, W), mode="bilinear", align_corners=False)
        logits_sum = sl if logits_sum is None else logits_sum + sl
    return logits_sum / len(scales)


def _per_image_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Per-image metrics for a single (H, W) pair."""
    from baseline_deeplab.boundary_f1 import boundary_f1_single
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
        val = boundary_f1_single(pred, gt, cid, BF1_TOLERANCE_PX)
        row[f"BF1_{name}"] = val if val is not None else float("nan")
        # clDice
        val = cldice_single(pred, gt, cid)
        row[f"clDice_{name}"] = val if val is not None else float("nan")
        # ConnR
        val = _connectivity_ratio(pred, gt, cid)
        row[f"ConnR_{name}"] = val if val is not None else float("nan")
    return row


@torch.no_grad()
def evaluate_cross_dataset(
    model_name: str, data_dir: Path, device: str,
    use_tta: bool = False, per_image: bool = False,
) -> Dict:
    """Evaluate a model on S2DS."""
    # Load file list
    list_path = data_dir / "test_files.txt"
    if not list_path.exists():
        raise FileNotFoundError(
            f"File list not found: {list_path}\n"
            f"Run: python scripts/prepare_s2ds.py --raw-dir /path/to/s2ds/"
        )
    with open(list_path) as f:
        file_list = [ln.strip() for ln in f if ln.strip()]

    print(f"[cross-eval] Model: {model_name}")
    print(f"[cross-eval] S2DS images: {len(file_list)}")
    print(f"[cross-eval] TTA: {use_tta}")

    # Load model
    model = load_model(model_name, device=device)

    # Build dataset
    transform = build_transforms(512, train=False)
    dataset = S2DSDataset(data_dir, file_list, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        num_workers=2, pin_memory=(device == "cuda"))

    # Aggregate metrics
    metrics = SegMetricsFull(C.NUM_CLASSES, BF1_TOLERANCE_PX)
    metrics.reset()

    per_image_rows = []

    for images, masks, stems in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = _forward_maybe_tta(model, images, use_tta)
        metrics.update(logits, masks)

        if per_image:
            pred_batch = logits.argmax(dim=1).cpu().numpy()
            gt_batch = masks.cpu().numpy()
            for b in range(pred_batch.shape[0]):
                row = _per_image_metrics(pred_batch[b], gt_batch[b])
                row["file"] = stems[b]
                per_image_rows.append(row)

    results = metrics.compute()
    return {"aggregated": results, "per_image": per_image_rows}


def format_results(model_name: str, results: Dict[str, float]) -> str:
    """Pretty-print results."""
    r = results
    lines = [
        f"\n{'='*60}",
        f"  {model_name}  →  S2DS (cross-dataset)",
        f"{'='*60}",
        f"  IoU     bg={r['IoU_background']:.4f}  crack={r['IoU_crack']:.4f}"
        f"  spalling={r['IoU_spalling']:.4f}  | mIoU_fg={r['mIoU_fg']:.4f}",
        f"  Dice    crack={r['Dice_crack']:.4f}  spalling={r['Dice_spalling']:.4f}",
        f"  BF1     crack={r['BF1_crack']:.4f}  spalling={r['BF1_spalling']:.4f}"
        f"  | BF1_fg={r['BF1_fg_mean']:.4f}",
        f"  clDice  crack={r['clDice_crack']:.4f}  spalling={r['clDice_spalling']:.4f}"
        f"  | clDice_fg={r['clDice_fg_mean']:.4f}",
        f"  ConnR   crack={r['ConnR_crack']:.4f}  spalling={r['ConnR_spalling']:.4f}"
        f"  | ConnR_fg={r['ConnR_fg_mean']:.4f}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation on S2DS",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model name (default: evaluate all key models)")
    parser.add_argument("--data-dir", type=str, default=str(S2DS_DIR))
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--per-image", action="store_true",
                        help="Save per-image CSV for statistical tests")
    parser.add_argument("--output-dir", type=str, default="results/cross_dataset/")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or _pick_device()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else DEFAULT_MODELS

    all_results = {}

    for model_name in models:
        try:
            out = evaluate_cross_dataset(
                model_name, data_dir, device,
                use_tta=args.tta, per_image=args.per_image,
            )
        except (KeyError, FileNotFoundError) as e:
            print(f"[WARN] Skipping {model_name}: {e}")
            continue

        agg = out["aggregated"]
        print(format_results(model_name, agg))
        all_results[model_name] = agg

        # Save per-model JSON
        suffix = "_tta" if args.tta else ""
        json_path = output_dir / f"s2ds_{model_name}{suffix}.json"
        with open(json_path, "w") as f:
            json.dump({"s2ds": agg}, f, indent=2)

        # Save per-image CSV
        if args.per_image and out["per_image"]:
            csv_path = output_dir / f"s2ds_{model_name}{suffix}_per_image.csv"
            rows = out["per_image"]
            fieldnames = ["file"] + [k for k in rows[0] if k != "file"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  [saved] {csv_path}")

    # Print comparison table
    if len(all_results) > 1:
        print(f"\n\n{'='*70}")
        print("  Cross-Dataset Summary: DamSegment → S2DS")
        print(f"{'='*70}")
        header = f"  {'Model':<35} {'mIoU_fg':>8} {'IoU_cr':>8} {'IoU_sp':>8} {'BF1_fg':>8} {'ConnR_fg':>8}"
        print(header)
        print(f"  {'-'*67}")
        for name, r in all_results.items():
            print(f"  {name:<35} {r['mIoU_fg']*100:>7.1f}% {r['IoU_crack']*100:>7.1f}%"
                  f" {r['IoU_spalling']*100:>7.1f}% {r['BF1_fg_mean']*100:>7.1f}%"
                  f" {r['ConnR_fg_mean']*100:>7.1f}%")

    # Save combined summary
    summary_path = output_dir / f"s2ds_summary{'_tta' if args.tta else ''}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[cross-eval] Summary saved: {summary_path}")


if __name__ == "__main__":
    main()

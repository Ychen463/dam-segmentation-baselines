"""Evaluate models retrained on the balanced group split.

One-time final evaluation — all design decisions were locked before this split
was constructed. Results from this script are the primary reported numbers.

Usage::

    # Evaluate all retrained models on balanced test set
    python scripts/eval_balanced_split.py

    # Also compare with original-split results
    python scripts/eval_balanced_split.py --compare

    # Custom test split path
    python scripts/eval_balanced_split.py --test-split path/to/test.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_unet import config as C
from baseline_unet.dataset import (
    build_transforms,
    decode_mask,
    has_spalling_from_mask,
    image_path,
    mask_path,
    read_mask_rgb,
)
from shared_eval.metrics_full import SegMetricsFull
from shared_eval.model_registry import load_model, list_models

CODES_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = CODES_DIR / "baseline_unet" / "splits"
BALANCED_SPLIT_DIR = SPLIT_DIR / "balanced_group_split"
BF1_TOLERANCE_PX = 2
SUFFIX = "bgsplit"

# Model registry name -> (package, bgsplit run name)
BGSPLIT_RUN_DIRS = {
    "unet_r34_320":              ("baseline_unet",      f"unet_r34_ce_dice_{SUFFIX}"),
    "deeplabv3p_r50_512":        ("baseline_deeplab",   f"deeplabv3p_r50_512_{SUFFIX}"),
    "segformer_b2_plain_512":    ("baseline_segformer", f"segformer_b2_plain_512_{SUFFIX}"),
    "mask2former_swin_small_512":("baseline_mask2former",f"mask2former_plain_M0_{SUFFIX}"),
    "dscformer_plain_G0":        ("full_method",        f"dscformer_plain_G0_{SUFFIX}"),
    "dscformer_srl_G1":          ("full_method",        f"dscformer_srl_G1_{SUFFIX}"),
    "dual_kd_classaware_DKD2":   ("full_method",        f"dual_kd_classaware_DKD2_{SUFFIX}"),
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DamSegDataset(Dataset):
    def __init__(self, data_root: Path, rels: List[str], img_size: int = 512):
        self.data_root = data_root
        self.rels = rels
        self.transform = build_transforms(img_size, train=False)

    def __len__(self):
        return len(self.rels)

    def __getitem__(self, idx):
        rel = self.rels[idx]
        img = cv2.imread(str(image_path(self.data_root, rel)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = read_mask_rgb(mask_path(self.data_root, rel))
        label, _ = decode_mask(m)
        out = self.transform(image=img, mask=label)
        image_t = out["image"]
        mask_t = out["mask"]
        if not torch.is_tensor(mask_t):
            mask_t = torch.from_numpy(mask_t)
        return image_t, mask_t.long(), rel


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_checkpoint(model_name: str) -> Path:
    if model_name not in BGSPLIT_RUN_DIRS:
        raise KeyError(f"No bgsplit mapping for '{model_name}'")
    pkg, run_name = BGSPLIT_RUN_DIRS[model_name]
    return CODES_DIR / pkg / "runs" / run_name / "best.pt"


def _load_model_from_checkpoint(model_name: str, ckpt_path: Path,
                                device: str) -> nn.Module:
    from shared_eval.model_registry import (
        get as get_entry,
        Mask2FormerLogitsWrapper,
        _build_mask2former,
    )
    entry = get_entry(model_name)

    if "mask2former" in model_name:
        model, processor = _build_mask2former()
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
        wrapper = Mask2FormerLogitsWrapper(
            model, processor, C.NUM_CLASSES, entry.img_size)
        wrapper.to(device).eval()
        return wrapper

    model = entry.build_fn()
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "ema_model" in state:
            model.load_state_dict(state["ema_model"])
        else:
            model.load_state_dict(state["model"])

    if entry.inference_wrapper is not None:
        model = entry.inference_wrapper(model, entry.img_size)

    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model_name: str, test_rels: List[str],
                   data_root: Path, device: str,
                   checkpoint_override: Path = None) -> Dict[str, float]:
    from shared_eval.model_registry import get as get_entry
    entry = get_entry(model_name)

    if checkpoint_override is not None:
        model = _load_model_from_checkpoint(model_name, checkpoint_override, device)
    else:
        model = load_model(model_name, device=device)

    dataset = DamSegDataset(data_root, test_rels, img_size=entry.img_size)
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        num_workers=2, pin_memory=(device == "cuda"))

    metrics = SegMetricsFull(C.NUM_CLASSES, BF1_TOLERANCE_PX)
    metrics.reset()

    for images, masks, rels in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        metrics.update(logits, masks)

    return metrics.compute()


def format_results(model_name: str, results: Dict[str, float]) -> str:
    r = results
    lines = [
        f"\n{'='*60}",
        f"  {model_name}  ->  Balanced-split test",
        f"{'='*60}",
        f"  IoU     bg={r['IoU_background']:.4f}  crack={r['IoU_crack']:.4f}"
        f"  spalling={r['IoU_spalling']:.4f}  | mIoU_fg={r['mIoU_fg']:.4f}",
        f"  Dice    crack={r['Dice_crack']:.4f}  spalling={r['Dice_spalling']:.4f}",
        f"  BF1     crack={r['BF1_crack']:.4f}  spalling={r['BF1_spalling']:.4f}"
        f"  | BF1_fg={r['BF1_fg_mean']:.4f}",
        f"  clDice  crack={r['clDice_crack']:.4f}  spalling={r['clDice_spalling']:.4f}"
        f"  | clDice_fg={r['clDice_fg_mean']:.4f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate models retrained on balanced group split")
    parser.add_argument("--test-split", type=str,
                        default=str(BALANCED_SPLIT_DIR / "test.txt"),
                        help="Path to test split file")
    parser.add_argument("--compare", action="store_true",
                        help="Also evaluate original-split checkpoints for comparison")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to evaluate")
    args = parser.parse_args()

    test_file = Path(args.test_split)
    if not test_file.exists():
        print(f"[ERROR] Test split not found: {test_file}")
        sys.exit(1)

    with open(test_file) as f:
        test_rels = [ln.strip() for ln in f if ln.strip()]

    # Print test set diagnostics
    from collections import Counter
    tiers = Counter(r.split("/")[0] for r in test_rels)
    print(f"[eval] Test set: {len(test_rels)} images")
    print(f"[eval] Tier distribution: {dict(tiers)}")

    data_root = C.DATA_ROOT
    device = _pick_device()

    # Find available retrained models
    model_names = args.models or list(BGSPLIT_RUN_DIRS.keys())
    available = []
    missing = []
    for m in model_names:
        if m in BGSPLIT_RUN_DIRS:
            ckpt = _find_checkpoint(m)
            if ckpt.exists():
                available.append(m)
            else:
                missing.append((m, str(ckpt)))

    if missing:
        print(f"\n[eval] Missing checkpoints ({len(missing)}):")
        for m, p in missing:
            print(f"  {m}: {p}")

    if not available:
        print("[ERROR] No retrained checkpoints found. "
              "Run scripts/run_balanced_split_retrain.sh first.")
        sys.exit(1)

    print(f"\n[eval] Evaluating {len(available)} models on balanced test set\n")

    csv_rows = []

    for model_name in available:
        ckpt = _find_checkpoint(model_name)
        print(f"--- {model_name} [{ckpt.parent.name}] ---")

        results = evaluate_model(
            model_name, test_rels, data_root, device,
            checkpoint_override=ckpt)
        print(format_results(model_name, results))

        row = {"model": model_name}
        for k, v in results.items():
            row[f"bgsplit_{k}"] = f"{v:.4f}"

        # Optionally compare with original-split checkpoint
        if args.compare:
            orig_results = evaluate_model(
                model_name, test_rels, data_root, device)
            for k, v in orig_results.items():
                row[f"origsplit_{k}"] = f"{v:.4f}"

            # Print delta
            keys = ["mIoU_fg", "BF1_crack", "clDice_crack"]
            deltas = []
            for k in keys:
                d = (results[k] - orig_results[k]) * 100
                deltas.append(f"{k}: {d:+.2f}")
            print(f"  Delta vs orig-ckpt: {', '.join(deltas)}")

        csv_rows.append(row)

    # Save CSV
    out_csv = BALANCED_SPLIT_DIR / "balanced_split_results.csv"
    if csv_rows:
        keys = list(csv_rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)
        print(f"\n[eval] Results saved to {out_csv}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  BALANCED GROUP SPLIT — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'mIoU_fg':>8} {'BF1_cr':>8} {'clDice_cr':>8}")
    print(f"  {'-'*56}")
    for row in csv_rows:
        m = row["model"]
        miou = float(row["bgsplit_mIoU_fg"]) * 100
        bf1 = float(row["bgsplit_BF1_crack"]) * 100
        cld = float(row["bgsplit_clDice_crack"]) * 100
        print(f"  {m:<30} {miou:>7.2f}% {bf1:>7.2f}% {cld:>7.2f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

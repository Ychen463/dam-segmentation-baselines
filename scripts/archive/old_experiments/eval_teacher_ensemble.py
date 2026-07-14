#!/usr/bin/env python
"""Evaluate teacher ensemble upper bounds on the DamSegment test set.

Computes metrics for:
  1. Equal-weight logit-space ensemble (T1+T2) / 2
  2. Per-class probability-space ensemble (same weights as DTKD)
  3. Agreement-weighted ensemble (pixel-level)
  4. Individual teachers (for reference)

No training needed — just forward passes through both teachers.

Usage (on RunPod):
    python scripts/eval_teacher_ensemble.py --device cuda
    python scripts/eval_teacher_ensemble.py --device cuda --output results/teacher_ensemble.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_unet import config as C
from baseline_unet.dataset import DamSegmentDataset, build_transforms, read_split_file
from baseline_unet.splits import SPLIT_FILES
from shared_eval.metrics_full import SegMetricsFull

from full_method.model import DSCformerDam, SegFormerWithBoundary
from full_method.sam_model import TopoLoRASAM
from full_method import config as FC

BF1_TOLERANCE_PX = 2
BATCH_SIZE = 4


def load_teacher(checkpoint_path: str, model_type: str, device: str) -> torch.nn.Module:
    """Load a teacher model from checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = (FC.PKG_DIR / ckpt_path).resolve()

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = FC.RunCfg()

    if model_type == "sam_lora":
        model = TopoLoRASAM(
            sam_checkpoint=cfg.sam_checkpoint,
            num_classes=C.NUM_CLASSES,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            fpn_dim=cfg.sam_fpn_dim,
            sam_img_size=cfg.sam_img_size,
        ).to(device)
    elif model_type == "dscformer":
        model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)
    else:
        model = SegFormerWithBoundary(cfg.pretrained).to(device)

    model.load_state_dict(state["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    epoch = state.get("epoch", "?")
    miou = state.get("mIoU_fg", state.get("best_miou_fg", "?"))
    print(f"  Loaded {model_type} from {ckpt_path} (epoch={epoch}, val_mIoU_fg={miou})")
    return model


def ensemble_equal_logit(t1_logits, t2_logits):
    """Equal-weight logit-space averaging."""
    return (t1_logits + t2_logits) / 2.0


def ensemble_perclass_prob(t1_logits, t2_logits,
                           crack_t2_w=0.6, spalling_t2_w=0.3):
    """Per-class probability-space ensemble with renormalisation."""
    t1_prob = F.softmax(t1_logits, dim=1)
    t2_prob = F.softmax(t2_logits, dim=1)

    # bg weight: average of remaining
    bg_t2_w = 1.0 - (crack_t2_w + spalling_t2_w) / 2.0
    t2_weights = torch.tensor([bg_t2_w, crack_t2_w, spalling_t2_w],
                              device=t1_logits.device).view(1, 3, 1, 1)
    t1_weights = 1.0 - t2_weights

    mixed = t1_weights * t1_prob + t2_weights * t2_prob
    # Renormalise
    mixed = mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-8)
    # Convert back to logits
    return mixed.clamp_min(1e-8).log()


def ensemble_agreement_weighted(t1_logits, t2_logits, temperature=4.0):
    """Agreement-weighted pixel-level ensemble.
    At high-agreement pixels, use equal weight.
    At high-disagreement pixels, down-weight both (trust neither).
    """
    t1_prob = F.softmax(t1_logits / temperature, dim=1)
    t2_prob = F.softmax(t2_logits / temperature, dim=1)

    # Per-pixel KL divergence
    kl = (t1_prob * (t1_prob.clamp_min(1e-8).log()
                     - t2_prob.clamp_min(1e-8).log())).sum(1)  # (B, H, W)
    kl_max = kl.max().clamp_min(1e-6)
    agreement = 1.0 - (kl / kl_max)  # (B, H, W), 1=agree, 0=disagree

    # Weight ensemble by agreement (equal weight at agreement pixels)
    ens_logits = (t1_logits + t2_logits) / 2.0
    # Scale logits by agreement (low agreement -> softer prediction)
    return ens_logits * agreement.unsqueeze(1)


@torch.no_grad()
def evaluate_ensemble(t1_model, t2_model, dataloader, ensemble_fn, device,
                      label="ensemble"):
    """Run ensemble inference and compute metrics."""
    metrics = SegMetricsFull(num_classes=C.NUM_CLASSES, tol_px=BF1_TOLERANCE_PX)

    for batch in dataloader:
        imgs, masks, _ = batch
        imgs = imgs.to(device)
        H, W = masks.shape[-2:]

        t1_out = t1_model(imgs)
        t1_logits = F.interpolate(t1_out["seg_logits"].float(), (H, W),
                                  mode="bilinear", align_corners=False)

        t2_out = t2_model(imgs)
        t2_logits = F.interpolate(t2_out["seg_logits"].float(), (H, W),
                                  mode="bilinear", align_corners=False)

        ens_logits = ensemble_fn(t1_logits, t2_logits)
        # SegMetricsFull.update expects (logits_tensor, target_tensor)
        metrics.update(ens_logits, masks.to(ens_logits.device))

    result = metrics.compute()
    # Scale all metrics from [0,1] to percentage for display consistency
    for k in list(result.keys()):
        if k != "pixel_acc":
            result[k] = result[k] * 100
    return result


@torch.no_grad()
def evaluate_single(model, dataloader, device, label="model"):
    """Evaluate a single model."""
    metrics = SegMetricsFull(num_classes=C.NUM_CLASSES, tol_px=BF1_TOLERANCE_PX)

    for batch in dataloader:
        imgs, masks, _ = batch
        imgs = imgs.to(device)
        H, W = masks.shape[-2:]

        out = model(imgs)
        logits = F.interpolate(out["seg_logits"].float(), (H, W),
                               mode="bilinear", align_corners=False)
        metrics.update(logits, masks.to(logits.device))

    result = metrics.compute()
    for k in list(result.keys()):
        if k != "pixel_acc":
            result[k] = result[k] * 100
    return result


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fmt(results, keys=None):
    """Format a results dict for display."""
    if keys is None:
        keys = ["mIoU_fg", "IoU_crack", "IoU_spalling",
                "BF1_fg_mean", "clDice_fg_mean", "ConnR_fg_mean"]
    parts = []
    for k in keys:
        if k in results:
            parts.append(f"{k}={results[k]:.1f}")
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Teacher ensemble evaluation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="results/teacher_ensemble.json")
    parser.add_argument("--t1-ckpt", default="runs/dscformer_srl_G1/best.pt",
                        help="Teacher 1 checkpoint (relative to full_method/)")
    parser.add_argument("--t2-ckpt", default="runs/sam_lora_srl_SAM2/best.pt",
                        help="Teacher 2 checkpoint (relative to full_method/)")
    parser.add_argument("--t1-type", default="dscformer")
    parser.add_argument("--t2-type", default="sam_lora")
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Device: {device}")

    # Load data
    split_file = SPLIT_FILES[args.split]
    records = read_split_file(split_file)
    transform = build_transforms(512, train=False)
    dataset = DamSegmentDataset(C.DATA_ROOT, records, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Dataset: {len(dataset)} images ({args.split})")

    # Load teachers
    print("Loading Teacher 1...")
    t1 = load_teacher(args.t1_ckpt, args.t1_type, device)
    print("Loading Teacher 2...")
    t2 = load_teacher(args.t2_ckpt, args.t2_type, device)

    all_results = {}

    # Individual teachers
    print("\n=== Individual Teachers ===")
    r = evaluate_single(t1, loader, device, "Teacher 1")
    all_results["teacher1_individual"] = r
    print(f"Teacher 1 (DSConv+SRL):     {fmt(r)}")

    r = evaluate_single(t2, loader, device, "Teacher 2")
    all_results["teacher2_individual"] = r
    print(f"Teacher 2 (SAM-LoRA+SRL):   {fmt(r)}")

    # Ensemble variants
    print("\n=== Ensemble Upper Bounds ===")

    r = evaluate_ensemble(t1, t2, loader,
                          ensemble_equal_logit, device, "equal_logit")
    all_results["ensemble_equal_logit"] = r
    print(f"Equal-weight logit avg:     {fmt(r)}")

    r = evaluate_ensemble(t1, t2, loader,
                          ensemble_perclass_prob, device, "perclass_prob")
    all_results["ensemble_perclass_prob"] = r
    print(f"Per-class prob ensemble:    {fmt(r)}")

    r = evaluate_ensemble(t1, t2, loader,
                          ensemble_agreement_weighted, device, "agreement")
    all_results["ensemble_agreement_weighted"] = r
    print(f"Agreement-weighted:         {fmt(r)}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary comparison
    print("\n=== Summary (key metrics, all in %) ===")
    print(f"{'Configuration':<30s} {'mIoU_fg':>8s} {'IoU_cr':>8s} {'IoU_sp':>8s} "
          f"{'BF1_fg':>8s} {'clDice_fg':>9s} {'ConnR_fg':>8s}")
    print("-" * 90)
    rows = [
        ("Teacher 1 (DSConv+SRL)", "teacher1_individual"),
        ("Teacher 2 (SAM-LoRA+SRL)", "teacher2_individual"),
        ("Ensemble: equal logit", "ensemble_equal_logit"),
        ("Ensemble: per-class prob", "ensemble_perclass_prob"),
        ("Ensemble: agreement-wt", "ensemble_agreement_weighted"),
    ]
    for label, key in rows:
        r = all_results[key]
        print(f"{label:<30s} {r.get('mIoU_fg', 0):>7.1f}% {r.get('IoU_crack', 0):>7.1f}% "
              f"{r.get('IoU_spalling', 0):>7.1f}% {r.get('BF1_fg_mean', 0):>7.1f}% "
              f"{r.get('clDice_fg_mean', 0):>8.1f}% {r.get('ConnR_fg_mean', 0):>7.1f}%")
    print()
    print("Compare with DTKD student: mIoU_fg=72.3, IoU_cr=58.5, IoU_sp=86.2, "
          "BF1_fg=70.6, clDice_fg=86.4, ConnR_fg=83.0")


if __name__ == "__main__":
    main()

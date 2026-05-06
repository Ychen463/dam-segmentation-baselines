"""Test-Time Augmentation (TTA) evaluation for DSCformerDam.

Loads a trained checkpoint and evaluates on the test set with multi-scale
and flip augmentations, averaging logits before argmax.

Usage:
    python eval_tta.py --run dscformer_srl_G1
    python eval_tta.py --run dscformer_srl_G1 --scales 0.75 1.0 1.25
    python eval_tta.py --run dscformer_srl_G1 --no-flip  # scales only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from full_method import config as C
from full_method.model import DSCformerDam, SegFormerWithBoundary
from full_method.sam_model import TopoLoRASAM
from full_method.dinov2_model import DINOv2LoRA
from full_method.dataset import FullMethodDataset
from full_method.train import build_val_loader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(run_dir: Path, device: str):
    """Load model from best.pt checkpoint."""
    best_pt = run_dir / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"No best.pt in {run_dir}")

    state = torch.load(best_pt, map_location=device, weights_only=False)

    # Detect model type from run name or config
    run_name = run_dir.name
    is_dinov2 = "dinov2" in run_name.lower()
    is_sam = ("sam_lora" in run_name.lower() or "sam" in run_name.lower()) and not is_dinov2
    is_dscformer = "dscformer" in run_name.lower() or "dsc" in run_name.lower()

    if is_dinov2:
        cfg = C.RunCfg()
        cfg.model_type = "dinov2_lora"
        model = DINOv2LoRA(
            num_classes=C.NUM_CLASSES,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            fpn_dim=cfg.dinov2_fpn_dim,
            img_size=cfg.dinov2_img_size,
        )
    elif is_sam:
        cfg = C.RunCfg()
        cfg.model_type = "sam_lora"
        model = TopoLoRASAM(
            sam_checkpoint=cfg.sam_checkpoint,
            num_classes=C.NUM_CLASSES,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            fpn_dim=cfg.sam_fpn_dim,
            sam_img_size=cfg.sam_img_size,
        )
    elif is_dscformer:
        # Build a minimal cfg to reconstruct model
        cfg = C.RunCfg()
        # Check if multiscale
        if "multiscale" in run_name.lower():
            cfg.use_multiscale_snake = True
        model = DSCformerDam(cfg.pretrained, C.NUM_CLASSES, cfg=cfg)
    else:
        model = SegFormerWithBoundary(C.RunCfg().pretrained, C.NUM_CLASSES)

    model.load_state_dict(state["model"])
    model.to(device).eval()

    best_epoch = state.get("epoch", "?")
    best_miou = state.get("mIoU_fg", state.get("best_miou_fg", "?"))
    print(f"[TTA] loaded {best_pt.name} (epoch={best_epoch}, val_mIoU_fg={best_miou})")
    return model


def tta_predict(model, image: torch.Tensor, scales: list, use_flip: bool,
                target_size: tuple, device: str, use_amp: bool = True) -> torch.Tensor:
    """Run TTA on a single image tensor (1, 3, H, W) already normalized.

    Returns averaged logits (1, C, target_H, target_W).
    """
    B, C_in, H, W = image.shape
    assert B == 1, "TTA operates on single images"

    logits_sum = None
    n_augs = 0

    for scale in scales:
        sH, sW = int(H * scale), int(W * scale)
        if scale != 1.0:
            img_s = F.interpolate(image, size=(sH, sW), mode="bilinear", align_corners=False)
        else:
            img_s = image

        # List of transforms: original + optional flips
        transforms = [lambda x: x]  # identity
        if use_flip:
            transforms.append(lambda x: torch.flip(x, dims=[-1]))   # horizontal flip
            transforms.append(lambda x: torch.flip(x, dims=[-2]))   # vertical flip

        for tfm in transforms:
            img_aug = tfm(img_s).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                out = model(img_aug)
                logits = out["seg_logits"].float()  # (1, C, h, w)

            # Undo flip on logits
            logits = tfm(logits)

            # Resize to target
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum = logits_sum + logits
            n_augs += 1

    return logits_sum / n_augs


def main():
    parser = argparse.ArgumentParser(description="TTA evaluation")
    parser.add_argument("--run", type=str, required=True,
                        help="run directory name (e.g. dscformer_srl_G1)")
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0.75, 1.0, 1.25],
                        help="TTA scales (default: 0.75 1.0 1.25)")
    parser.add_argument("--no-flip", action="store_true",
                        help="disable flip augmentation")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="must be 1 for TTA (default)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = C.RUNS_DIR / args.run
    if not run_dir.exists():
        print(f"[ERROR] run directory not found: {run_dir}")
        sys.exit(1)

    # Load model
    model = load_model(run_dir, device)

    # Build test loader
    cfg = C.RunCfg()
    cfg.batch_size = 1  # TTA requires batch_size=1
    test_files = []
    with open(C.SPLIT_FILES["test"]) as f:
        test_files = [line.strip() for line in f if line.strip()]
    test_loader = build_val_loader(test_files, cfg, device)

    # Metrics
    try:
        from shared_eval.metrics_full import SegMetricsFull
        metrics = SegMetricsFull(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    except ImportError:
        print("[TTA] WARNING: shared_eval not available, using basic metrics")
        from full_method.train import SegMetricsBF1
        metrics = SegMetricsBF1(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)

    metrics.reset()
    use_amp = (device == "cuda")
    use_flip = not args.no_flip
    scales = args.scales

    aug_desc = f"scales={scales}"
    if use_flip:
        aug_desc += " + hflip + vflip"
    n_augs = len(scales) * (3 if use_flip else 1)
    print(f"[TTA] {aug_desc}  ({n_augs} views per image)")
    print(f"[TTA] evaluating {len(test_files)} test images...")

    for i, batch in enumerate(test_loader):
        imgs = batch["image"].to(device, non_blocking=True).float()
        masks = batch["mask"].to(device, non_blocking=True).long()
        target_size = masks.shape[-2:]

        avg_logits = tta_predict(model, imgs, scales, use_flip, target_size,
                                 device, use_amp=use_amp)
        metrics.update(avg_logits, masks)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_loader)}]")

    m = metrics.compute()

    # Print results
    print(f"\n{'='*60}")
    print(f"TTA Results: {args.run}")
    print(f"Augmentations: {aug_desc}")
    print(f"{'='*60}")
    for k in ("IoU_background", "IoU_crack", "IoU_spalling",
              "Dice_background", "Dice_crack", "Dice_spalling",
              "mIoU_fg", "mIoU_all", "pixel_acc",
              "BF1_crack", "BF1_spalling", "BF1_fg_mean",
              "clDice_crack", "clDice_spalling", "clDice_fg_mean",
              "ConnR_crack", "ConnR_spalling", "ConnR_fg_mean"):
        val = m.get(k)
        if val is not None:
            print(f"  {k}: {val:.6f}")
    print(f"{'='*60}")

    # Save report
    report_path = run_dir / "test_report_tta.txt"
    with open(report_path, "w") as f:
        f.write(f"run: {args.run}\n")
        f.write(f"tta: scales={scales} flip={use_flip} n_views={n_augs}\n")
        f.write("test set metrics (TTA):\n")
        for k in ("IoU_background", "IoU_crack", "IoU_spalling",
                  "Dice_background", "Dice_crack", "Dice_spalling",
                  "mIoU_fg", "mIoU_all", "pixel_acc",
                  "BF1_crack", "BF1_spalling", "BF1_fg_mean",
                  "clDice_crack", "clDice_spalling", "clDice_fg_mean",
                  "ConnR_crack", "ConnR_spalling", "ConnR_fg_mean"):
            f.write(f"  {k}: {m.get(k)}\n")
        if hasattr(metrics, 'cm'):
            f.write("confusion matrix (rows=gt, cols=pred):\n")
            f.write(str(metrics.cm) + "\n")
    print(f"[TTA] wrote {report_path}")


if __name__ == "__main__":
    main()

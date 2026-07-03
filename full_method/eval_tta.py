"""Test set evaluation with optional TTA for all model types.

Supports preset-based model loading via --ablation flag, which correctly
reconstructs the model architecture from ABLATION_PRESETS in config.py.

Usage:
    # Evaluate DKD5e with TTA (preset-based)
    python eval_tta.py --run dkd5e_crk07_spl02 --ablation DKD5e

    # Evaluate DKD5e WITHOUT TTA (test-only)
    python eval_tta.py --run dkd5e_crk07_spl02 --ablation DKD5e --no-tta

    # Evaluate G1 with TTA (auto-detected by run name)
    python eval_tta.py --run dscformer_srl_G1

    # Evaluate with explicit model type
    python eval_tta.py --run my_run --model-type dscformer

    # Custom TTA scales
    python eval_tta.py --run dkd5e_crk07_spl02 --ablation DKD5e --scales 0.75 1.0 1.25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from full_method import config as C
from full_method.model import DSCformerDam, SegFormerWithBoundary
from full_method.sam_model import TopoLoRASAM
from full_method.dinov2_model import DINOv2LoRA
from full_method.train import build_val_loader


def load_model(run_dir: Path, device: str, ablation: str | None = None,
               model_type: str | None = None):
    """Load model from best.pt checkpoint.

    Args:
        run_dir: path to run directory containing best.pt
        device: torch device
        ablation: preset name (e.g. "DKD5e") to reconstruct cfg via apply_preset
        model_type: explicit model type override ("dscformer", "sam_lora", "dinov2_lora", "segformer")
    """
    best_pt = run_dir / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"No best.pt in {run_dir}")

    state = torch.load(best_pt, map_location=device, weights_only=False)

    # Determine model type: ablation preset > explicit flag > run name heuristic
    cfg = C.RunCfg()

    if ablation is not None:
        C.apply_preset(cfg, ablation)
        mtype = cfg.model_type
        print(f"[eval] using ablation preset '{ablation}' -> model_type='{mtype}'")
    elif model_type is not None:
        mtype = model_type
        cfg.model_type = mtype
        print(f"[eval] using explicit model_type='{mtype}'")
    else:
        # Fallback: detect from run name, then from state_dict keys
        run_name = run_dir.name.lower()
        if "dinov2" in run_name:
            mtype = "dinov2_lora"
        elif "sam_lora" in run_name or "sam" in run_name:
            mtype = "sam_lora"
        elif "dscformer" in run_name or "dsc" in run_name:
            mtype = "dscformer"
        elif any(k.startswith("snake_branch.") for k in state["model"]):
            mtype = "dscformer"
        else:
            mtype = "segformer"
        print(f"[eval] auto-detected model_type='{mtype}' from run name '{run_dir.name}'")

    # Build model
    if mtype == "dinov2_lora":
        model = DINOv2LoRA(
            num_classes=C.NUM_CLASSES,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            fpn_dim=cfg.dinov2_fpn_dim,
            img_size=cfg.dinov2_img_size,
        )
    elif mtype == "sam_lora":
        model = TopoLoRASAM(
            sam_checkpoint=cfg.sam_checkpoint,
            num_classes=C.NUM_CLASSES,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            fpn_dim=cfg.sam_fpn_dim,
            sam_img_size=cfg.sam_img_size,
        )
    elif mtype == "dscformer":
        model = DSCformerDam(cfg.pretrained, C.NUM_CLASSES, cfg=cfg)
    else:
        model = SegFormerWithBoundary(cfg.pretrained, C.NUM_CLASSES)

    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    # Filter out known legacy keys (boundary_head was removed)
    unexpected = [k for k in unexpected if "boundary_head" not in k]
    if unexpected:
        print(f"[eval] WARNING: unexpected keys: {unexpected[:5]}")
    if missing:
        print(f"[eval] WARNING: missing keys: {missing[:5]}")
    model.to(device).eval()

    best_epoch = state.get("epoch", "?")
    best_miou = state.get("mIoU_fg", state.get("best_miou_fg", "?"))
    print(f"[eval] loaded {best_pt.name} (epoch={best_epoch}, val_mIoU_fg={best_miou})")
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


def plain_predict(model, image: torch.Tensor, target_size: tuple,
                  device: str, use_amp: bool = True) -> torch.Tensor:
    """Standard forward pass without TTA. Returns logits (B, C, H, W)."""
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        out = model(image)
        logits = out["seg_logits"].float()
    logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    return logits


METRIC_KEYS = (
    "IoU_background", "IoU_crack", "IoU_spalling",
    "Dice_background", "Dice_crack", "Dice_spalling",
    "mIoU_fg", "mIoU_all", "pixel_acc",
    "BF1_crack", "BF1_spalling", "BF1_fg_mean",
    "clDice_crack", "clDice_spalling", "clDice_fg_mean",
    "ConnR_crack", "ConnR_spalling", "ConnR_fg_mean",
)


def main():
    parser = argparse.ArgumentParser(description="Test set evaluation (with optional TTA)")
    parser.add_argument("--run", type=str, required=True,
                        help="run directory name (e.g. dkd5e_crk07_spl02)")
    parser.add_argument("--ablation", type=str, default=None,
                        help="ablation preset name (e.g. DKD5e) for correct model reconstruction")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=["dscformer", "sam_lora", "dinov2_lora", "segformer"],
                        help="explicit model type override")
    parser.add_argument("--no-tta", action="store_true",
                        help="disable TTA, run plain test evaluation")
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0.75, 1.0, 1.25],
                        help="TTA scales (default: 0.75 1.0 1.25)")
    parser.add_argument("--no-flip", action="store_true",
                        help="disable flip augmentation in TTA")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = C.RUNS_DIR / args.run
    if not run_dir.exists():
        print(f"[ERROR] run directory not found: {run_dir}")
        sys.exit(1)

    # Load model
    model = load_model(run_dir, device, ablation=args.ablation,
                       model_type=args.model_type)

    # Build test loader
    cfg = C.RunCfg()
    cfg.batch_size = 1
    test_files = []
    with open(C.SPLIT_FILES["test"]) as f:
        test_files = [line.strip() for line in f if line.strip()]
    test_loader = build_val_loader(test_files, cfg, device)

    # Metrics
    try:
        from shared_eval.metrics_full import SegMetricsFull
        metrics = SegMetricsFull(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    except ImportError:
        print("[eval] WARNING: shared_eval not available, using basic metrics")
        from full_method.train import SegMetricsBF1
        metrics = SegMetricsBF1(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)

    metrics.reset()
    use_amp = (device == "cuda")
    use_tta = not args.no_tta
    use_flip = not args.no_flip
    scales = args.scales

    if use_tta:
        aug_desc = f"scales={scales}"
        if use_flip:
            aug_desc += " + hflip + vflip"
        n_augs = len(scales) * (3 if use_flip else 1)
        mode_str = f"TTA ({aug_desc}, {n_augs} views)"
    else:
        mode_str = "plain (no TTA)"

    print(f"[eval] mode: {mode_str}")
    print(f"[eval] evaluating {len(test_files)} test images...")

    for i, batch in enumerate(test_loader):
        imgs = batch["image"].to(device, non_blocking=True).float()
        masks = batch["mask"].to(device, non_blocking=True).long()
        target_size = masks.shape[-2:]

        if use_tta:
            logits = tta_predict(model, imgs, scales, use_flip, target_size,
                                 device, use_amp=use_amp)
        else:
            logits = plain_predict(model, imgs, target_size, device,
                                   use_amp=use_amp)

        metrics.update(logits, masks)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_loader)}]")

    m = metrics.compute()

    # Print results
    suffix = "TTA" if use_tta else "plain"
    print(f"\n{'='*60}")
    print(f"Test Results ({suffix}): {args.run}")
    if use_tta:
        print(f"Augmentations: {aug_desc}")
    print(f"{'='*60}")
    for k in METRIC_KEYS:
        val = m.get(k)
        if val is not None:
            print(f"  {k}: {val:.6f}")
    print(f"{'='*60}")

    # Save report
    report_name = f"test_report_{'tta' if use_tta else 'plain'}.txt"
    report_path = run_dir / report_name
    with open(report_path, "w") as f:
        f.write(f"run: {args.run}\n")
        if args.ablation:
            f.write(f"ablation_preset: {args.ablation}\n")
        f.write(f"mode: {mode_str}\n")
        if use_tta:
            f.write(f"tta: scales={scales} flip={use_flip} n_views={n_augs}\n")
        f.write(f"test set metrics ({suffix}):\n")
        for k in METRIC_KEYS:
            f.write(f"  {k}: {m.get(k)}\n")
        if hasattr(metrics, 'cm'):
            f.write("confusion matrix (rows=gt, cols=pred):\n")
            f.write(str(metrics.cm) + "\n")
    print(f"[eval] wrote {report_path}")


if __name__ == "__main__":
    main()

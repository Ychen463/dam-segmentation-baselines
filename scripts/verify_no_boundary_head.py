"""Verify that removing boundary_head does not change segmentation results.

Steps:
1. Load existing checkpoint (which has boundary_head weights) with strict=False
2. Evaluate on test set — results should be IDENTICAL to original
3. Report new parameter count and FPS

Usage (on RunPod):
    cd /workspace/Codes
    python -m scripts.verify_no_boundary_head \
        --checkpoint full_method/runs/DKD10_segformer_b2_full_512/best.pt \
        --name "TopoDistill (row c)"

    # Or run all key checkpoints:
    python -m scripts.verify_no_boundary_head --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline_unet.dataset import (
    DamSegmentDataset, build_transforms, read_split_file,
)
from baseline_unet import config as base_C
from full_method import config as C
from full_method.model import DSCformerDam, SegFormerWithBoundary
from shared_eval.metrics_full import SegMetricsFull

# FPS measurement settings (match shared_eval/efficiency.py)
WARMUP_ITERS = 10
MEASURE_ITERS = 50


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_fps(model, img_size, device):
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    is_cuda = device == "cuda"

    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            model(dummy)
        if is_cuda:
            torch.cuda.synchronize()

    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_ITERS):
        with torch.no_grad():
            model(dummy)
        if is_cuda:
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) / MEASURE_ITERS * 1000.0
    return 1000.0 / latency_ms if latency_ms > 0 else 0.0


def evaluate_checkpoint(ckpt_path, name, device, test_loader, cfg):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    # Build model WITHOUT boundary_head
    model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)

    # Load old checkpoint (has boundary_head keys -> strict=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    key = "ema_model" if "ema_model" in state else "model"
    missing, unexpected = model.load_state_dict(state[key], strict=False)

    print(f"\n  Loading with strict=False:")
    print(f"    Missing keys:    {missing if missing else '(none)'}")
    print(f"    Unexpected keys: {unexpected if unexpected else '(none)'}")

    # Verify unexpected keys are ONLY boundary_head
    non_boundary = [k for k in unexpected if "boundary_head" not in k]
    if non_boundary:
        print(f"  WARNING: unexpected non-boundary keys: {non_boundary}")
    else:
        print(f"  OK: all unexpected keys are boundary_head (removed as expected)")

    model.eval()

    # Parameter count
    n_params = count_params(model)
    print(f"\n  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # FPS
    fps = measure_fps(model, cfg.img_size, device)
    print(f"  FPS (bs=1, {cfg.img_size}x{cfg.img_size}): {fps:.1f}")

    # Evaluate on test set
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

    m = metrics.compute()
    print(f"\n  Test results:")
    print(f"    mIoU_fg:    {m['mIoU_fg']:.4f}")
    print(f"    IoU_cr:     {m['IoU_crack']:.4f}")
    print(f"    IoU_sp:     {m['IoU_spalling']:.4f}")
    print(f"    BF1_fg:     {m['BF1_fg_mean']:.4f}")
    if 'clDice_fg_mean' in m:
        print(f"    clDice_fg:  {m['clDice_fg_mean']:.4f}")
    if 'ConnR_fg_mean' in m:
        print(f"    ConnR_fg:   {m['ConnR_fg_mean']:.4f}")

    return {
        "name": name,
        "params_M": round(n_params / 1e6, 2),
        "fps": round(fps, 1),
        "mIoU_fg": round(m["mIoU_fg"], 4),
        "IoU_cr": round(m["IoU_crack"], 4),
        "IoU_sp": round(m["IoU_spalling"], 4),
        "BF1_fg": round(m["BF1_fg_mean"], 4),
    }


# Known checkpoints to verify
ALL_CHECKPOINTS = [
    ("full_method/runs/DKD10_segformer_b2_full_512/best.pt", "TopoDistill (row c)"),
    ("full_method/runs/segformer_b2_dsconv_only/best.pt", "DSConv-only (row b)"),
]


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Single checkpoint to verify")
    group.add_argument("--all", action="store_true", help="Verify all known checkpoints")
    parser.add_argument("--name", type=str, default="model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = C.RunCfg()
    cfg.use_boundary_loss = False

    # Load test set
    test_files = read_split_file(C.SPLIT_FILES["test"])
    test_ds = DamSegmentDataset(base_C.DATA_ROOT, test_files,
                                 build_transforms(cfg.img_size, train=False))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                              num_workers=2, pin_memory=(device == "cuda"))
    print(f"Test set: {len(test_files)} images")

    if args.all:
        results = []
        for ckpt, name in ALL_CHECKPOINTS:
            ckpt_path = ROOT / ckpt
            if ckpt_path.exists():
                r = evaluate_checkpoint(ckpt_path, name, device, test_loader, cfg)
                results.append(r)
            else:
                print(f"\nSKIPPED (not found): {ckpt_path}")
        # Also measure plain SegFormer (no DSConv, no boundary head)
        print(f"\n{'='*60}")
        print(f"  Plain SegFormer-B2 (no DSConv, no boundary head)")
        print(f"{'='*60}")
        plain = SegFormerWithBoundary(cfg.pretrained).to(device).eval()
        n = count_params(plain)
        fps = measure_fps(plain, cfg.img_size, device)
        print(f"  Parameters: {n:,} ({n/1e6:.2f}M)")
        print(f"  FPS: {fps:.1f}")
        results.append({"name": "SegFormer-B2 (plain)", "params_M": round(n/1e6, 2), "fps": round(fps, 1)})

        out_path = ROOT / "results" / "verify_no_boundary_head.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
        evaluate_checkpoint(ckpt_path, args.name, device, test_loader, cfg)


if __name__ == "__main__":
    main()

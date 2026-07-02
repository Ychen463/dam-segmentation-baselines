"""Evaluate Multi-Scale DSConv experiments.

Compares:
  1. SegFormer-B2 (no DSConv)
  2. DSConv single-scale k=9 (no SRL)
  3. DSConv single-scale k=9 + SRL
  4. MS0: Multi-Scale DSConv k=5/9/15 (no SRL)
  5. MS1: Multi-Scale DSConv k=5/9/15 + SRL

Usage on RunPod:
    cd /workspace/dam-segmentation-baselines
    python scripts/eval_multiscale_dsconv.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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

# (path, name, model_class, use_multiscale)
CHECKPOINTS = [
    ("full_method/runs/plain_segformer_P0/best.pt", "SegFormer-B2 (no DSConv)", "segformer", False),
    ("full_method/runs/dscformer_plain_G0/best.pt", "DSConv k=9 (no SRL)", "dscformer", False),
    ("full_method/runs/dscformer_srl_G1_v2/best.pt", "DSConv k=9 + SRL", "dscformer", False),
    ("full_method/runs/dscformer_multiscale_MS0/best.pt", "MS-DSConv k=5/9/15 (no SRL)", "dscformer", True),
    ("full_method/runs/dscformer_multiscale_MS1/best.pt", "MS-DSConv k=5/9/15 + SRL", "dscformer", True),
]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def evaluate(ckpt_path, name, model_type, use_multiscale, device, test_loader):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    cfg = C.RunCfg()
    cfg.use_boundary_loss = False
    cfg.use_multiscale_snake = use_multiscale
    if use_multiscale:
        cfg.snake_kernel_sizes = (5, 9, 15)

    if model_type == "segformer":
        model = SegFormerWithBoundary(cfg.pretrained).to(device)
    else:
        model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    key = "ema_model" if "ema_model" in state else "model"
    missing, unexpected = model.load_state_dict(state[key], strict=False)

    other_keys = [k for k in unexpected if "boundary_head" not in k]
    if other_keys:
        print(f"  WARNING: unexpected keys: {other_keys[:5]}")
    if missing:
        print(f"  WARNING: missing keys: {missing[:5]}")

    model.eval()
    n_params = count_params(model)
    print(f"  Params: {n_params/1e6:.2f}M")

    metrics = SegMetricsFull(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    metrics.reset()
    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs = imgs.to(device).float()
            masks = masks.to(device).long()

            if model_type == "segformer":
                logits = model(imgs)
                if isinstance(logits, dict):
                    logits = logits["seg_logits"]
                seg_logits = F.interpolate(logits, masks.shape[-2:],
                                           mode="bilinear", align_corners=False)
            else:
                out = model(imgs)
                seg_logits = F.interpolate(out["seg_logits"], masks.shape[-2:],
                                           mode="bilinear", align_corners=False)
            metrics.update(seg_logits, masks)

    m = metrics.compute()
    print(f"  mIoU_fg={m['mIoU_fg']:.4f}  IoU_cr={m['IoU_crack']:.4f}"
          f"  IoU_sp={m['IoU_spalling']:.4f}  BF1_fg={m['BF1_fg_mean']:.4f}")
    if 'clDice_fg_mean' in m:
        print(f"  clDice_fg={m['clDice_fg_mean']:.4f}", end="")
    if 'ConnR_fg_mean' in m:
        print(f"  CompR_fg={m['ConnR_fg_mean']:.4f}", end="")
    print()

    return {
        "name": name,
        "params_M": round(n_params / 1e6, 2),
        "mIoU_fg": round(m["mIoU_fg"], 4),
        "IoU_cr": round(m["IoU_crack"], 4),
        "IoU_sp": round(m["IoU_spalling"], 4),
        "BF1_fg": round(m["BF1_fg_mean"], 4),
        "clDice_fg": round(m.get("clDice_fg_mean", 0), 4),
        "CompR_fg": round(m.get("ConnR_fg_mean", 0), 4),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = C.RunCfg()
    test_files = read_split_file(C.SPLIT_FILES["test"])
    test_ds = DamSegmentDataset(base_C.DATA_ROOT, test_files,
                                 build_transforms(cfg.img_size, train=False))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                              num_workers=2, pin_memory=(device == "cuda"))
    print(f"Test set: {len(test_files)} images")

    results = []
    for rel_path, name, model_type, use_ms in CHECKPOINTS:
        ckpt = ROOT / rel_path
        if not ckpt.exists():
            print(f"\nSKIPPED (not found): {ckpt}")
            continue
        r = evaluate(ckpt, name, model_type, use_ms, device, test_loader)
        results.append(r)

    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY: Multi-Scale DSConv Comparison")
    print(f"{'='*80}")
    print(f"{'Configuration':<30s} {'Params':>7s} {'mIoU_fg':>8s} {'IoU_cr':>8s} "
          f"{'IoU_sp':>8s} {'BF1_fg':>8s} {'clDice':>8s} {'CompR':>8s}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<30s} {r['params_M']:>6.2f}M {r['mIoU_fg']:>8.4f} "
              f"{r['IoU_cr']:>8.4f} {r['IoU_sp']:>8.4f} {r['BF1_fg']:>8.4f} "
              f"{r['clDice_fg']:>8.4f} {r['CompR_fg']:>8.4f}")

    # Key deltas
    by_name = {r["name"]: r for r in results}
    ss = by_name.get("DSConv k=9 (no SRL)")
    ms = by_name.get("MS-DSConv k=5/9/15 (no SRL)")
    if ss and ms:
        print(f"\n--- Multi-Scale vs Single-Scale (no SRL) ---")
        for k in ["mIoU_fg", "IoU_cr", "IoU_sp", "BF1_fg", "clDice_fg", "CompR_fg"]:
            d = ms[k] - ss[k]
            print(f"  {k}: {d:+.4f}")
        print(f"  Param overhead: {ms['params_M'] - ss['params_M']:+.2f}M")

    out_path = ROOT / "results" / "multiscale_dsconv.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

"""Evaluate Snake Channel / Kernel-size sweep experiments.

Compares snake_channels={32,64} x snake_kernel_size={7,9,13} (6 configs total).
G0 (ch=64, k=9) is the baseline — already trained.

Usage on RunPod:
    cd /workspace/dam-segmentation-baselines
    python scripts/eval_snake_sweep.py
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
from full_method.model import DSCformerDam
from shared_eval.metrics_full import SegMetricsFull

# (run_dir, display_name, snake_channels, snake_kernel_size)
CONFIGS = [
    ("dscformer_plain_G0",      "ch=64, k=9  (G0 baseline)", 64,  9),
    ("dscformer_ch32_k9_SC0",   "ch=32, k=9  (SC0)",         32,  9),
    ("dscformer_ch64_k7_SC1",   "ch=64, k=7  (SC1)",         64,  7),
    ("dscformer_ch64_k13_SC2",  "ch=64, k=13 (SC2)",         64, 13),
    ("dscformer_ch32_k7_SC3",   "ch=32, k=7  (SC3)",         32,  7),
    ("dscformer_ch32_k13_SC4",  "ch=32, k=13 (SC4)",         32, 13),
]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def evaluate(run_dir, name, snake_channels, snake_kernel_size, device, test_loader):
    ckpt_path = ROOT / "full_method" / "runs" / run_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"\n  SKIPPED (not found): {ckpt_path}")
        return None

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    cfg = C.RunCfg()
    cfg.model_type = "dscformer"
    cfg.snake_channels = snake_channels
    cfg.snake_kernel_size = snake_kernel_size
    cfg.use_boundary_loss = False

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
        "snake_channels": snake_channels,
        "snake_kernel_size": snake_kernel_size,
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
    for run_dir, name, ch, ks in CONFIGS:
        r = evaluate(run_dir, name, ch, ks, device, test_loader)
        if r:
            results.append(r)

    # Summary table
    print(f"\n{'='*90}")
    print("  SUMMARY: Snake Channel / Kernel-size Sweep")
    print(f"{'='*90}")
    print(f"{'Configuration':<30s} {'ch':>4s} {'k':>4s} {'Params':>7s} "
          f"{'mIoU_fg':>8s} {'IoU_cr':>8s} {'IoU_sp':>8s} {'BF1_fg':>8s} "
          f"{'clDice':>8s} {'CompR':>8s}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<30s} {r['snake_channels']:>4d} {r['snake_kernel_size']:>4d} "
              f"{r['params_M']:>6.2f}M {r['mIoU_fg']:>8.4f} {r['IoU_cr']:>8.4f} "
              f"{r['IoU_sp']:>8.4f} {r['BF1_fg']:>8.4f} {r['clDice_fg']:>8.4f} "
              f"{r['CompR_fg']:>8.4f}")

    # Deltas vs baseline (G0: ch=64, k=9)
    by_name = {(r["snake_channels"], r["snake_kernel_size"]): r for r in results}
    baseline = by_name.get((64, 9))
    if baseline:
        print(f"\n--- Deltas vs baseline (ch=64, k=9) ---")
        for r in results:
            if r is baseline:
                continue
            print(f"\n  {r['name']}:")
            for k in ["mIoU_fg", "IoU_cr", "IoU_sp", "BF1_fg", "clDice_fg", "CompR_fg"]:
                d = r[k] - baseline[k]
                print(f"    {k}: {d:+.4f}")
            dp = r["params_M"] - baseline["params_M"]
            print(f"    Params: {dp:+.2f}M")

    out_path = ROOT / "results" / "snake_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

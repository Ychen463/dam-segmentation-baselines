"""Verify that removing boundary_head does not change segmentation results.

Loads each checkpoint with strict=False (old checkpoints have boundary_head
keys that the new model no longer has), evaluates on the test set, and
reports new parameter counts and FPS.

Usage (on RunPod):
    cd /workspace/dam-segmentation-baselines
    python -m scripts.verify_no_boundary_head
"""
from __future__ import annotations

import json
import sys
import time
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

WARMUP_ITERS = 10
MEASURE_ITERS = 50

# All checkpoints to verify (path relative to ROOT, display name)
CHECKPOINTS = [
    # Row (c): HeteroDistill student (DSConv + DTKD, no SRL)
    ("full_method/runs/dkd10_no_srl_rerun/best.pt", "HeteroDistill row(c)"),
    # Row (b): DSConv-only baseline
    ("full_method/runs/dscformer_plain_G0/best.pt", "DSConv-only row(b)"),
    # Teacher 1: DSConv + SRL
    ("full_method/runs/dscformer_srl_G1/best.pt", "Teacher1 (DSConv+SRL)"),
    # Multi-seed DSConv
    ("full_method/runs/dscformer_plain_G0_seed42/best.pt", "DSConv seed42"),
    ("full_method/runs/dscformer_plain_G0_seed123/best.pt", "DSConv seed123"),
    ("full_method/runs/dscformer_plain_G0_seed2024/best.pt", "DSConv seed2024"),
    # Controls
    ("full_method/runs/dscformer_dup_t1_CTRL/best.pt", "CTRL: dup teacher"),
    ("full_method/runs/dscformer_samearch_CTRL/best.pt", "CTRL: same-arch"),
    ("full_method/runs/dscformer_labelsmooth_CTRL/best.pt", "CTRL: label smooth"),
    # Factorial
    ("full_method/runs/segformer_dtkd_FACT1/best.pt", "FACT1: SegFormer+DTKD"),
]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_fps(model, img_size, device):
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    is_cuda = (device == "cuda")
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
    print(f"  {ckpt_path}")
    print(f"{'='*60}")

    model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    key = "ema_model" if "ema_model" in state else "model"
    missing, unexpected = model.load_state_dict(state[key], strict=False)

    bd_keys = [k for k in unexpected if "boundary_head" in k]
    other_keys = [k for k in unexpected if "boundary_head" not in k]
    print(f"  boundary_head keys (dropped): {len(bd_keys)}")
    if other_keys:
        print(f"  WARNING: other unexpected keys: {other_keys}")
    if missing:
        print(f"  WARNING: missing keys: {missing}")

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
        print(f"  ConnR_fg={m['ConnR_fg_mean']:.4f}", end="")
    print()

    return {
        "name": name,
        "params_M": round(n_params / 1e6, 2),
        "mIoU_fg": round(m["mIoU_fg"], 4),
        "IoU_cr": round(m["IoU_crack"], 4),
        "IoU_sp": round(m["IoU_spalling"], 4),
        "BF1_fg": round(m["BF1_fg_mean"], 4),
        "clDice_fg": round(m.get("clDice_fg_mean", 0), 4),
        "ConnR_fg": round(m.get("ConnR_fg_mean", 0), 4),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = C.RunCfg()
    cfg.use_boundary_loss = False

    # Test data
    test_files = read_split_file(C.SPLIT_FILES["test"])
    test_ds = DamSegmentDataset(base_C.DATA_ROOT, test_files,
                                 build_transforms(cfg.img_size, train=False))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                              num_workers=2, pin_memory=(device == "cuda"))
    print(f"Test set: {len(test_files)} images")

    # ---- Part 1: Evaluate all checkpoints ----
    results = []
    for rel_path, name in CHECKPOINTS:
        ckpt = ROOT / rel_path
        if not ckpt.exists():
            print(f"\nSKIPPED (not found): {ckpt}")
            continue
        r = evaluate_checkpoint(ckpt, name, device, test_loader, cfg)
        results.append(r)

    # ---- Part 2: New parameter counts ----
    print(f"\n{'='*60}")
    print("  New parameter counts (boundary_head removed)")
    print(f"{'='*60}")

    plain = SegFormerWithBoundary(cfg.pretrained).to(device).eval()
    n_plain = count_params(plain)
    print(f"  SegFormer-B2 (plain):  {n_plain/1e6:.2f}M")

    dsc = DSCformerDam(cfg.pretrained, cfg=cfg).to(device).eval()
    n_dsc = count_params(dsc)
    print(f"  SegFormer-DSConv:      {n_dsc/1e6:.2f}M")
    print(f"  DSConv overhead:       {(n_dsc - n_plain)/1e6:.2f}M")

    # ---- Part 3: FPS measurement ----
    print(f"\n{'='*60}")
    print("  FPS measurement (bs=1, 512x512, A40)")
    print(f"{'='*60}")

    fps_plain = measure_fps(plain, 512, device)
    print(f"  SegFormer-B2:     {fps_plain:.1f} FPS")

    fps_dsc = measure_fps(dsc, 512, device)
    print(f"  SegFormer-DSConv: {fps_dsc:.1f} FPS")
    del plain, dsc

    # ---- Save results ----
    summary = {
        "params": {"SegFormer_B2_M": round(n_plain/1e6, 2),
                    "DSConv_M": round(n_dsc/1e6, 2),
                    "overhead_M": round((n_dsc - n_plain)/1e6, 2)},
        "fps": {"SegFormer_B2": round(fps_plain, 1),
                "DSConv": round(fps_dsc, 1)},
        "checkpoints": results,
    }
    out_path = ROOT / "results" / "verify_no_boundary_head.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

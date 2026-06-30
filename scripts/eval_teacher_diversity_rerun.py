"""Evaluate the three matched-checkpoint teacher diversity reruns.

Reports mIoU_fg, IoU_cr, IoU_sp, BF1_fg, clDice_fg, CompR_fg for:
  1. rerun_t1only     — T1-only KD
  2. rerun_dup_t1     — Duplicated T1 (same checkpoint twice)
  3. rerun_hetero     — Heterogeneous T1+T2 DTKD

Also includes the DSConv-only baseline (no KD) for reference.

Usage on RunPod:
    cd /workspace/dam-segmentation-baselines
    python scripts/eval_teacher_diversity_rerun.py
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

CHECKPOINTS = [
    ("full_method/runs/dscformer_plain_G0/best.pt", "DSConv-only (no KD)"),
    ("full_method/runs/rerun_t1only/best.pt", "RERUN: T1-only KD"),
    ("full_method/runs/rerun_dup_t1/best.pt", "RERUN: Duplicated T1"),
    ("full_method/runs/rerun_hetero/best.pt", "RERUN: Heterogeneous T1+T2"),
]


def evaluate(ckpt_path, name, device, test_loader, cfg):
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
    if other_keys:
        print(f"  WARNING: unexpected keys: {other_keys}")
    if missing:
        print(f"  WARNING: missing keys: {missing}")

    model.eval()

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
    cfg.use_boundary_loss = False

    test_files = read_split_file(C.SPLIT_FILES["test"])
    test_ds = DamSegmentDataset(base_C.DATA_ROOT, test_files,
                                 build_transforms(cfg.img_size, train=False))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                              num_workers=2, pin_memory=(device == "cuda"))
    print(f"Test set: {len(test_files)} images")

    # Verify T1 checkpoint hash
    t1_ckpt = ROOT / "full_method" / "runs" / "dscformer_srl_G1" / "best.pt"
    if t1_ckpt.exists():
        import hashlib
        h = hashlib.md5(t1_ckpt.read_bytes()).hexdigest()
        print(f"T1 checkpoint MD5: {h}")

    results = []
    for rel_path, name in CHECKPOINTS:
        ckpt = ROOT / rel_path
        if not ckpt.exists():
            print(f"\nSKIPPED (not found): {ckpt}")
            continue
        r = evaluate(ckpt, name, device, test_loader, cfg)
        results.append(r)

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY: Teacher Diversity Matched-Checkpoint Comparison")
    print(f"{'='*60}")
    print(f"{'Configuration':<30s} {'mIoU_fg':>8s} {'IoU_cr':>8s} {'IoU_sp':>8s} {'BF1_fg':>8s}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<30s} {r['mIoU_fg']:>8.4f} {r['IoU_cr']:>8.4f} "
              f"{r['IoU_sp']:>8.4f} {r['BF1_fg']:>8.4f}")

    # Key comparisons
    by_name = {r["name"]: r for r in results}
    if "RERUN: Heterogeneous T1+T2" in by_name and "RERUN: Duplicated T1" in by_name:
        delta = by_name["RERUN: Heterogeneous T1+T2"]["mIoU_fg"] - by_name["RERUN: Duplicated T1"]["mIoU_fg"]
        print(f"\nHeterogeneous - Duplicated = {delta:+.4f} mIoU_fg")
    if "RERUN: Heterogeneous T1+T2" in by_name and "RERUN: T1-only KD" in by_name:
        delta = by_name["RERUN: Heterogeneous T1+T2"]["mIoU_fg"] - by_name["RERUN: T1-only KD"]["mIoU_fg"]
        print(f"Heterogeneous - T1-only    = {delta:+.4f} mIoU_fg")
    if "RERUN: Duplicated T1" in by_name and "DSConv-only (no KD)" in by_name:
        delta = by_name["RERUN: Duplicated T1"]["mIoU_fg"] - by_name["DSConv-only (no KD)"]["mIoU_fg"]
        print(f"Duplicated    - DSConv-only = {delta:+.4f} mIoU_fg")

    out_path = ROOT / "results" / "teacher_diversity_rerun.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

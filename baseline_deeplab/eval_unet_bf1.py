"""Backfill Boundary F1 for the Step 0 U-Net best checkpoint.

Evaluated at Step 0's native IMG_SIZE=320 to reflect its actual deployment
state. BF1 uses absolute 2px tolerance (same as Step 1); see STEP1_REPORT
footnote about cross-resolution fairness.

Writes:
    baseline_deeplab/runs/deeplabv3p_r50_512/unet_bf1_backfill.json

Usage:
    python -m baseline_deeplab.eval_unet_bf1
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baseline_unet import config as base
from baseline_unet.dataset import DamSegmentDataset, build_transforms, read_split_file

from . import config as C
from .metrics import SegMetricsBF1, format_metrics


def build_unet() -> torch.nn.Module:
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=base.NUM_CLASSES,
    )


def pick_device() -> str:
    if base.DEVICE == "mps" and torch.backends.mps.is_available():
        return "mps"
    if base.DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def run_split(model, files, device, img_size):
    ds = DamSegmentDataset(base.DATA_ROOT, files, build_transforms(img_size, train=False))
    loader = DataLoader(ds, batch_size=base.BATCH_SIZE, shuffle=False, num_workers=2)
    m = SegMetricsBF1(base.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    m.reset()
    model.eval()
    for imgs, masks, _ in loader:
        imgs = imgs.to(device).float(); masks = masks.to(device).long()
        logits = model(imgs)
        m.update(logits, masks)
    return m.compute()


def main() -> None:
    device = pick_device()
    ckpt_path = base.RUN_DIR / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"U-Net best.pt not found at {ckpt_path}")

    print(f"[backfill] device={device}")
    print(f"[backfill] loading {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_unet().to(device)
    model.load_state_dict(state["model"])

    val_files = read_split_file(C.SPLIT_FILES["val"])
    test_files = read_split_file(C.SPLIT_FILES["test"])

    print(f"[backfill] evaluating at IMG_SIZE={base.IMG_SIZE} (Step 0 native)")
    val_m = run_split(model, val_files, device, base.IMG_SIZE)
    print("[backfill] VAL:")
    print(format_metrics(val_m))

    test_m = run_split(model, test_files, device, base.IMG_SIZE)
    print("[backfill] TEST:")
    print(format_metrics(test_m))

    out_dir = C.RUNS_DIR / "deeplabv3p_r50_512"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "unet_bf1_backfill.json"
    payload = {
        "checkpoint": str(ckpt_path),
        "eval_img_size": base.IMG_SIZE,
        "bf1_tol_px": C.BF1_TOLERANCE_PX,
        "best_epoch": state.get("epoch"),
        "val": {k: float(v) for k, v in val_m.items()},
        "test": {k: float(v) for k, v in test_m.items()},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[backfill] wrote {out_path}")


if __name__ == "__main__":
    main()

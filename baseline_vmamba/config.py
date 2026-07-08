"""VMamba baseline config.

Inherits data pipeline from baseline_unet. Training recipe mirrors the
SegFormer-B2 baseline (512, bs=4, 100 epochs, lr=6e-5) for fair comparison.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from baseline_unet import config as base

# ----- Shared (inherited) -----
DATA_ROOT = base.DATA_ROOT
NUM_CLASSES = base.NUM_CLASSES
CLASS_NAMES = base.CLASS_NAMES
SPLITS_DIR = base.SPLITS_DIR
SPLIT_FILES = {k: base.SPLITS_DIR / f"{k}.txt" for k in ("train", "val", "test")}

PKG_DIR = Path(__file__).resolve().parent
RUNS_DIR = PKG_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

LOSS_CE_W = base.LOSS_CE_W
LOSS_DICE_W = base.LOSS_DICE_W
CE_WEIGHTS = base.CE_WEIGHTS
SEED = base.SEED
DEVICE = base.DEVICE

BF1_TOLERANCE_PX = 2

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass
class RunCfg:
    name: str = "vmamba_tiny_512"
    img_size: int = 512
    batch_size: int = 4
    grad_accum: int = 1
    epochs: int = 100
    lr: float = 6e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    # VMamba-Tiny encoder config
    dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    depths: List[int] = field(default_factory=lambda: [2, 2, 9, 2])
    d_state: int = 16
    expand: int = 2
    d_conv: int = 3
    # Decoder
    decode_dim: int = 256
    # Pretrained weights (local path or None)
    pretrained_ckpt: str = None


PRESETS = {
    "T": RunCfg(name="vmamba_tiny_512"),
    "S": RunCfg(
        name="vmamba_small_512",
        depths=[2, 2, 27, 2],
    ),
}


def run_dir(cfg: RunCfg) -> Path:
    d = RUNS_DIR / cfg.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "samples").mkdir(parents=True, exist_ok=True)
    return d

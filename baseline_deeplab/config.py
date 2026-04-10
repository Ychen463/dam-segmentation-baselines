"""Step-1 DeepLabV3+ config: two presets (Run A / Run B).

Inherits data protocol (splits, class names, loss weights, device, seed) from
baseline_unet.config so that Step 0 and Step 1 share a byte-identical data
pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from baseline_unet import config as base

# ----- Shared (inherited) -----
DATA_ROOT = base.DATA_ROOT
NUM_CLASSES = base.NUM_CLASSES
CLASS_NAMES = base.CLASS_NAMES
SPLITS_DIR = base.SPLITS_DIR  # reuse Step 0 splits/ (byte-identical)
SPLIT_FILES = {k: base.SPLITS_DIR / f"{k}.txt" for k in ("train", "val", "test")}

PKG_DIR = Path(__file__).resolve().parent
RUNS_DIR = PKG_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ----- Loss / training defaults shared across presets -----
LOSS_CE_W = base.LOSS_CE_W
LOSS_DICE_W = base.LOSS_DICE_W
CE_WEIGHTS = base.CE_WEIGHTS  # manual fallback
SEED = base.SEED
DEVICE = base.DEVICE
ENCODER_WEIGHTS = "imagenet"

# Boundary F1 absolute pixel tolerance (see report footnote about cross-res).
BF1_TOLERANCE_PX = 2

# MPS op fallback (inherited from baseline_unet.config side-effect, set again for safety).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass
class RunCfg:
    name: str
    encoder: str         # resnet34 / resnet50
    img_size: int
    batch_size: int
    grad_accum: int
    epochs: int
    lr: float
    weight_decay: float = 1e-4


# Run A: practical high-resolution baseline
RUN_A = RunCfg(
    name="deeplabv3p_r50_512",
    encoder="resnet50",
    img_size=512,
    batch_size=4,
    grad_accum=1,
    epochs=50,
    lr=5e-4,
)

# Run B: head-only control vs Step 0 U-Net
RUN_B = RunCfg(
    name="deeplabv3p_r34_320",
    encoder="resnet34",
    img_size=320,
    batch_size=8,
    grad_accum=1,
    epochs=30,
    lr=1e-3,
)

PRESETS = {"A": RUN_A, "B": RUN_B}


def run_dir(cfg: RunCfg) -> Path:
    d = RUNS_DIR / cfg.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "samples").mkdir(parents=True, exist_ok=True)
    return d

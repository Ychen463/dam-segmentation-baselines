"""Mask2Former Swin-Small config — single plain run (no presets, no curriculum)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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

SEED = base.SEED
DEVICE = base.DEVICE

PRETRAINED_MODEL_ID = "facebook/mask2former-swin-small-ade-semantic"
BF1_TOLERANCE_PX = 2

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass
class RunCfg:
    name: str = "mask2former_swin_small_512"
    pretrained: str = PRETRAINED_MODEL_ID
    img_size: int = 512
    batch_size: int = 4
    grad_accum: int = 1
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5


CFG = RunCfg()


def run_dir(cfg: RunCfg) -> Path:
    d = RUNS_DIR / cfg.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "samples").mkdir(parents=True, exist_ok=True)
    return d

"""Mask2Former Swin-Small config — M0 plain baseline + M1 soft curriculum."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

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
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    # Curriculum controls (M1 only)
    use_soft_curriculum: bool = False
    no_curriculum: bool = True


# ----- Ablation presets -----
ABLATION_PRESETS: Dict[str, Dict] = {
    "M0": {"name": "mask2former_plain_M0",
           "no_curriculum": True, "use_soft_curriculum": False},
    "M1": {"name": "mask2former_softcurr_M1",
           "no_curriculum": False, "use_soft_curriculum": True},
}


def apply_preset(cfg: RunCfg, preset_key: str) -> None:
    """Apply an ablation preset to cfg (mutates in place)."""
    p = ABLATION_PRESETS[preset_key]
    for k, v in p.items():
        setattr(cfg, k, v)


CFG = RunCfg()


def run_dir(cfg: RunCfg) -> Path:
    d = RUNS_DIR / cfg.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "samples").mkdir(parents=True, exist_ok=True)
    return d

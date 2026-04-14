"""Mask2Former Swin-Small config — M0-M5 ablation presets."""
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
    # Curriculum controls (M1, M5)
    use_soft_curriculum: bool = False
    no_curriculum: bool = True
    # M2: Dynamic difficulty-aware loss weighting
    use_difficulty_weighting: bool = False
    diff_alpha: float = 1.0
    diff_beta: float = 0.5
    diff_gamma: float = 0.3
    diff_delta: float = 0.3
    diff_ema: float = 0.9
    loss_reweight_lambda: float = 0.5
    aux_ce_weight: float = 0.3
    # M3: clDice auxiliary loss
    use_cldice_loss: bool = False
    cldice_weight: float = 0.05
    cldice_start_epoch: int = 20
    cldice_iters: int = 7


# ----- Ablation presets -----
ABLATION_PRESETS: Dict[str, Dict] = {
    "M0": {"name": "mask2former_plain_M0",
           "no_curriculum": True, "use_soft_curriculum": False,
           "use_difficulty_weighting": False, "use_cldice_loss": False},
    "M1": {"name": "mask2former_softcurr_M1",
           "no_curriculum": False, "use_soft_curriculum": True,
           "use_difficulty_weighting": False, "use_cldice_loss": False},
    "M2": {"name": "mask2former_diffwt_M2",
           "no_curriculum": True, "use_soft_curriculum": False,
           "use_difficulty_weighting": True, "use_cldice_loss": False},
    "M3": {"name": "mask2former_cldice_M3",
           "no_curriculum": True, "use_soft_curriculum": False,
           "use_difficulty_weighting": False, "use_cldice_loss": True},
    "M4": {"name": "mask2former_diff_cldice_M4",
           "no_curriculum": True, "use_soft_curriculum": False,
           "use_difficulty_weighting": True, "use_cldice_loss": True},
    "M5": {"name": "mask2former_full_M5",
           "no_curriculum": False, "use_soft_curriculum": True,
           "use_difficulty_weighting": True, "use_cldice_loss": True},
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

"""Step-5 Full Method config: dynamic difficulty-aware curriculum + boundary refinement.

Inherits shared constants from baseline_unet.config.
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
SPLITS_DIR = base.SPLITS_DIR
SPLIT_FILES = {k: base.SPLITS_DIR / f"{k}.txt" for k in ("train", "val", "test")}

PKG_DIR = Path(__file__).resolve().parent
RUNS_DIR = PKG_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ----- Loss / training defaults -----
CE_WEIGHTS = base.CE_WEIGHTS  # manual fallback
SEED = base.SEED
DEVICE = base.DEVICE

# Boundary F1 absolute pixel tolerance.
BF1_TOLERANCE_PX = 2

# MPS op fallback.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass
class RunCfg:
    name: str = "segformer_b2_full_512"
    pretrained: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    img_size: int = 512
    batch_size: int = 4
    grad_accum: int = 1
    epochs: int = 100
    lr: float = 6e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Difficulty score weights (Module 1)
    diff_alpha: float = 1.0       # EMA loss
    diff_beta: float = 0.5        # uncertainty
    diff_gamma: float = 0.3       # boundary complexity
    diff_delta: float = 0.3       # sparsity
    diff_ema: float = 0.9         # EMA decay
    diff_tau: float = 0.5         # sampling temperature

    # Class-aware sampling bonus (Module 2)
    spalling_bonus: float = 0.3
    late_hard_crack_bonus: float = 0.4

    # Loss weights (Module 2+3)
    loss_ce_w: float = 0.5
    loss_dice_w: float = 0.5
    loss_tversky_alpha: float = 0.3   # FP weight (crack Tversky)
    loss_tversky_beta: float = 0.7    # FN weight (penalize missed crack)

    # Ablation switches
    use_dynamic_difficulty: bool = True       # dynamic scoring + softmax sampling
    use_class_sampling_bonus: bool = True     # spalling_bonus + late_hard_crack_bonus
    use_class_loss_schedule: bool = True      # scheduled crack_weight / boundary_weight
    use_boundary_loss: bool = True            # boundary BCE loss
    use_tversky_loss: bool = True             # crack Tversky loss


# ---------------------------------------------------------------------------
# Ablation presets
# ---------------------------------------------------------------------------

ABLATION_PRESETS = {
    # Set A: module-level cumulative ablation (all build on A1 static curriculum)
    "A2": {"name": "ablation_A2_dynamic_refinement",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": False,
           "use_class_loss_schedule": False, "use_boundary_loss": False,
           "use_tversky_loss": False},
    "A3": {"name": "ablation_A3_dynamic_classaware",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": False,
           "use_tversky_loss": False},
    "A3a": {"name": "ablation_A3a_sampling_bonus_only",
            "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
            "use_class_loss_schedule": False, "use_boundary_loss": False,
            "use_tversky_loss": False},
    "A3b": {"name": "ablation_A3b_loss_schedule_only",
            "use_dynamic_difficulty": True, "use_class_sampling_bonus": False,
            "use_class_loss_schedule": True, "use_boundary_loss": False,
            "use_tversky_loss": False},
    "A4": {"name": "ablation_A4_classaware_boundary",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": False},
    "A5": {"name": "ablation_A5_full",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True},
    # Set B: difficulty score mechanism ablation (all modules ON, vary diff_* weights)
    "B0": {"name": "ablation_B0_diff_loss",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.0, "diff_gamma": 0.0, "diff_delta": 0.0},
    "B1": {"name": "ablation_B1_diff_loss_uncert",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.5, "diff_gamma": 0.0, "diff_delta": 0.0},
    "B2": {"name": "ablation_B2_diff_loss_boundary",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.0, "diff_gamma": 0.3, "diff_delta": 0.0},
    "B3": {"name": "ablation_B3_diff_loss_sparsity",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.0, "diff_gamma": 0.0, "diff_delta": 0.3},
    "B4": {"name": "ablation_B4_diff_loss_uncert_sparsity",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.5, "diff_gamma": 0.0, "diff_delta": 0.3},
    "B5": {"name": "ablation_B5_diff_all",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.5, "diff_gamma": 0.3, "diff_delta": 0.3},
}


def apply_preset(cfg: RunCfg, preset_name: str) -> None:
    """Apply a named ablation preset to *cfg* in-place."""
    if preset_name not in ABLATION_PRESETS:
        raise ValueError(
            f"Unknown ablation preset '{preset_name}'. "
            f"Available: {sorted(ABLATION_PRESETS.keys())}"
        )
    for key, val in ABLATION_PRESETS[preset_name].items():
        setattr(cfg, key, val)


def run_dir(cfg: RunCfg) -> Path:
    d = RUNS_DIR / cfg.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "samples").mkdir(parents=True, exist_ok=True)
    return d

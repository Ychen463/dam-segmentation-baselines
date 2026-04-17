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
    use_class_sampling_bonus: bool = False    # spalling_bonus + late_hard_crack_bonus
    use_class_loss_schedule: bool = False     # scheduled crack_weight / boundary_weight
    use_boundary_loss: bool = True            # boundary BCE loss
    use_tversky_loss: bool = False            # crack Tversky loss (off by default)

    # Soft curriculum (Change 1): smooth tier mixing instead of hard stage gates
    use_soft_curriculum: bool = True

    # Loss reweighting (Change 2): weight loss by difficulty instead of softmax sampling
    use_softmax_sampling: bool = False        # legacy softmax sampling (off by default)
    use_dynamic_loss_reweight: bool = True
    loss_reweight_lambda: float = 0.5         # w_i = 1 + λ * norm(d_i)

    # Late boundary (Change 4): ramp boundary loss in from boundary_start_ratio
    use_soft_boundary_schedule: bool = True
    boundary_start_ratio: float = 0.6
    boundary_max_weight: float = 0.10

    # Crack-only soft-clDice (topology-preserving)
    use_cldice_loss: bool = False
    cldice_weight: float = 0.15
    cldice_start_epoch: int = 40
    cldice_iters: int = 7

    # Skeleton Recall Loss (SRL) — Kirchhoff et al., ECCV 2024
    # Lighter alternative to clDice: only GT skeleton recall, no online skeletonization.
    # Shares cldice_weight and cldice_start_epoch for scheduling.
    use_srl_loss: bool = False

    # No-curriculum mode: fully uniform sampling over all samples
    no_curriculum: bool = False

    # Competence-based curriculum (Platanios et al. 2019)
    use_competence_curriculum: bool = False       # C1: hard unlock
    use_competence_soft_mixing: bool = False      # C2: soft weight mixing
    competence_c0: float = 0.333                  # initial competence
    competence_duration: int = 70                 # epochs to reach c=1.0
    competence_floor_easy: float = 0.05           # C2: Easy replay floor
    competence_floor_medium: float = 0.02         # C2: Medium replay floor
    competence_floor_hard: float = 0.00           # C2: Hard replay floor (0 = no early replay)


# ---------------------------------------------------------------------------
# Ablation presets
# ---------------------------------------------------------------------------

# Backward-compat keys shared by all legacy presets (A/B series)
_LEGACY_COMPAT = {
    "use_soft_curriculum": False, "use_softmax_sampling": True,
    "use_dynamic_loss_reweight": False, "use_soft_boundary_schedule": False,
    "use_cldice_loss": False,
    "loss_tversky_alpha": 0.3, "loss_tversky_beta": 0.7,
}

ABLATION_PRESETS = {
    # Set A: module-level cumulative ablation (all build on A1 static curriculum)
    "A2": {**_LEGACY_COMPAT, "name": "ablation_A2_dynamic_refinement",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": False,
           "use_class_loss_schedule": False, "use_boundary_loss": False,
           "use_tversky_loss": False},
    "A3": {**_LEGACY_COMPAT, "name": "ablation_A3_dynamic_classaware",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": False,
           "use_tversky_loss": False},
    "A3a": {**_LEGACY_COMPAT, "name": "ablation_A3a_sampling_bonus_only",
            "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
            "use_class_loss_schedule": False, "use_boundary_loss": False,
            "use_tversky_loss": False},
    "A3b": {**_LEGACY_COMPAT, "name": "ablation_A3b_loss_schedule_only",
            "use_dynamic_difficulty": True, "use_class_sampling_bonus": False,
            "use_class_loss_schedule": True, "use_boundary_loss": False,
            "use_tversky_loss": False},
    "A4": {**_LEGACY_COMPAT, "name": "ablation_A4_classaware_boundary",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": False},
    "A5": {**_LEGACY_COMPAT, "name": "ablation_A5_full",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True},
    # Set B: difficulty score mechanism ablation (all modules ON, vary diff_* weights)
    "B0": {**_LEGACY_COMPAT, "name": "ablation_B0_diff_loss",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.0, "diff_gamma": 0.0, "diff_delta": 0.0},
    "B1": {**_LEGACY_COMPAT, "name": "ablation_B1_diff_loss_uncert",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.5, "diff_gamma": 0.0, "diff_delta": 0.0},
    "B2": {**_LEGACY_COMPAT, "name": "ablation_B2_diff_loss_boundary",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.0, "diff_gamma": 0.3, "diff_delta": 0.0},
    "B3": {**_LEGACY_COMPAT, "name": "ablation_B3_diff_loss_sparsity",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.0, "diff_gamma": 0.0, "diff_delta": 0.3},
    "B4": {**_LEGACY_COMPAT, "name": "ablation_B4_diff_loss_uncert_sparsity",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.5, "diff_gamma": 0.0, "diff_delta": 0.3},
    "B5": {**_LEGACY_COMPAT, "name": "ablation_B5_diff_all",
           "use_dynamic_difficulty": True, "use_class_sampling_bonus": True,
           "use_class_loss_schedule": True, "use_boundary_loss": True,
           "use_tversky_loss": True,
           "diff_alpha": 1.0, "diff_beta": 0.5, "diff_gamma": 0.3, "diff_delta": 0.3},

    # Set S: stabilized presets (soft curriculum + optional clDice)
    "S1": {"name": "stabilized_S1_softcurr",
           "use_soft_curriculum": True, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False, "use_soft_boundary_schedule": False},
    "S2": {"name": "stabilized_S2_softcurr_cldice",
           "use_soft_curriculum": True, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},

    # Set C: competence-based curriculum (Algorithm V3)
    "C1": {"name": "competence_hard_C1",
           "no_curriculum": False,
           "use_competence_curriculum": True,
           "competence_c0": 0.333, "competence_duration": 70,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False, "use_soft_boundary_schedule": False},
    "C2": {"name": "competence_soft_C2",
           "no_curriculum": False,
           "use_competence_soft_mixing": True,
           "competence_c0": 0.333, "competence_duration": 70,
           "competence_floor_easy": 0.05,
           "competence_floor_medium": 0.02,
           "competence_floor_hard": 0.00,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False, "use_soft_boundary_schedule": False},

    # Set P: 2x2 experiment matrix (Algorithm V2 main line)
    "P0": {"name": "plain_segformer_P0",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False, "use_soft_boundary_schedule": False},
    "P1": {"name": "plain_cldice_P1",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},

    # Set D: difficulty weighting ablation (matches Mask2Former M2/M4)
    "D0": {"name": "difficulty_only_D0",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": True, "use_dynamic_loss_reweight": True,
           "loss_reweight_lambda": 0.5,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False, "use_soft_boundary_schedule": False},
    "D1": {"name": "difficulty_cldice_D1",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": True, "use_dynamic_loss_reweight": True,
           "loss_reweight_lambda": 0.5,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},

    # Set D v2: difficulty → softmax sampling (not loss reweight)
    "D0v2": {"name": "difficulty_sampling_D0v2",
             "no_curriculum": True,
             "use_soft_curriculum": False,
             "use_softmax_sampling": True,
             "use_dynamic_difficulty": True,
             "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False, "use_soft_boundary_schedule": False},
    "D1v2": {"name": "difficulty_sampling_cldice_D1v2",
             "no_curriculum": True,
             "use_soft_curriculum": False,
             "use_softmax_sampling": True,
             "use_dynamic_difficulty": True,
             "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},

    # Set F v2: full method with softmax sampling
    "F1v2": {"name": "full_method_F1v2",
             "no_curriculum": False,
             "use_competence_soft_mixing": True,
             "competence_c0": 0.333, "competence_duration": 70,
             "competence_floor_easy": 0.05,
             "competence_floor_medium": 0.02,
             "competence_floor_hard": 0.00,
             "use_soft_curriculum": False,
             "use_softmax_sampling": True,
             "use_dynamic_difficulty": True,
             "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},

    # Set D v3: difficulty + SRL (replacing clDice with Skeleton Recall Loss)
    "D1v3": {"name": "difficulty_srl_D1v3",
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": True, "use_dynamic_loss_reweight": True,
             "loss_reweight_lambda": 0.5,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},

    # Set P v3: plain + SRL
    "P1v3": {"name": "plain_srl_P1v3",
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},

    # Set F v3: full method with SRL instead of clDice
    "F1v3": {"name": "full_method_srl_F1v3",
             "no_curriculum": False,
             "use_competence_soft_mixing": True,
             "competence_c0": 0.333, "competence_duration": 70,
             "competence_floor_easy": 0.05,
             "competence_floor_medium": 0.02,
             "competence_floor_hard": 0.00,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": True, "use_dynamic_loss_reweight": True,
             "loss_reweight_lambda": 0.5,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},

    # Set F: full method (C2 curriculum + difficulty + clDice)
    "F1": {"name": "full_method_F1",
           "no_curriculum": False,
           "use_competence_soft_mixing": True,
           "competence_c0": 0.333, "competence_duration": 70,
           "competence_floor_easy": 0.05,
           "competence_floor_medium": 0.02,
           "competence_floor_hard": 0.00,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": True, "use_dynamic_loss_reweight": True,
           "loss_reweight_lambda": 0.5,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},
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

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

    # Snake branch auxiliary loss
    use_snake_aux_loss: bool = False
    snake_aux_weight: float = 0.10

    # Model architecture
    model_type: str = "segformer"            # "segformer" or "dscformer"
    snake_channels: int = 64                 # crack branch hidden dim
    snake_kernel_size: int = 9               # DSConv kernel size

    # EMA model averaging
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_start_epoch: int = 5        # warmup 期间不更新 EMA

    # Lovász-Softmax loss (IoU surrogate, replaces Dice for fg classes)
    use_lovasz_loss: bool = False
    lovasz_weight: float = 0.5      # replaces loss_dice_w when active

    # OHEM CE loss (keep only top-k% hardest pixels)
    use_ohem: bool = False
    ohem_ratio: float = 0.25        # keep top 25% hardest pixels

    # Augmentation level ("basic", "moderate", or "strong")
    aug_level: str = "basic"

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

    # Morphology-Aware Curriculum (MAC)
    use_mac: bool = False                         # master MAC switch
    use_mac_morph_difficulty: bool = False         # replace generic difficulty with morph features
    use_mac_adaptive_pacing: bool = False          # validation-driven stage transitions
    use_mac_class_loss: bool = False               # per-class IoU gap → CE weight adjustment
    mac_morph_width_w: float = 0.4                # weight for 1/width in morph difficulty
    mac_morph_topo_w: float = 0.3                 # weight for junction_density
    mac_morph_prox_w: float = 0.2                 # weight for crack_spalling_proximity
    mac_morph_sparse_w: float = 0.1               # weight for log(components+1)
    mac_diff_gamma: float = 0.5                   # weight for morph_difficulty in final score
    mac_adaptive_patience: int = 3                # stagnation count before forced promotion
    mac_adaptive_min_epochs: int = 10             # minimum epochs per stage
    mac_class_loss_max_boost: float = 2.0         # maximum CE weight multiplier
    mac_class_loss_ema: float = 0.8               # EMA smoothing for class loss scheduler


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

    # Set G: DSCformerDam (SegFormer + Dynamic Snake Conv crack branch)
    "G0": {"name": "dscformer_plain_G0",
           "model_type": "dscformer",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False, "use_srl_loss": False,
           "use_soft_boundary_schedule": False},
    "G1": {"name": "dscformer_srl_G1",
           "model_type": "dscformer",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False,
           "use_srl_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},
    "G2": {"name": "dscformer_full_G2",
           "model_type": "dscformer",
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

    # Set H: DSCformerDam + snake aux loss (vs G series)
    "H0": {"name": "dscformer_aux_H0",
           "model_type": "dscformer",
           "use_snake_aux_loss": True, "snake_aux_weight": 0.10,
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False, "use_srl_loss": False,
           "use_soft_boundary_schedule": False},
    "H1": {"name": "dscformer_aux_srl_H1",
           "model_type": "dscformer",
           "use_snake_aux_loss": True, "snake_aux_weight": 0.10,
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False,
           "use_srl_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},
    "H2": {"name": "dscformer_aux_full_H2",
           "model_type": "dscformer",
           "use_snake_aux_loss": True, "snake_aux_weight": 0.10,
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

    # Set M: Morphology-Aware Curriculum (MAC) ablation on DSCformer + SRL base
    # Base: DSCformer + SRL (G1), no old curriculum, dynamic difficulty ON
    "M0": {"name": "mac_morph_M0",
           "model_type": "dscformer",
           "use_mac": True,
           "use_mac_morph_difficulty": True,
           "use_mac_adaptive_pacing": False,
           "use_mac_class_loss": False,
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
    "M1": {"name": "mac_morph_pacing_M1",
           "model_type": "dscformer",
           "use_mac": True,
           "use_mac_morph_difficulty": True,
           "use_mac_adaptive_pacing": True,
           "use_mac_class_loss": False,
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
    "M2": {"name": "mac_morph_classloss_M2",
           "model_type": "dscformer",
           "use_mac": True,
           "use_mac_morph_difficulty": True,
           "use_mac_adaptive_pacing": False,
           "use_mac_class_loss": True,
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
    "M3": {"name": "mac_full_M3",
           "model_type": "dscformer",
           "use_mac": True,
           "use_mac_morph_difficulty": True,
           "use_mac_adaptive_pacing": True,
           "use_mac_class_loss": True,
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

    # Set N: data-driven optimizations (orthogonal to curriculum, based on G1)
    # N0 = G1 reproduction (sanity check)
    "N0": {"name": "dataopt_baseline_N0",
           "model_type": "dscformer",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False,
           "use_srl_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},
    # N1 = G1 + strong augmentation
    "N1": {"name": "dataopt_strongaug_N1",
           "model_type": "dscformer",
           "aug_level": "strong",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False,
           "use_srl_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},
    # N2 = N1 + SRL starts at epoch 30
    "N2": {"name": "dataopt_earlysrl_N2",
           "model_type": "dscformer",
           "aug_level": "strong",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False,
           "use_srl_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 30, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},
    # N3 = N2 + SRL weight 0.10 + Tversky loss
    "N3": {"name": "dataopt_full_N3",
           "model_type": "dscformer",
           "aug_level": "strong",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False,
           "use_tversky_loss": True, "loss_tversky_alpha": 0.3, "loss_tversky_beta": 0.7,
           "use_cldice_loss": False,
           "use_srl_loss": True, "cldice_weight": 0.10,
           "cldice_start_epoch": 30, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},
    # MN3 = M3 (full MAC) + N3 data optimizations
    "MN3": {"name": "mac_dataopt_MN3",
            "model_type": "dscformer",
            "aug_level": "strong",
            "use_mac": True,
            "use_mac_morph_difficulty": True,
            "use_mac_adaptive_pacing": True,
            "use_mac_class_loss": True,
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
            "use_boundary_loss": False,
            "use_tversky_loss": True, "loss_tversky_alpha": 0.3, "loss_tversky_beta": 0.7,
            "use_cldice_loss": False,
            "use_srl_loss": True, "cldice_weight": 0.10,
            "cldice_start_epoch": 30, "cldice_iters": 7,
            "use_soft_boundary_schedule": False},

    # Set N round 13: decoupled ablation (orthogonal to strong aug)
    # N1m = N0 + moderate augmentation (no elastic)
    "N1m": {"name": "dataopt_modaug_N1m",
            "model_type": "dscformer",
            "aug_level": "moderate",
            "no_curriculum": True,
            "use_soft_curriculum": False, "use_softmax_sampling": False,
            "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
            "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
            "use_boundary_loss": False, "use_tversky_loss": False,
            "use_cldice_loss": False,
            "use_srl_loss": True, "cldice_weight": 0.05,
            "cldice_start_epoch": 60, "cldice_iters": 7,
            "use_soft_boundary_schedule": False},
    # N2a = N0 + early SRL (epoch 30, no aug change)
    "N2a": {"name": "dataopt_earlysrl_N2a",
            "model_type": "dscformer",
            "no_curriculum": True,
            "use_soft_curriculum": False, "use_softmax_sampling": False,
            "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
            "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
            "use_boundary_loss": False, "use_tversky_loss": False,
            "use_cldice_loss": False,
            "use_srl_loss": True, "cldice_weight": 0.05,
            "cldice_start_epoch": 30, "cldice_iters": 7,
            "use_soft_boundary_schedule": False},
    # N3a = N2a + SRL weight 0.10 + Tversky (no aug change)
    "N3a": {"name": "dataopt_lossopts_N3a",
            "model_type": "dscformer",
            "no_curriculum": True,
            "use_soft_curriculum": False, "use_softmax_sampling": False,
            "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
            "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
            "use_boundary_loss": False,
            "use_tversky_loss": True, "loss_tversky_alpha": 0.3, "loss_tversky_beta": 0.7,
            "use_cldice_loss": False,
            "use_srl_loss": True, "cldice_weight": 0.10,
            "cldice_start_epoch": 30, "cldice_iters": 7,
            "use_soft_boundary_schedule": False},
    # N4a = N1m (moderate aug) + N3a (loss improvements)
    "N4a": {"name": "dataopt_combined_N4a",
            "model_type": "dscformer",
            "aug_level": "moderate",
            "no_curriculum": True,
            "use_soft_curriculum": False, "use_softmax_sampling": False,
            "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
            "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
            "use_boundary_loss": False,
            "use_tversky_loss": True, "loss_tversky_alpha": 0.3, "loss_tversky_beta": 0.7,
            "use_cldice_loss": False,
            "use_srl_loss": True, "cldice_weight": 0.10,
            "cldice_start_epoch": 30, "cldice_iters": 7,
            "use_soft_boundary_schedule": False},
    # MN4 = M3 (full MAC) + N4a data optimizations
    "MN4": {"name": "mac_dataopt_MN4",
            "model_type": "dscformer",
            "aug_level": "moderate",
            "use_mac": True,
            "use_mac_morph_difficulty": True,
            "use_mac_adaptive_pacing": True,
            "use_mac_class_loss": True,
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
            "use_boundary_loss": False,
            "use_tversky_loss": True, "loss_tversky_alpha": 0.3, "loss_tversky_beta": 0.7,
            "use_cldice_loss": False,
            "use_srl_loss": True, "cldice_weight": 0.10,
            "cldice_start_epoch": 30, "cldice_iters": 7,
            "use_soft_boundary_schedule": False},

    # Set E: Round 14 — EMA + Lovász quick gains (based on N0 = DSCformer + SRL)
    # E1: N0 + EMA only
    "E1": {"name": "ema_only_E1",
           "model_type": "dscformer",
           "use_ema": True,
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False,
           "use_srl_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},
    # E2: N0 + Lovász only (replaces Dice with Lovász-Softmax)
    "E2": {"name": "lovasz_only_E2",
           "model_type": "dscformer",
           "use_lovasz_loss": True,
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False,
           "use_srl_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},
    # E3: N0 + EMA + Lovász
    "E3": {"name": "ema_lovasz_E3",
           "model_type": "dscformer",
           "use_ema": True,
           "use_lovasz_loss": True,
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False,
           "use_srl_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False},

    # Set R15: Round 15 — capacity-first improvements (based on N0)
    # R15a: native resolution 640×640
    "R15a": {"name": "r15_hires_640",
             "model_type": "dscformer", "img_size": 640,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # R15b: SegFormer-B5 backbone at 512
    "R15b": {"name": "r15_b5_512",
             "model_type": "dscformer",
             "pretrained": "nvidia/segformer-b5-finetuned-ade-640-640",
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # R15d: OHEM CE loss (top 25% hardest pixels)
    "R15d": {"name": "r15_ohem_R15d",
             "model_type": "dscformer",
             "use_ohem": True, "ohem_ratio": 0.25,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # R15e: lr=3e-5
    "R15e": {"name": "r15_lr3e5_R15e",
             "model_type": "dscformer", "lr": 3e-5,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # R15f: lr=1e-4
    "R15f": {"name": "r15_lr1e4_R15f",
             "model_type": "dscformer", "lr": 1e-4,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
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

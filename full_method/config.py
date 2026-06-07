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
    model_type: str = "segformer"            # "segformer", "dscformer", "sam_lora", or "dinov2_lora"
    snake_channels: int = 64                 # crack branch hidden dim
    snake_kernel_size: int = 9               # DSConv kernel size (single-scale)
    use_multiscale_snake: bool = False       # Multi-Scale DSConv (E1 preset)
    snake_kernel_sizes: tuple = (5, 9, 15)   # kernel sizes for multi-scale branch

    # SAM LoRA settings (model_type="sam_lora")
    sam_checkpoint: str = "sam_vit_b_01ec64.pth"   # path to SAM ViT-B weights
    lora_rank: int = 16
    lora_alpha: float = 16.0
    sam_img_size: int = 1024
    sam_fpn_dim: int = 256
    sam_lr_lora: float = 1e-4               # LR for LoRA params
    sam_lr_decoder: float = 3e-4            # LR for decoder params

    # DINOv2 LoRA settings (model_type="dinov2_lora")
    dinov2_img_size: int = 518              # multiple of patch_size=14 (37*14=518)
    dinov2_fpn_dim: int = 256

    # CALoRA settings (model_type="calora_sam")
    calora_ranks: tuple = (8, 32)           # MoE-LoRA expert ranks (low, high)
    calora_router_hidden: int = 64          # router MLP hidden dimension
    calora_router_aux_weight: float = 0.1   # auxiliary routing loss weight (0=disabled)

    # Knowledge Distillation (self-distillation from teacher checkpoint)
    use_kd: bool = False
    kd_teacher_checkpoint: str = ""         # path to teacher best.pt
    kd_alpha: float = 0.5                   # weight for hard loss (1-alpha for KD loss)
    kd_temperature: float = 4.0             # softmax temperature for KD

    # Dual-Teacher KD (complementary distillation from task-specific + foundation model)
    use_dual_kd: bool = False
    kd_teacher2_checkpoint: str = ""        # path to 2nd teacher best.pt
    kd_teacher2_model_type: str = "sam_lora"  # model type for 2nd teacher
    kd_t1_weight: float = 0.5              # weight for teacher 1 in ensemble
    kd_t2_weight: float = 0.5              # weight for teacher 2 in ensemble
    # Class-conditional weighting: per-class [teacher1_w, teacher2_w]
    kd_class_weights: bool = False          # use per-class teacher weights
    kd_crack_t2_weight: float = 0.6        # SAM2 weight for crack class (higher = more SAM2)
    kd_spalling_t2_weight: float = 0.3     # SAM2 weight for spalling class (lower = more G1)

    # Skeleton-Distance Weighted Loss (SDWL)
    use_sdwl: bool = False               # enable skeleton-distance weighted CE
    sdwl_start_epoch: int = 30           # start SDWL after this epoch
    sdwl_w_skel: float = 3.0            # weight for skeleton pixels
    sdwl_w_near: float = 2.0            # weight for near-skeleton pixels
    sdwl_near_radius: int = 3           # "near skeleton" radius in pixels
    sdwl_w_bg_near: float = 1.5         # weight for background near crack boundary

    # Progressive Loss Curriculum phases
    # Phase 1: CE + Dice (basic segmentation)
    # Phase 2: + SDWL (focus on skeleton topology)
    # Phase 3: + SRL (fine-tune connectivity)
    use_progressive_loss: bool = False   # enable progressive loss scheduling
    progressive_phase2_epoch: int = 30   # start Phase 2 (SDWL)
    progressive_phase3_epoch: int = 60   # start Phase 3 (SRL)

    # Dual-Task Skeleton Consistency (G3/G4 preset — Plan A)
    use_skeleton_head: bool = False           # add skeleton prediction branch
    skeleton_head_hidden: int = 128           # hidden dim for skeleton head
    skel_pred_weight: float = 0.10           # weight for skeleton prediction loss
    skel_consist_weight: float = 0.05        # weight for skeleton consistency loss
    skel_consist_start_epoch: int = 40       # start consistency loss after this epoch

    # Attention-Gated DSConv Fusion (G4 preset — Plan B)
    use_attention_gate: bool = False          # replace additive fusion with learned gate

    # Contrastive Skeleton Learning (G4 preset — Plan C)
    use_contrastive: bool = False             # add contrastive projection head
    contrastive_dim: int = 64                 # projection dimension
    contrastive_weight: float = 0.05         # loss weight
    contrastive_temperature: float = 0.1     # InfoNCE temperature
    contrastive_n_samples: int = 256         # pixels sampled per class
    contrastive_start_epoch: int = 30        # start contrastive after this epoch

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

    # G4: DSCformerDam + SRL + Plans A+B+C (revised weights after G3 analysis)
    # Plan A: reduced weight + delayed start (G3 showed high weights hurt ConnR)
    # Plan B: Attention-Gated DSConv Fusion
    # Plan C: Contrastive Skeleton Learning
    "G4": {"name": "dscformer_full_G4",
           "model_type": "dscformer",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False,
           "use_srl_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False,
           # Plan A: Skeleton Head + Consistency (reduced from G3)
           "use_skeleton_head": True,
           "skeleton_head_hidden": 128,
           "skel_pred_weight": 0.03,            # was 0.10, reduced to avoid gradient conflict
           "skel_consist_weight": 0.02,         # was 0.05, reduced
           "skel_consist_start_epoch": 70,      # was 40, delayed until model is mature
           # Plan B: Attention Gate
           "use_attention_gate": True,
           # Plan C: Contrastive Learning
           "use_contrastive": True,
           "contrastive_dim": 64,
           "contrastive_weight": 0.03,          # was 0.05, slightly reduced
           "contrastive_temperature": 0.1,
           "contrastive_n_samples": 256,
           "contrastive_start_epoch": 30},

    # G4b: DSCformerDam + SRL + Plans B+C only (no skeleton head)
    # Ablation to isolate whether Attention Gate + Contrastive help without Plan A
    "G4b": {"name": "dscformer_attn_contrast_G4b",
            "model_type": "dscformer",
            "no_curriculum": True,
            "use_soft_curriculum": False, "use_softmax_sampling": False,
            "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
            "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
            "use_boundary_loss": False, "use_tversky_loss": False,
            "use_cldice_loss": False,
            "use_srl_loss": True, "cldice_weight": 0.05,
            "cldice_start_epoch": 60, "cldice_iters": 7,
            "use_soft_boundary_schedule": False,
            # No Plan A
            "use_skeleton_head": False,
            # Plan B: Attention Gate
            "use_attention_gate": True,
            # Plan C: Contrastive Learning
            "use_contrastive": True,
            "contrastive_dim": 64,
            "contrastive_weight": 0.03,
            "contrastive_temperature": 0.1,
            "contrastive_n_samples": 256,
            "contrastive_start_epoch": 30},

    # G3: DSCformerDam + SRL + Dual-Task Skeleton Consistency
    # Adds a skeleton prediction head + consistency loss for novelty
    "G3": {"name": "dscformer_skelconsist_G3",
           "model_type": "dscformer",
           "no_curriculum": True,
           "use_soft_curriculum": False, "use_softmax_sampling": False,
           "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
           "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
           "use_boundary_loss": False, "use_tversky_loss": False,
           "use_cldice_loss": False,
           "use_srl_loss": True, "cldice_weight": 0.05,
           "cldice_start_epoch": 60, "cldice_iters": 7,
           "use_soft_boundary_schedule": False,
           "use_skeleton_head": True,
           "skeleton_head_hidden": 128,
           "skel_pred_weight": 0.10,
           "skel_consist_weight": 0.05,
           "skel_consist_start_epoch": 40},

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

    # Set MS: Multi-Scale DSConv experiments
    # MS1: Multi-Scale DSConv (kernel 5/9/15) + SRL — architectural novelty
    "MS1": {"name": "dscformer_multiscale_MS1",
            "model_type": "dscformer",
            "use_multiscale_snake": True,
            "snake_kernel_sizes": (5, 9, 15),
            "snake_channels": 64,
            "no_curriculum": True,
            "use_soft_curriculum": False, "use_softmax_sampling": False,
            "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
            "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
            "use_boundary_loss": False, "use_tversky_loss": False,
            "use_cldice_loss": False,
            "use_srl_loss": True, "cldice_weight": 0.05,
            "cldice_start_epoch": 60, "cldice_iters": 7,
            "use_soft_boundary_schedule": False},

    # Set PL: Progressive Loss + SDWL experiments
    # PL1: DSCformer + SDWL only (no SRL) — test SDWL in isolation
    "PL1": {"name": "dscformer_sdwl_PL1",
            "model_type": "dscformer",
            "no_curriculum": True,
            "use_soft_curriculum": False, "use_softmax_sampling": False,
            "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
            "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
            "use_boundary_loss": False, "use_tversky_loss": False,
            "use_cldice_loss": False, "use_srl_loss": False,
            "use_soft_boundary_schedule": False,
            "use_sdwl": True,
            "sdwl_start_epoch": 30,
            "sdwl_w_skel": 3.0, "sdwl_w_near": 2.0,
            "sdwl_near_radius": 3, "sdwl_w_bg_near": 1.5},
    # PL2: DSCformer + Progressive Loss (SDWL phase2 + SRL phase3)
    "PL2": {"name": "dscformer_progressive_PL2",
            "model_type": "dscformer",
            "no_curriculum": True,
            "use_soft_curriculum": False, "use_softmax_sampling": False,
            "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
            "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
            "use_boundary_loss": False, "use_tversky_loss": False,
            "use_cldice_loss": False,
            "use_soft_boundary_schedule": False,
            "use_sdwl": True,
            "sdwl_start_epoch": 30,
            "sdwl_w_skel": 3.0, "sdwl_w_near": 2.0,
            "sdwl_near_radius": 3, "sdwl_w_bg_near": 1.5,
            "use_srl_loss": True, "cldice_weight": 0.05,
            "cldice_start_epoch": 60, "cldice_iters": 7,
            "use_progressive_loss": True,
            "progressive_phase2_epoch": 30,
            "progressive_phase3_epoch": 60},
    # PL2b: PL2 with reduced SDWL weights (avoid early-stop instability)
    "PL2b": {"name": "dscformer_progressive_PL2b",
             "model_type": "dscformer",
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_soft_boundary_schedule": False,
             "use_sdwl": True,
             "sdwl_start_epoch": 30,
             "sdwl_w_skel": 2.0, "sdwl_w_near": 1.5,
             "sdwl_near_radius": 3, "sdwl_w_bg_near": 1.2,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_progressive_loss": True,
             "progressive_phase2_epoch": 30,
             "progressive_phase3_epoch": 60},
    # PL3: DSCformer + SDWL + SRL (both from epoch 30, no progressive)
    "PL3": {"name": "dscformer_sdwl_srl_PL3",
            "model_type": "dscformer",
            "no_curriculum": True,
            "use_soft_curriculum": False, "use_softmax_sampling": False,
            "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
            "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
            "use_boundary_loss": False, "use_tversky_loss": False,
            "use_cldice_loss": False,
            "use_soft_boundary_schedule": False,
            "use_sdwl": True,
            "sdwl_start_epoch": 30,
            "sdwl_w_skel": 3.0, "sdwl_w_near": 2.0,
            "sdwl_near_radius": 3, "sdwl_w_bg_near": 1.5,
            "use_srl_loss": True, "cldice_weight": 0.05,
            "cldice_start_epoch": 30, "cldice_iters": 7},

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

    # Set SAM: TopoLoRA-SAM (frozen SAM ViT-B + LoRA + FPN decoder)
    # SAM1: SAM LoRA baseline (CE + Dice, no SRL)
    "SAM1": {"name": "sam_lora_SAM1",
             "model_type": "sam_lora",
             "epochs": 50, "batch_size": 2,
             "lora_rank": 16, "lora_alpha": 16.0,
             "sam_img_size": 1024, "sam_fpn_dim": 256,
             "sam_lr_lora": 1e-4, "sam_lr_decoder": 3e-4,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False, "use_srl_loss": False,
             "use_soft_boundary_schedule": False},
    # SAM2: SAM LoRA + SRL (topology-aware)
    "SAM2": {"name": "sam_lora_srl_SAM2",
             "model_type": "sam_lora",
             "epochs": 50, "batch_size": 2,
             "lora_rank": 16, "lora_alpha": 16.0,
             "sam_img_size": 1024, "sam_fpn_dim": 256,
             "sam_lr_lora": 1e-4, "sam_lr_decoder": 3e-4,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 20, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},

    # Set DINO: DINOv2-LoRA (frozen DINOv2 ViT-B/14 + LoRA + FPN decoder)
    # DINO1: DINOv2 LoRA baseline (CE + Dice, no SRL)
    "DINO1": {"name": "dinov2_lora_DINO1",
              "model_type": "dinov2_lora",
              "epochs": 50, "batch_size": 2,
              "lora_rank": 16, "lora_alpha": 16.0,
              "dinov2_img_size": 518, "dinov2_fpn_dim": 256,
              "sam_lr_lora": 1e-4, "sam_lr_decoder": 3e-4,
              "no_curriculum": True,
              "use_soft_curriculum": False, "use_softmax_sampling": False,
              "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
              "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
              "use_boundary_loss": False, "use_tversky_loss": False,
              "use_cldice_loss": False, "use_srl_loss": False,
              "use_soft_boundary_schedule": False},
    # DINO2: DINOv2 LoRA + SRL (topology-aware)
    "DINO2": {"name": "dinov2_lora_srl_DINO2",
              "model_type": "dinov2_lora",
              "epochs": 50, "batch_size": 2,
              "lora_rank": 16, "lora_alpha": 16.0,
              "dinov2_img_size": 518, "dinov2_fpn_dim": 256,
              "sam_lr_lora": 1e-4, "sam_lr_decoder": 3e-4,
              "no_curriculum": True,
              "use_soft_curriculum": False, "use_softmax_sampling": False,
              "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
              "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
              "use_boundary_loss": False, "use_tversky_loss": False,
              "use_cldice_loss": False,
              "use_srl_loss": True, "cldice_weight": 0.05,
              "cldice_start_epoch": 20, "cldice_iters": 7,
              "use_soft_boundary_schedule": False},

    # Set KD: Knowledge Distillation (self-distillation from G1 teacher)
    # KD1: Born-Again G1 (same arch, distilled from G1, α=0.5, T=4)
    "KD1": {"name": "kd_born_again_KD1",
            "model_type": "dscformer",
            "use_kd": True,
            "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
            "kd_alpha": 0.5, "kd_temperature": 4.0,
            "no_curriculum": True,
            "use_soft_curriculum": False, "use_softmax_sampling": False,
            "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
            "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
            "use_boundary_loss": False, "use_tversky_loss": False,
            "use_cldice_loss": False,
            "use_srl_loss": True, "cldice_weight": 0.05,
            "cldice_start_epoch": 60, "cldice_iters": 7,
            "use_soft_boundary_schedule": False},
    # KD2: Born-Again G1 with higher KD weight (α=0.3 → more teacher influence)
    "KD2": {"name": "kd_born_again_KD2",
            "model_type": "dscformer",
            "use_kd": True,
            "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
            "kd_alpha": 0.3, "kd_temperature": 4.0,
            "no_curriculum": True,
            "use_soft_curriculum": False, "use_softmax_sampling": False,
            "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
            "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
            "use_boundary_loss": False, "use_tversky_loss": False,
            "use_cldice_loss": False,
            "use_srl_loss": True, "cldice_weight": 0.05,
            "cldice_start_epoch": 60, "cldice_iters": 7,
            "use_soft_boundary_schedule": False},

    # Set DKD: Dual-Teacher Knowledge Distillation
    # Teacher 1: G1 (task-specific, high IoU)
    # Teacher 2: SAM2 (foundation model, high ConnR_crack)
    # DKD1: equal-weight ensemble teacher
    "DKD1": {"name": "dual_kd_equal_DKD1",
             "model_type": "dscformer",
             "use_kd": True, "use_dual_kd": True,
             "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
             "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
             "kd_teacher2_model_type": "sam_lora",
             "kd_alpha": 0.5, "kd_temperature": 4.0,
             "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
             "kd_class_weights": False,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # DKD2: class-conditional weights (SAM2 heavier for crack, G1 heavier for spalling)
    "DKD2": {"name": "dual_kd_classaware_DKD2",
             "model_type": "dscformer",
             "use_kd": True, "use_dual_kd": True,
             "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
             "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
             "kd_teacher2_model_type": "sam_lora",
             "kd_alpha": 0.5, "kd_temperature": 4.0,
             "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
             "kd_class_weights": True,
             "kd_crack_t2_weight": 0.6,
             "kd_spalling_t2_weight": 0.3,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # Set DKD round 2: DKD2 + data/loss optimizations from N2/N3
    # DKD3: DKD2 + strong aug + early SRL(30) — main combination
    "DKD3": {"name": "dual_kd_strongaug_DKD3",
             "model_type": "dscformer",
             "aug_level": "strong",
             "use_kd": True, "use_dual_kd": True,
             "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
             "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
             "kd_teacher2_model_type": "sam_lora",
             "kd_alpha": 0.5, "kd_temperature": 4.0,
             "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
             "kd_class_weights": True,
             "kd_crack_t2_weight": 0.6,
             "kd_spalling_t2_weight": 0.3,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 30, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # DKD4: DKD3 + moderate aug (N1m showed moderate can beat strong)
    "DKD4": {"name": "dual_kd_modaug_DKD4",
             "model_type": "dscformer",
             "aug_level": "moderate",
             "use_kd": True, "use_dual_kd": True,
             "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
             "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
             "kd_teacher2_model_type": "sam_lora",
             "kd_alpha": 0.5, "kd_temperature": 4.0,
             "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
             "kd_class_weights": True,
             "kd_crack_t2_weight": 0.6,
             "kd_spalling_t2_weight": 0.3,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 30, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # DKD5: DKD3 + more teacher influence (alpha=0.3 → 70% KD, 30% hard)
    "DKD5": {"name": "dual_kd_moreteacher_DKD5",
             "model_type": "dscformer",
             "aug_level": "strong",
             "use_kd": True, "use_dual_kd": True,
             "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
             "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
             "kd_teacher2_model_type": "sam_lora",
             "kd_alpha": 0.3, "kd_temperature": 4.0,
             "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
             "kd_class_weights": True,
             "kd_crack_t2_weight": 0.6,
             "kd_spalling_t2_weight": 0.3,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 30, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # DKD6: DKD3 + N3-style loss (Tversky + SRL weight 0.10)
    "DKD6": {"name": "dual_kd_tversky_DKD6",
             "model_type": "dscformer",
             "aug_level": "strong",
             "use_kd": True, "use_dual_kd": True,
             "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
             "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
             "kd_teacher2_model_type": "sam_lora",
             "kd_alpha": 0.5, "kd_temperature": 4.0,
             "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
             "kd_class_weights": True,
             "kd_crack_t2_weight": 0.6,
             "kd_spalling_t2_weight": 0.3,
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
    # DKD7: DKD3 + higher temperature (T=6, softer targets)
    "DKD7": {"name": "dual_kd_hightemp_DKD7",
             "model_type": "dscformer",
             "aug_level": "strong",
             "use_kd": True, "use_dual_kd": True,
             "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
             "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
             "kd_teacher2_model_type": "sam_lora",
             "kd_alpha": 0.5, "kd_temperature": 6.0,
             "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
             "kd_class_weights": True,
             "kd_crack_t2_weight": 0.6,
             "kd_spalling_t2_weight": 0.3,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 30, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # DKD8: equal-weight ensemble + strong aug + early SRL (vs DKD3 class-cond)
    "DKD8": {"name": "dual_kd_equal_strongaug_DKD8",
             "model_type": "dscformer",
             "aug_level": "strong",
             "use_kd": True, "use_dual_kd": True,
             "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
             "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
             "kd_teacher2_model_type": "sam_lora",
             "kd_alpha": 0.5, "kd_temperature": 4.0,
             "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
             "kd_class_weights": False,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 30, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},

    # Set DKD round 3: fine-tune around DKD5 (α=0.3, T=4, strong aug, early SRL)
    # --- Axis 1: KD alpha (teacher influence) ---
    # DKD5a: α=0.2 (80% KD, 20% hard — even more teacher)
    "DKD5a": {"name": "dkd5a_alpha02",
              "model_type": "dscformer", "aug_level": "strong",
              "use_kd": True, "use_dual_kd": True,
              "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
              "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
              "kd_teacher2_model_type": "sam_lora",
              "kd_alpha": 0.2, "kd_temperature": 4.0,
              "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
              "kd_class_weights": True,
              "kd_crack_t2_weight": 0.6, "kd_spalling_t2_weight": 0.3,
              "no_curriculum": True,
              "use_soft_curriculum": False, "use_softmax_sampling": False,
              "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
              "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
              "use_boundary_loss": False, "use_tversky_loss": False,
              "use_cldice_loss": False,
              "use_srl_loss": True, "cldice_weight": 0.05,
              "cldice_start_epoch": 30, "cldice_iters": 7,
              "use_soft_boundary_schedule": False},
    # DKD5b: α=0.4 (60% KD, 40% hard — slightly less teacher)
    "DKD5b": {"name": "dkd5b_alpha04",
              "model_type": "dscformer", "aug_level": "strong",
              "use_kd": True, "use_dual_kd": True,
              "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
              "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
              "kd_teacher2_model_type": "sam_lora",
              "kd_alpha": 0.4, "kd_temperature": 4.0,
              "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
              "kd_class_weights": True,
              "kd_crack_t2_weight": 0.6, "kd_spalling_t2_weight": 0.3,
              "no_curriculum": True,
              "use_soft_curriculum": False, "use_softmax_sampling": False,
              "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
              "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
              "use_boundary_loss": False, "use_tversky_loss": False,
              "use_cldice_loss": False,
              "use_srl_loss": True, "cldice_weight": 0.05,
              "cldice_start_epoch": 30, "cldice_iters": 7,
              "use_soft_boundary_schedule": False},
    # --- Axis 2: temperature ---
    # DKD5c: T=3 (sharper teacher targets)
    "DKD5c": {"name": "dkd5c_temp3",
              "model_type": "dscformer", "aug_level": "strong",
              "use_kd": True, "use_dual_kd": True,
              "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
              "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
              "kd_teacher2_model_type": "sam_lora",
              "kd_alpha": 0.3, "kd_temperature": 3.0,
              "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
              "kd_class_weights": True,
              "kd_crack_t2_weight": 0.6, "kd_spalling_t2_weight": 0.3,
              "no_curriculum": True,
              "use_soft_curriculum": False, "use_softmax_sampling": False,
              "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
              "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
              "use_boundary_loss": False, "use_tversky_loss": False,
              "use_cldice_loss": False,
              "use_srl_loss": True, "cldice_weight": 0.05,
              "cldice_start_epoch": 30, "cldice_iters": 7,
              "use_soft_boundary_schedule": False},
    # DKD5d: T=5 (slightly softer, between DKD5 and DKD7)
    "DKD5d": {"name": "dkd5d_temp5",
              "model_type": "dscformer", "aug_level": "strong",
              "use_kd": True, "use_dual_kd": True,
              "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
              "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
              "kd_teacher2_model_type": "sam_lora",
              "kd_alpha": 0.3, "kd_temperature": 5.0,
              "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
              "kd_class_weights": True,
              "kd_crack_t2_weight": 0.6, "kd_spalling_t2_weight": 0.3,
              "no_curriculum": True,
              "use_soft_curriculum": False, "use_softmax_sampling": False,
              "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
              "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
              "use_boundary_loss": False, "use_tversky_loss": False,
              "use_cldice_loss": False,
              "use_srl_loss": True, "cldice_weight": 0.05,
              "cldice_start_epoch": 30, "cldice_iters": 7,
              "use_soft_boundary_schedule": False},
    # --- Axis 3: class-conditional teacher weights ---
    # DKD5e: crack_t2=0.7, spall_t2=0.2 (SAM gets more crack weight)
    "DKD5e": {"name": "dkd5e_crk07_spl02",
              "model_type": "dscformer", "aug_level": "strong",
              "use_kd": True, "use_dual_kd": True,
              "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
              "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
              "kd_teacher2_model_type": "sam_lora",
              "kd_alpha": 0.3, "kd_temperature": 4.0,
              "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
              "kd_class_weights": True,
              "kd_crack_t2_weight": 0.7, "kd_spalling_t2_weight": 0.2,
              "no_curriculum": True,
              "use_soft_curriculum": False, "use_softmax_sampling": False,
              "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
              "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
              "use_boundary_loss": False, "use_tversky_loss": False,
              "use_cldice_loss": False,
              "use_srl_loss": True, "cldice_weight": 0.05,
              "cldice_start_epoch": 30, "cldice_iters": 7,
              "use_soft_boundary_schedule": False},
    # --- Axis 4: SRL schedule ---
    # DKD5f: SRL weight=0.08, start epoch=20 (earlier + stronger)
    "DKD5f": {"name": "dkd5f_srl008_ep20",
              "model_type": "dscformer", "aug_level": "strong",
              "use_kd": True, "use_dual_kd": True,
              "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
              "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
              "kd_teacher2_model_type": "sam_lora",
              "kd_alpha": 0.3, "kd_temperature": 4.0,
              "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
              "kd_class_weights": True,
              "kd_crack_t2_weight": 0.6, "kd_spalling_t2_weight": 0.3,
              "no_curriculum": True,
              "use_soft_curriculum": False, "use_softmax_sampling": False,
              "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
              "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
              "use_boundary_loss": False, "use_tversky_loss": False,
              "use_cldice_loss": False,
              "use_srl_loss": True, "cldice_weight": 0.08,
              "cldice_start_epoch": 20, "cldice_iters": 7,
              "use_soft_boundary_schedule": False},

    # =========================================================================
    # Set DKD round 3: Targeted optimizations based on experiment analysis
    # =========================================================================
    # DKD7: DKD2 + curriculum learning (G2 showed 71.5 mIoU without KD;
    #        adding KD on top of curriculum may push further)
    "DKD7": {"name": "dkd7_curriculum",
             "model_type": "dscformer",
             "use_kd": True, "use_dual_kd": True,
             "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
             "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
             "kd_teacher2_model_type": "sam_lora",
             "kd_alpha": 0.5, "kd_temperature": 4.0,
             "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
             "kd_class_weights": True,
             "kd_crack_t2_weight": 0.6, "kd_spalling_t2_weight": 0.3,
             # Curriculum from G2
             "no_curriculum": False,
             "use_competence_soft_mixing": True,
             "competence_c0": 0.333, "competence_duration": 70,
             "competence_floor_easy": 0.05,
             "competence_floor_medium": 0.02,
             "competence_floor_hard": 0.00,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": True,
             "use_dynamic_loss_reweight": True,
             "loss_reweight_lambda": 0.5,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # DKD8: DKD2 but Teacher 1 = G0 (DSConv only, no SRL)
    #        G0 has ConnR_fg=83.6 vs G1's 77.8; avoids passing SRL's
    #        ConnR regression into the teacher signal
    "DKD8": {"name": "dkd8_teacher_g0",
             "model_type": "dscformer",
             "use_kd": True, "use_dual_kd": True,
             "kd_teacher_checkpoint": "runs/dscformer_plain_G0/best.pt",
             "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
             "kd_teacher2_model_type": "sam_lora",
             "kd_alpha": 0.5, "kd_temperature": 4.0,
             "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
             "kd_class_weights": True,
             "kd_crack_t2_weight": 0.6, "kd_spalling_t2_weight": 0.3,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # DKD9: DKD2 + moderate augmentation
    #        N1m showed moderate aug helps (+0.1 mIoU, +1.2 BF1 over basic);
    #        keeps SRL at epoch 60 (unlike DKD4 which used epoch 30)
    "DKD9": {"name": "dkd9_modaug",
             "model_type": "dscformer",
             "aug_level": "moderate",
             "use_kd": True, "use_dual_kd": True,
             "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
             "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
             "kd_teacher2_model_type": "sam_lora",
             "kd_alpha": 0.5, "kd_temperature": 4.0,
             "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
             "kd_class_weights": True,
             "kd_crack_t2_weight": 0.6, "kd_spalling_t2_weight": 0.3,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 60, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},

    # =========================================================================
    # Set DKD round 3b: SRL ConnR regression fixes
    # Problem: SRL causes ConnR_sp -9.0 (82.4→73.4), ConnR_fg -5.7
    # =========================================================================
    # DKD10: DKD2 but NO SRL — let DTKD alone provide topology guidance
    #         Tests whether DTKD can substitute for SRL entirely
    "DKD10": {"name": "dkd10_no_srl",
              "model_type": "dscformer",
              "use_kd": True, "use_dual_kd": True,
              "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
              "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
              "kd_teacher2_model_type": "sam_lora",
              "kd_alpha": 0.5, "kd_temperature": 4.0,
              "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
              "kd_class_weights": True,
              "kd_crack_t2_weight": 0.6, "kd_spalling_t2_weight": 0.3,
              "no_curriculum": True,
              "use_soft_curriculum": False, "use_softmax_sampling": False,
              "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
              "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
              "use_boundary_loss": False, "use_tversky_loss": False,
              "use_cldice_loss": False,
              "use_srl_loss": False,
              "use_soft_boundary_schedule": False},
    # DKD11: DKD2 + weaker SRL (weight 0.02 instead of 0.05)
    #         Reduces ConnR damage while preserving some BF1 gain
    "DKD11": {"name": "dkd11_weak_srl",
              "model_type": "dscformer",
              "use_kd": True, "use_dual_kd": True,
              "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
              "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
              "kd_teacher2_model_type": "sam_lora",
              "kd_alpha": 0.5, "kd_temperature": 4.0,
              "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
              "kd_class_weights": True,
              "kd_crack_t2_weight": 0.6, "kd_spalling_t2_weight": 0.3,
              "no_curriculum": True,
              "use_soft_curriculum": False, "use_softmax_sampling": False,
              "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
              "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
              "use_boundary_loss": False, "use_tversky_loss": False,
              "use_cldice_loss": False,
              "use_srl_loss": True, "cldice_weight": 0.02,
              "cldice_start_epoch": 60, "cldice_iters": 7,
              "use_soft_boundary_schedule": False},
    # DKD12: DKD2 + late SRL (epoch 80 instead of 60)
    #         Gives model more time to stabilize before SRL perturbs gradients
    "DKD12": {"name": "dkd12_late_srl",
              "model_type": "dscformer",
              "use_kd": True, "use_dual_kd": True,
              "kd_teacher_checkpoint": "runs/dscformer_srl_G1/best.pt",
              "kd_teacher2_checkpoint": "runs/sam_lora_srl_SAM2/best.pt",
              "kd_teacher2_model_type": "sam_lora",
              "kd_alpha": 0.5, "kd_temperature": 4.0,
              "kd_t1_weight": 0.5, "kd_t2_weight": 0.5,
              "kd_class_weights": True,
              "kd_crack_t2_weight": 0.6, "kd_spalling_t2_weight": 0.3,
              "no_curriculum": True,
              "use_soft_curriculum": False, "use_softmax_sampling": False,
              "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
              "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
              "use_boundary_loss": False, "use_tversky_loss": False,
              "use_cldice_loss": False,
              "use_srl_loss": True, "cldice_weight": 0.05,
              "cldice_start_epoch": 80, "cldice_iters": 7,
              "use_soft_boundary_schedule": False},

    # Set CAL: Crack-Aware Adaptive LoRA (MoE-LoRA + crack-complexity router)
    # CAL1: CALoRA-SAM baseline (CE + Dice, no SRL)
    "CAL1": {"name": "calora_sam_CAL1",
             "model_type": "calora_sam",
             "epochs": 50, "batch_size": 2,
             "calora_ranks": (8, 32),
             "calora_router_hidden": 64,
             "calora_router_aux_weight": 0.1,
             "lora_alpha": 16.0,
             "sam_img_size": 1024, "sam_fpn_dim": 256,
             "sam_lr_lora": 1e-4, "sam_lr_decoder": 3e-4,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False, "use_srl_loss": False,
             "use_soft_boundary_schedule": False},
    # CAL2: CALoRA-SAM + SRL (topology-aware)
    "CAL2": {"name": "calora_sam_srl_CAL2",
             "model_type": "calora_sam",
             "epochs": 50, "batch_size": 2,
             "calora_ranks": (8, 32),
             "calora_router_hidden": 64,
             "calora_router_aux_weight": 0.1,
             "lora_alpha": 16.0,
             "sam_img_size": 1024, "sam_fpn_dim": 256,
             "sam_lr_lora": 1e-4, "sam_lr_decoder": 3e-4,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 20, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # CAL3: CALoRA-SAM + SRL, 3 experts (rank 4/16/32)
    "CAL3": {"name": "calora_3exp_srl_CAL3",
             "model_type": "calora_sam",
             "epochs": 50, "batch_size": 2,
             "calora_ranks": (4, 16, 32),
             "calora_router_hidden": 64,
             "calora_router_aux_weight": 0.1,
             "lora_alpha": 16.0,
             "sam_img_size": 1024, "sam_fpn_dim": 256,
             "sam_lr_lora": 1e-4, "sam_lr_decoder": 3e-4,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 20, "cldice_iters": 7,
             "use_soft_boundary_schedule": False},
    # CAL4: CALoRA-SAM + SRL, no aux routing loss (ablation)
    "CAL4": {"name": "calora_noaux_srl_CAL4",
             "model_type": "calora_sam",
             "epochs": 50, "batch_size": 2,
             "calora_ranks": (8, 32),
             "calora_router_hidden": 64,
             "calora_router_aux_weight": 0.0,
             "lora_alpha": 16.0,
             "sam_img_size": 1024, "sam_fpn_dim": 256,
             "sam_lr_lora": 1e-4, "sam_lr_decoder": 3e-4,
             "no_curriculum": True,
             "use_soft_curriculum": False, "use_softmax_sampling": False,
             "use_dynamic_difficulty": False, "use_dynamic_loss_reweight": False,
             "use_class_sampling_bonus": False, "use_class_loss_schedule": False,
             "use_boundary_loss": False, "use_tversky_loss": False,
             "use_cldice_loss": False,
             "use_srl_loss": True, "cldice_weight": 0.05,
             "cldice_start_epoch": 20, "cldice_iters": 7,
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

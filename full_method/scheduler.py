"""Curriculum scheduler: stage progression + loss weight scheduling.

Stage 0 (0-30%): Easy only, lower crack/boundary weights
Stage 1 (30-70%): Easy+Medium, full crack weight
Stage 2 (70-100%): All tiers, boosted crack weight

Soft curriculum mode: smooth tier_mix ratios instead of hard stage gates.

MAC extensions:
- AdaptivePacer: validation-metric-driven stage transitions
- ClassLossScheduler: per-class IoU gap → dynamic CE weight multipliers
"""
from __future__ import annotations

import math
from typing import Dict, Tuple


class CurriculumScheduler:
    """Single source of truth for curriculum stage and loss weight scheduling."""

    def __init__(self, total_epochs: int, cfg=None):
        self.total_epochs = total_epochs
        self.cfg = cfg

    def stage(self, epoch: int) -> int:
        ratio = epoch / self.total_epochs
        if ratio < 0.3:
            return 0
        elif ratio < 0.7:
            return 1
        return 2

    def tier_mix(self, epoch: int) -> Dict[int, float]:
        """Smooth tier mixing ratios for soft curriculum."""
        ratio = epoch / self.total_epochs
        if ratio < 0.30:
            return {0: 0.80, 1: 0.20, 2: 0.00}
        elif ratio < 0.60:
            return {0: 0.50, 1: 0.35, 2: 0.15}
        return {0: 0.30, 1: 0.35, 2: 0.35}

    def competence(self, epoch: int) -> float:
        """Competence c(t) in [0, 1]. Platanios et al. 2019. t = epoch - 1 (0-based)."""
        t = epoch - 1  # epoch 1 → t=0, so c(1) = c0 exactly
        c0 = self.cfg.competence_c0
        T_c = self.cfg.competence_duration
        if t >= T_c:
            return 1.0
        return min(1.0, math.sqrt(t * (1 - c0**2) / T_c + c0**2))

    def max_allowed_tier(self, epoch: int) -> int:
        """C1: hard tier unlock based on competence."""
        c = self.competence(epoch)
        if c < 2/3:
            return 0
        if c < 1.0:
            return 1
        return 2

    def competence_tier_weights(self, epoch: int) -> Dict[int, float]:
        """C2: soft tier weights derived from competence, with asymmetric floors."""
        c = self.competence(epoch)
        floors = {
            0: self.cfg.competence_floor_easy,
            1: self.cfg.competence_floor_medium,
            2: self.cfg.competence_floor_hard,
        }
        ramp = 1.0 / 3.0
        weights = {}
        for k in range(3):
            threshold = k / 3.0
            activation = min(1.0, max(0.0, (c - threshold) / ramp))
            weights[k] = floors[k] + (1 - floors[k]) * activation
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def crack_weight(self, epoch: int) -> float:
        ratio = epoch / self.total_epochs
        if ratio < 0.3:
            return 0.5
        elif ratio < 0.7:
            return 1.0
        return 1.5

    def boundary_weight(self, epoch: int) -> float:
        if self.cfg and self.cfg.use_soft_boundary_schedule:
            ratio = epoch / self.total_epochs
            start = self.cfg.boundary_start_ratio   # 0.6
            max_w = self.cfg.boundary_max_weight     # 0.10
            if ratio < start:
                return 0.0
            progress = (ratio - start) / (1.0 - start)
            return max_w * progress
        # Legacy behavior
        ratio = epoch / self.total_epochs
        if ratio < 0.3:
            return 0.5
        return 1.0


# ---------------------------------------------------------------------------
# MAC: Adaptive Pacer (validation-metric-driven stage transitions)
# ---------------------------------------------------------------------------

class AdaptivePacer:
    """Adaptive curriculum pacing driven by validation crack/spalling IoU.

    Stage transitions occur when both crack_IoU and spalling_IoU exceed
    thresholds, or after `patience` consecutive validations with stagnation.
    Each stage has a minimum epoch duration.
    """

    def __init__(self, cfg):
        self.stage = 0
        self.crack_thresholds = [0.45, 0.55]
        self.spalling_thresholds = [0.50, 0.60]
        self.patience = cfg.mac_adaptive_patience
        self.min_epochs = cfg.mac_adaptive_min_epochs
        self.stagnation_count = 0
        self.best_miou_fg = -1.0
        self.stage_start_epoch = 1
        self.current_epoch = 0

    def update(self, val_metrics: Dict, epoch: int) -> int:
        """Called after each validation. Returns current stage (0/1/2)."""
        self.current_epoch = epoch
        crack_iou = val_metrics.get("IoU_crack", 0.0)
        spalling_iou = val_metrics.get("IoU_spalling", 0.0)
        miou_fg = val_metrics.get("mIoU_fg", 0.0)

        # Check stagnation
        if miou_fg > self.best_miou_fg + 0.001:
            self.best_miou_fg = miou_fg
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        if self.stage >= 2:
            return self.stage

        # Minimum epoch guard
        epochs_in_stage = epoch - self.stage_start_epoch + 1
        if epochs_in_stage < self.min_epochs:
            return self.stage

        # Check threshold or stagnation for promotion
        thresh_crack = self.crack_thresholds[self.stage]
        thresh_spalling = self.spalling_thresholds[self.stage]
        threshold_met = (crack_iou > thresh_crack and spalling_iou > thresh_spalling)
        stagnation_met = (self.stagnation_count >= self.patience)

        if threshold_met or stagnation_met:
            reason = "threshold" if threshold_met else "stagnation"
            self.stage += 1
            self.stage_start_epoch = epoch
            self.stagnation_count = 0
            self.best_miou_fg = miou_fg
            print(f"[MAC pacer] stage {self.stage - 1} → {self.stage} "
                  f"at epoch {epoch} ({reason}) "
                  f"crack_IoU={crack_iou:.3f} spalling_IoU={spalling_iou:.3f}")

        return self.stage

    def tier_weights(self) -> Dict[int, float]:
        """Return tier sampling weights for current stage."""
        if self.stage == 0:
            return {0: 0.80, 1: 0.20, 2: 0.00}
        elif self.stage == 1:
            return {0: 0.40, 1: 0.40, 2: 0.20}
        else:
            return {0: 0.25, 1: 0.35, 2: 0.40}


# ---------------------------------------------------------------------------
# MAC: Class Loss Scheduler (per-class IoU gap → CE weight adjustment)
# ---------------------------------------------------------------------------

class ClassLossScheduler:
    """Dynamically adjust CE class weights based on per-class IoU gap.

    When crack IoU lags behind target, boost crack CE weight (and topology loss).
    When spalling IoU lags, boost spalling CE weight.
    """

    def __init__(self, cfg):
        self.target_crack_iou = 0.65
        self.target_spalling_iou = 0.75
        self.max_boost = cfg.mac_class_loss_max_boost
        self.ema_alpha = cfg.mac_class_loss_ema
        self.crack_mult = 1.0
        self.spalling_mult = 1.0
        self.topo_mult = 1.0

    def update(self, val_metrics: Dict) -> None:
        """Update multipliers based on latest validation metrics."""
        crack_iou = val_metrics.get("IoU_crack", 0.0)
        spalling_iou = val_metrics.get("IoU_spalling", 0.0)

        crack_gap = max(0.0, self.target_crack_iou - crack_iou) / self.target_crack_iou
        spalling_gap = max(0.0, self.target_spalling_iou - spalling_iou) / self.target_spalling_iou

        raw_crack = 1.0 + crack_gap * (self.max_boost - 1.0)
        raw_spalling = 1.0 + spalling_gap * (self.max_boost - 1.0)

        self.crack_mult = self.ema_alpha * self.crack_mult + (1 - self.ema_alpha) * raw_crack
        self.spalling_mult = self.ema_alpha * self.spalling_mult + (1 - self.ema_alpha) * raw_spalling

        # Topology loss multiplier: boost when crack underperforms
        self.topo_mult = self.ema_alpha * self.topo_mult + (1 - self.ema_alpha) * raw_crack

    def get_ce_weight_multipliers(self) -> Tuple[float, float]:
        """Return (crack_multiplier, spalling_multiplier)."""
        return (self.crack_mult, self.spalling_mult)

    def get_topo_multiplier(self) -> float:
        """Return topology loss multiplier."""
        return self.topo_mult

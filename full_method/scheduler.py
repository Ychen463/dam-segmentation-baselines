"""Curriculum scheduler: stage progression + loss weight scheduling.

Stage 0 (0-30%): Easy only, lower crack/boundary weights
Stage 1 (30-70%): Easy+Medium, full crack weight
Stage 2 (70-100%): All tiers, boosted crack weight

Soft curriculum mode: smooth tier_mix ratios instead of hard stage gates.
"""
from __future__ import annotations

import math
from typing import Dict


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

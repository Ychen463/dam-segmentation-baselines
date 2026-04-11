"""Curriculum scheduler: stage progression + loss weight scheduling.

Stage 0 (0-30%): Easy only, lower crack/boundary weights
Stage 1 (30-70%): Easy+Medium, full crack weight
Stage 2 (70-100%): All tiers, boosted crack weight

Soft curriculum mode: smooth tier_mix ratios instead of hard stage gates.
"""
from __future__ import annotations

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
        if ratio < 0.25:
            return {0: 0.85, 1: 0.15, 2: 0.00}
        elif ratio < 0.55:
            return {0: 0.55, 1: 0.35, 2: 0.10}
        elif ratio < 0.80:
            return {0: 0.30, 1: 0.40, 2: 0.30}
        return {0: 0.20, 1: 0.35, 2: 0.45}

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

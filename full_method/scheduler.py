"""Curriculum scheduler: stage progression + loss weight scheduling.

Stage 0 (0-30%): Easy only, lower crack/boundary weights
Stage 1 (30-70%): Easy+Medium, full crack weight
Stage 2 (70-100%): All tiers, boosted crack weight
"""
from __future__ import annotations


class CurriculumScheduler:
    """Single source of truth for curriculum stage and loss weight scheduling."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs

    def stage(self, epoch: int) -> int:
        ratio = epoch / self.total_epochs
        if ratio < 0.3:
            return 0
        elif ratio < 0.7:
            return 1
        return 2

    def crack_weight(self, epoch: int) -> float:
        ratio = epoch / self.total_epochs
        if ratio < 0.3:
            return 0.5
        elif ratio < 0.7:
            return 1.0
        return 1.5

    def boundary_weight(self, epoch: int) -> float:
        ratio = epoch / self.total_epochs
        if ratio < 0.3:
            return 0.5
        return 1.0

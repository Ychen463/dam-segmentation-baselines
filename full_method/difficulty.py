"""Online difficulty estimation with epoch-level z-score normalization.

Each training sample has a SampleState tracking EMA loss, uncertainty,
boundary complexity, and sparsity. After each epoch, all scores are
z-normalized and combined into a single difficulty score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SampleState:
    ema_loss: float = 0.0
    ema_uncertainty: float = 0.0
    boundary_complexity: float = 0.0
    sparsity: float = 0.0
    difficulty: float = 0.0


class DifficultyEstimator:
    """Maintains per-sample difficulty scores with EMA + z-score normalization."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 0.3, delta: float = 0.3,
                 ema_decay: float = 0.9):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.ema_decay = ema_decay

    def update(self, state: SampleState, loss_val: float, uncert_val: float,
               mask: np.ndarray) -> None:
        """Update a single sample's state after a training step."""
        state.ema_loss = self.ema_decay * state.ema_loss + (1 - self.ema_decay) * loss_val
        state.ema_uncertainty = (self.ema_decay * state.ema_uncertainty
                                 + (1 - self.ema_decay) * uncert_val)
        state.boundary_complexity = self._compute_boundary_complexity(mask)
        state.sparsity = self._compute_sparsity(mask)

    @staticmethod
    def _compute_boundary_complexity(mask: np.ndarray) -> float:
        """Crack boundary pixels / crack pixels. Crack = class 1."""
        crack = (mask == 1).astype(np.float32)
        crack_px = crack.sum()
        if crack_px < 1:
            return 0.0
        # Morphological dilation/erosion via max_pool2d
        crack_t = torch.from_numpy(crack).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        dil = F.max_pool2d(crack_t, 3, 1, 1)
        ero = -F.max_pool2d(-crack_t, 3, 1, 1)
        boundary_px = float((dil != ero).float().sum())
        return boundary_px / (crack_px + 1e-6)

    @staticmethod
    def _compute_sparsity(mask: np.ndarray) -> float:
        """1 - foreground_ratio. Higher = sparser foreground."""
        return 1.0 - float((mask > 0).mean())

    def normalize_and_score(self, sample_bank: Dict[str, SampleState]) -> None:
        """Epoch-level z-score normalization, then weighted combination."""
        if not sample_bank:
            return

        ids = list(sample_bank.keys())
        losses = np.array([sample_bank[k].ema_loss for k in ids])
        uncerts = np.array([sample_bank[k].ema_uncertainty for k in ids])
        bds = np.array([sample_bank[k].boundary_complexity for k in ids])
        spars = np.array([sample_bank[k].sparsity for k in ids])

        def zscore(arr: np.ndarray) -> np.ndarray:
            return (arr - arr.mean()) / (arr.std() + 1e-6)

        z_loss = zscore(losses)
        z_uncert = zscore(uncerts)
        z_bd = zscore(bds)
        z_spar = zscore(spars)

        for i, k in enumerate(ids):
            sample_bank[k].difficulty = float(
                self.alpha * z_loss[i]
                + self.beta * z_uncert[i]
                + self.gamma * z_bd[i]
                + self.delta * z_spar[i]
            )

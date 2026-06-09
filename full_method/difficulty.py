"""Online difficulty estimation with epoch-level z-score normalization.

Each training sample has a SampleState tracking EMA loss, uncertainty,
boundary complexity, and sparsity. After each epoch, all scores are
z-normalized and combined into a single difficulty score.

MAC extension: morphology-aware difficulty scoring using precomputed
crack geometry features (width, junctions, proximity, components).
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
    # --- MAC fields ---
    ema_loss_crack: float = 0.0
    ema_loss_spalling: float = 0.0
    morph_difficulty: float = 0.0      # static morphology-based difficulty
    # --- DGACL fields ---
    teacher_disagreement: float = 0.0  # static: KL(T1 || T2) pixel-mean
    student_teacher_gap: float = 0.0   # dynamic: KL(ensemble || student) EMA


class DifficultyEstimator:
    """Maintains per-sample difficulty scores with EMA + z-score normalization."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 0.3, delta: float = 0.3,
                 ema_decay: float = 0.9,
                 use_mac_morph_difficulty: bool = False,
                 mac_diff_gamma: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.ema_decay = ema_decay
        self.use_mac_morph_difficulty = use_mac_morph_difficulty
        self.mac_diff_gamma = mac_diff_gamma

    def init_morph(self, state: SampleState, morph_difficulty: float) -> None:
        """Set the static morphology difficulty from precomputed features."""
        state.morph_difficulty = morph_difficulty

    def update(self, state: SampleState, loss_val: float, uncert_val: float,
               mask: np.ndarray) -> None:
        """Update a single sample's state after a training step."""
        state.ema_loss = self.ema_decay * state.ema_loss + (1 - self.ema_decay) * loss_val
        state.ema_uncertainty = (self.ema_decay * state.ema_uncertainty
                                 + (1 - self.ema_decay) * uncert_val)
        state.boundary_complexity = self._compute_boundary_complexity(mask)
        state.sparsity = self._compute_sparsity(mask)

    def update_class_loss(self, state: SampleState,
                          loss_crack: float, loss_spalling: float) -> None:
        """Update per-class EMA losses (MAC)."""
        state.ema_loss_crack = (self.ema_decay * state.ema_loss_crack
                                + (1 - self.ema_decay) * loss_crack)
        state.ema_loss_spalling = (self.ema_decay * state.ema_loss_spalling
                                   + (1 - self.ema_decay) * loss_spalling)

    def update_dgacl(self, state: SampleState,
                     disagree_val: float, gap_val: float) -> None:
        """Update DGACL fields: static disagreement + EMA student-teacher gap."""
        state.teacher_disagreement = disagree_val  # static per-sample
        state.student_teacher_gap = (self.ema_decay * state.student_teacher_gap
                                     + (1 - self.ema_decay) * gap_val)

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

    def normalize_and_score(self, sample_bank: Dict[str, SampleState],
                            use_dgacl: bool = False,
                            dgacl_w_disagree: float = 0.5,
                            dgacl_w_gap: float = 0.3,
                            dgacl_w_loss: float = 0.2) -> None:
        """Epoch-level z-score normalization, then weighted combination."""
        if not sample_bank:
            return

        ids = list(sample_bank.keys())
        losses = np.array([sample_bank[k].ema_loss for k in ids])
        uncerts = np.array([sample_bank[k].ema_uncertainty for k in ids])

        def zscore(arr: np.ndarray) -> np.ndarray:
            return (arr - arr.mean()) / (arr.std() + 1e-6)

        z_loss = zscore(losses)
        z_uncert = zscore(uncerts)

        if use_dgacl:
            # DGACL mode: teacher disagreement + student-teacher gap + EMA loss
            disagrees = np.array([sample_bank[k].teacher_disagreement for k in ids])
            gaps = np.array([sample_bank[k].student_teacher_gap for k in ids])
            z_disagree = zscore(disagrees)
            z_gap = zscore(gaps)
            for i, k in enumerate(ids):
                sample_bank[k].difficulty = float(
                    dgacl_w_disagree * z_disagree[i]
                    + dgacl_w_gap * z_gap[i]
                    + dgacl_w_loss * z_loss[i]
                )
        elif self.use_mac_morph_difficulty:
            # MAC mode: replace boundary_complexity + sparsity with morph_difficulty
            morphs = np.array([sample_bank[k].morph_difficulty for k in ids])
            z_morph = zscore(morphs)
            for i, k in enumerate(ids):
                sample_bank[k].difficulty = float(
                    self.alpha * z_loss[i]
                    + self.beta * z_uncert[i]
                    + self.mac_diff_gamma * z_morph[i]
                )
        else:
            # Legacy mode: boundary_complexity + sparsity
            bds = np.array([sample_bank[k].boundary_complexity for k in ids])
            spars = np.array([sample_bank[k].sparsity for k in ids])
            z_bd = zscore(bds)
            z_spar = zscore(spars)
            for i, k in enumerate(ids):
                sample_bank[k].difficulty = float(
                    self.alpha * z_loss[i]
                    + self.beta * z_uncert[i]
                    + self.gamma * z_bd[i]
                    + self.delta * z_spar[i]
                )

"""Tier-aware dynamic sampler with difficulty-based sampling.

During warmup: uniform random sampling within allowed tiers.
After warmup: softmax(difficulty / tau) sampling without replacement,
with class-aware bonuses for spalling and late-stage hard crack samples.

Soft curriculum mode: tier-mix proportional sampling (no hard tier gating).
"""
from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional, Set

import torch
from torch.utils.data import Sampler

from .difficulty import SampleState


class TierAwareDynamicSampler(Sampler[int]):
    """Dynamic curriculum sampler: tier gating + difficulty-based sampling."""

    def __init__(self, records: List[Dict], sample_bank: Dict[str, SampleState],
                 tau: float = 0.5, spalling_bonus: float = 0.3,
                 late_hard_crack_bonus: float = 0.4,
                 enable_dynamic: bool = True,
                 use_soft_curriculum: bool = False,
                 use_softmax_sampling: bool = True):
        self.records = records
        self.sample_bank = sample_bank
        self.tau = tau
        self.spalling_bonus = spalling_bonus
        self.late_hard_crack_bonus = late_hard_crack_bonus
        self._enable_dynamic = enable_dynamic
        self._use_softmax_sampling = use_softmax_sampling
        self._use_soft_curriculum = use_soft_curriculum
        self._stage = 0
        self._epoch_ratio = 0.0
        self._use_dynamic = False
        self._tier_mix: Optional[Dict[int, float]] = None
        self._last_sampled_indices: List[int] = []

    def set_epoch(self, epoch: int, total_epochs: int,
                  warmup_epochs: int, stage: int,
                  tier_mix: Optional[Dict[int, float]] = None) -> None:
        """Update state for the new epoch. Stage is computed externally."""
        self._stage = stage
        self._epoch_ratio = epoch / total_epochs
        self._use_dynamic = self._enable_dynamic and (epoch > warmup_epochs)
        self._tier_mix = tier_mix

    def _allowed_tiers(self) -> Set[int]:
        if self._stage == 0:
            return {0}
        elif self._stage == 1:
            return {0, 1}
        return {0, 1, 2}

    def __iter__(self) -> Iterator[int]:
        # Soft curriculum mode: tier-mix proportional sampling
        if self._tier_mix is not None:
            by_tier: Dict[int, List[int]] = {0: [], 1: [], 2: []}
            for i, r in enumerate(self.records):
                by_tier[r["tier"]].append(i)

            total_n = len(self.records)
            indices: List[int] = []
            for tier, ratio in self._tier_mix.items():
                pool = by_tier.get(tier, [])
                n = round(ratio * total_n)
                if n > 0 and pool:
                    n = min(n, len(pool))
                    indices.extend(random.sample(pool, n))
            random.shuffle(indices)
            self._last_sampled_indices = indices
            yield from indices
            return

        # Legacy path: hard tier gating
        allowed = self._allowed_tiers()
        valid = [(i, r) for i, r in enumerate(self.records) if r["tier"] in allowed]

        if not self._use_dynamic or not self._use_softmax_sampling:
            # Uniform random within allowed tiers
            indices = [i for i, _ in valid]
            random.shuffle(indices)
            self._last_sampled_indices = indices
            yield from indices
            return

        # Legacy softmax sampling path
        scores = []
        for i, r in valid:
            sid = r["id"]
            base = self.sample_bank[sid].difficulty if sid in self.sample_bank else 0.0
            bonus = 0.0
            if r["has_spalling"]:
                bonus += self.spalling_bonus
            if self._epoch_ratio > 0.7 and r["tier"] == 2 and not r["has_spalling"]:
                bonus += self.late_hard_crack_bonus
            scores.append(base + bonus)

        scores_t = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(scores_t / self.tau, dim=0)
        sampled = torch.multinomial(probs, num_samples=len(valid), replacement=False)

        result_indices = [valid[k][0] for k in sampled.tolist()]
        self._last_sampled_indices = result_indices

        yield from result_indices

    def get_sampling_stats(self) -> Dict:
        """Return tier histogram and has_spalling ratio from last __iter__."""
        if not self._last_sampled_indices:
            return {}
        tier_hist = {0: 0, 1: 0, 2: 0}
        sp_count = 0
        for idx in self._last_sampled_indices:
            r = self.records[idx]
            tier_hist[r["tier"]] += 1
            if r["has_spalling"]:
                sp_count += 1
        total = len(self._last_sampled_indices)
        return {
            "tier_hist": tier_hist,
            "has_spalling_ratio": sp_count / max(total, 1),
            "total": total,
        }

    def __len__(self) -> int:
        if self._tier_mix is not None:
            return len(self.records)
        allowed = self._allowed_tiers()
        return sum(1 for r in self.records if r["tier"] in allowed)

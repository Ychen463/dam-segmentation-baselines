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
                 use_softmax_sampling: bool = True,
                 no_curriculum: bool = False):
        self.records = records
        self.sample_bank = sample_bank
        self.tau = tau
        self.spalling_bonus = spalling_bonus
        self.late_hard_crack_bonus = late_hard_crack_bonus
        self._enable_dynamic = enable_dynamic
        self._use_softmax_sampling = use_softmax_sampling
        self._use_soft_curriculum = use_soft_curriculum
        self._no_curriculum = no_curriculum
        self._stage = 0
        self._epoch_ratio = 0.0
        self._use_dynamic = False
        self._tier_mix: Optional[Dict[int, float]] = None
        self._competence_tier: Optional[int] = None
        self._competence_tier_weights: Optional[Dict[int, float]] = None
        self._last_sampled_indices: List[int] = []

    def set_epoch(self, epoch: int, total_epochs: int,
                  warmup_epochs: int, stage: int,
                  tier_mix: Optional[Dict[int, float]] = None,
                  competence_tier: Optional[int] = None,
                  competence_tier_weights: Optional[Dict[int, float]] = None) -> None:
        """Update state for the new epoch. Stage is computed externally."""
        self._stage = stage
        self._epoch_ratio = epoch / total_epochs
        self._use_dynamic = self._enable_dynamic and (epoch > warmup_epochs)
        self._tier_mix = tier_mix
        self._competence_tier = competence_tier
        self._competence_tier_weights = competence_tier_weights

    def _allowed_tiers(self) -> Set[int]:
        if self._stage == 0:
            return {0}
        elif self._stage == 1:
            return {0, 1}
        return {0, 1, 2}

    def __iter__(self) -> Iterator[int]:
        # No-curriculum mode
        if self._no_curriculum:
            if self._use_dynamic and self._use_softmax_sampling:
                # Difficulty-biased sampling over all samples
                scores = torch.tensor([
                    self.sample_bank.get(r["id"], SampleState()).difficulty
                    for r in self.records
                ], dtype=torch.float32)
                probs = torch.softmax(scores / self.tau, dim=0)
                sampled = torch.multinomial(probs, num_samples=len(self.records),
                                            replacement=False)
                indices = sampled.tolist()
            else:
                indices = list(range(len(self.records)))
                random.shuffle(indices)
            self._last_sampled_indices = indices
            yield from indices
            return

        # C2: competence soft mixing -- fixed epoch size with per-tier replacement
        if self._competence_tier_weights is not None:
            by_tier: Dict[int, List[int]] = {0: [], 1: [], 2: []}
            for i, r in enumerate(self.records):
                by_tier[r["tier"]].append(i)
            total_n = len(self.records)  # 1200

            # Largest-remainder quota allocation (guarantees sum = total_n)
            tiers = sorted(self._competence_tier_weights.keys())
            float_quotas = {k: self._competence_tier_weights[k] * total_n for k in tiers}
            int_quotas = {k: int(float_quotas[k]) for k in tiers}
            remainders = {k: float_quotas[k] - int_quotas[k] for k in tiers}
            leftover = total_n - sum(int_quotas.values())
            for k in sorted(tiers, key=lambda k: remainders[k], reverse=True):
                if leftover <= 0:
                    break
                int_quotas[k] += 1
                leftover -= 1

            indices: List[int] = []
            for tier in tiers:
                pool = by_tier.get(tier, [])
                n = int_quotas[tier]
                if n > 0 and pool:
                    if (self._use_dynamic and self._use_softmax_sampling
                            and len(pool) > 1):
                        # Difficulty-weighted sampling within tier
                        scores = torch.tensor([
                            self.sample_bank.get(self.records[i]["id"],
                                                 SampleState()).difficulty
                            for i in pool
                        ], dtype=torch.float32)
                        probs = torch.softmax(scores / self.tau, dim=0)
                        sampled = torch.multinomial(
                            probs, num_samples=min(n, len(pool)),
                            replacement=False)
                        picked = [pool[k] for k in sampled.tolist()]
                        if n > len(pool):
                            extra = torch.multinomial(
                                probs, num_samples=n - len(pool),
                                replacement=True)
                            picked += [pool[k] for k in extra.tolist()]
                        indices.extend(picked)
                    else:
                        if n <= len(pool):
                            indices.extend(random.sample(pool, n))
                        else:
                            indices.extend(random.choices(pool, k=n))
            random.shuffle(indices)
            self._last_sampled_indices = indices
            yield from indices
            return

        # C1: competence hard unlock (uniform within accessible tiers, no replacement)
        if self._competence_tier is not None:
            indices = [i for i, r in enumerate(self.records)
                       if r["tier"] <= self._competence_tier]
            random.shuffle(indices)
            self._last_sampled_indices = indices
            yield from indices
            return

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
        # Unique sample tracking (useful for C2 oversampling)
        unique_per_tier: Dict[int, set] = {0: set(), 1: set(), 2: set()}
        for idx in self._last_sampled_indices:
            unique_per_tier[self.records[idx]["tier"]].add(idx)
        unique_hist = {k: len(v) for k, v in unique_per_tier.items()}
        unique_total = sum(unique_hist.values())
        return {
            "tier_hist": tier_hist,
            "unique_hist": unique_hist,
            "unique_total": unique_total,
            "dup_ratio": total / max(unique_total, 1),
            "has_spalling_ratio": sp_count / max(total, 1),
            "total": total,
        }

    def __len__(self) -> int:
        if self._no_curriculum or self._competence_tier_weights is not None:
            return len(self.records)  # always 1200 for C2 and no_curriculum
        if self._competence_tier is not None:
            return sum(1 for r in self.records if r["tier"] <= self._competence_tier)
        if self._tier_mix is not None:
            return len(self.records)
        allowed = self._allowed_tiers()
        return sum(1 for r in self.records if r["tier"] in allowed)

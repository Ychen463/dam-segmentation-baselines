"""Full metric suite: IoU + Dice + BF1 + clDice per class.

Extends ``baseline_deeplab.metrics.SegMetricsBF1`` with clDice so that a
single ``update()`` / ``compute()`` cycle produces every metric the paper
needs for experiments A–C.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from baseline_deeplab.metrics import SegMetricsBF1
from .cldice import cldice_single


class SegMetricsFull(SegMetricsBF1):
    """Confusion-matrix metrics + BF1 + clDice."""

    def __init__(self, num_classes: int, tol_px: float):
        super().__init__(num_classes, tol_px)
        self.cldice_sums: Dict[str, float] = {"crack": 0.0, "spalling": 0.0}
        self.cldice_counts: Dict[str, int] = {"crack": 0, "spalling": 0}

    def reset(self) -> None:
        super().reset()
        for k in self.cldice_sums:
            self.cldice_sums[k] = 0.0
            self.cldice_counts[k] = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        super().update(logits, target)
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        tgt = target.detach().cpu().numpy()
        for b in range(pred.shape[0]):
            for name, cid in (("crack", 1), ("spalling", 2)):
                val = cldice_single(pred[b], tgt[b], cid)
                if val is not None:
                    self.cldice_sums[name] += val
                    self.cldice_counts[name] += 1

    def compute(self) -> Dict[str, float]:
        m = super().compute()
        for name in ("crack", "spalling"):
            cnt = self.cldice_counts[name]
            m[f"clDice_{name}"] = (self.cldice_sums[name] / cnt) if cnt > 0 else 0.0
        m["clDice_fg_mean"] = 0.5 * (m["clDice_crack"] + m["clDice_spalling"])
        return m

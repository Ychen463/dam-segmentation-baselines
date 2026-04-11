"""Val/test metrics that extend Step 0 confusion-matrix metrics with Boundary F1.

Train loops should keep using ``baseline_unet.metrics.SegMetrics`` to avoid the
per-image EDT cost. Only val and test should use ``SegMetricsBF1``.
"""
from __future__ import annotations

from typing import Dict

import torch

from baseline_unet.metrics import SegMetrics, format_metrics as _base_fmt
from .boundary_f1 import boundary_f1_single


class SegMetricsBF1(SegMetrics):
    def __init__(self, num_classes: int, tol_px: float):
        super().__init__(num_classes)
        self.tol_px = tol_px
        self.bf1_sums = {"crack": 0.0, "spalling": 0.0}
        self.bf1_counts = {"crack": 0, "spalling": 0}

    def reset(self) -> None:
        super().reset()
        for k in self.bf1_sums:
            self.bf1_sums[k] = 0.0
            self.bf1_counts[k] = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        super().update(logits, target)
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        tgt = target.detach().cpu().numpy()
        for b in range(pred.shape[0]):
            for name, cid in (("crack", 1), ("spalling", 2)):
                bf1 = boundary_f1_single(pred[b], tgt[b], cid, self.tol_px)
                if bf1 is not None:
                    self.bf1_sums[name] += bf1
                    self.bf1_counts[name] += 1

    def compute(self) -> Dict[str, float]:
        m = super().compute()
        for name in ("crack", "spalling"):
            cnt = self.bf1_counts[name]
            m[f"BF1_{name}"] = (self.bf1_sums[name] / cnt) if cnt > 0 else 0.0
        m["BF1_fg_mean"] = 0.5 * (m["BF1_crack"] + m["BF1_spalling"])
        return m


def format_metrics(m: Dict[str, float]) -> str:
    base = _base_fmt(m)
    if "BF1_fg_mean" in m:
        base += (
            "\nBF1   crack={BF1_crack:.4f}  spalling={BF1_spalling:.4f}"
            "  | BF1_fg_mean={BF1_fg_mean:.4f}".format(**m)
        )
    if "clDice_crack" in m:
        base += "\nclDice crack={clDice_crack:.4f}".format(**m)
    return base

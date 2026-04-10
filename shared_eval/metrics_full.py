"""Full metric suite: IoU + Dice + BF1 + clDice + Connectivity per class.

Extends ``baseline_deeplab.metrics.SegMetricsBF1`` with clDice and crack
connectivity ratio so that a single ``update()`` / ``compute()`` cycle
produces every metric the paper needs for experiments A–C.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy import ndimage

from baseline_deeplab.metrics import SegMetricsBF1
from .cldice import cldice_single


def _connectivity_ratio(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float | None:
    """Fraction of GT connected components that are 'hit' by the prediction.

    A GT component is considered preserved if at least 50% of its pixels
    are correctly predicted.  Returns None when the GT has no components
    for the given class (same skip policy as clDice / BF1).
    """
    gt_mask = (gt == class_id)
    if not gt_mask.any():
        return None
    pred_mask = (pred == class_id)
    labeled, n_comp = ndimage.label(gt_mask)
    if n_comp == 0:
        return None
    hit = 0
    for c in range(1, n_comp + 1):
        comp = (labeled == c)
        overlap = float((comp & pred_mask).sum()) / float(comp.sum())
        if overlap >= 0.5:
            hit += 1
    return hit / n_comp


class SegMetricsFull(SegMetricsBF1):
    """Confusion-matrix metrics + BF1 + clDice + connectivity ratio."""

    def __init__(self, num_classes: int, tol_px: float):
        super().__init__(num_classes, tol_px)
        self.cldice_sums: Dict[str, float] = {"crack": 0.0, "spalling": 0.0}
        self.cldice_counts: Dict[str, int] = {"crack": 0, "spalling": 0}
        self.conn_sums: Dict[str, float] = {"crack": 0.0, "spalling": 0.0}
        self.conn_counts: Dict[str, int] = {"crack": 0, "spalling": 0}

    def reset(self) -> None:
        super().reset()
        for k in self.cldice_sums:
            self.cldice_sums[k] = 0.0
            self.cldice_counts[k] = 0
            self.conn_sums[k] = 0.0
            self.conn_counts[k] = 0

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
                cr = _connectivity_ratio(pred[b], tgt[b], cid)
                if cr is not None:
                    self.conn_sums[name] += cr
                    self.conn_counts[name] += 1

    def compute(self) -> Dict[str, float]:
        m = super().compute()
        for name in ("crack", "spalling"):
            cnt = self.cldice_counts[name]
            m[f"clDice_{name}"] = (self.cldice_sums[name] / cnt) if cnt > 0 else 0.0
            cnt_c = self.conn_counts[name]
            m[f"ConnR_{name}"] = (self.conn_sums[name] / cnt_c) if cnt_c > 0 else 0.0
        m["clDice_fg_mean"] = 0.5 * (m["clDice_crack"] + m["clDice_spalling"])
        m["ConnR_fg_mean"] = 0.5 * (m["ConnR_crack"] + m["ConnR_spalling"])
        return m

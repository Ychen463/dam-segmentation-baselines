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


def _component_precision(pred: np.ndarray, gt: np.ndarray, class_id: int,
                         threshold: float = 0.5) -> float | None:
    """Fraction of predicted components that correspond to a GT component.

    A predicted component is 'valid' if at least `threshold` of its pixels
    overlap with GT pixels of the same class. Returns None when the
    prediction has no components for the given class.
    """
    pred_mask = (pred == class_id)
    if not pred_mask.any():
        return None
    gt_mask = (gt == class_id)
    labeled, n_comp = ndimage.label(pred_mask)
    if n_comp == 0:
        return None
    valid = 0
    for c in range(1, n_comp + 1):
        comp = (labeled == c)
        overlap = float((comp & gt_mask).sum()) / float(comp.sum())
        if overlap >= threshold:
            valid += 1
    return valid / n_comp


def _path_continuity(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float | None:
    """Fraction of GT skeleton paths preserved as connected in the prediction.

    For each GT connected component's skeleton, extract the endpoints
    (skeleton pixels with exactly 1 neighbor). If a component has >= 2
    endpoints, check whether all endpoints are mutually reachable through
    the predicted mask (using connected component analysis on the
    prediction restricted to the GT component's bounding region).

    Returns: fraction of GT skeleton components whose endpoint connectivity
    is fully preserved. Returns None if no GT components have >= 2 endpoints.
    """
    from skimage.morphology import skeletonize

    gt_mask = (gt == class_id)
    if not gt_mask.any():
        return None
    pred_mask = (pred == class_id)

    gt_labeled, n_gt = ndimage.label(gt_mask)
    testable = 0
    preserved = 0

    for c in range(1, n_gt + 1):
        comp_mask = (gt_labeled == c)
        skel = skeletonize(comp_mask)
        if skel.sum() < 2:
            continue

        # Find endpoints: skeleton pixels with exactly 1 neighbor
        # Use 8-connectivity neighbor count
        neighbor_count = ndimage.convolve(
            skel.astype(np.int32),
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
            mode='constant', cval=0
        )
        endpoints = skel & (neighbor_count == 1)
        ep_coords = np.argwhere(endpoints)
        if len(ep_coords) < 2:
            continue

        testable += 1

        # Check if all endpoints are in the same connected component
        # of the prediction within this GT component's region
        pred_in_region = pred_mask & comp_mask
        if not pred_in_region.any():
            continue

        pred_labeled, _ = ndimage.label(pred_in_region)
        ep_labels = set()
        for r, col in ep_coords:
            lbl = pred_labeled[r, col]
            if lbl == 0:  # endpoint not covered by prediction
                ep_labels.add(-1)  # sentinel for "not covered"
            else:
                ep_labels.add(lbl)

        if len(ep_labels) == 1 and -1 not in ep_labels:
            preserved += 1

    if testable == 0:
        return None
    return preserved / testable


class SegMetricsFull(SegMetricsBF1):
    """Confusion-matrix metrics + BF1 + clDice + connectivity ratio."""

    def __init__(self, num_classes: int, tol_px: float,
                 compute_topology: bool = False):
        super().__init__(num_classes, tol_px)
        self.compute_topology = compute_topology
        self.cldice_sums: Dict[str, float] = {"crack": 0.0, "spalling": 0.0}
        self.cldice_counts: Dict[str, int] = {"crack": 0, "spalling": 0}
        self.conn_sums: Dict[str, float] = {"crack": 0.0, "spalling": 0.0}
        self.conn_counts: Dict[str, int] = {"crack": 0, "spalling": 0}
        # Component precision + path continuity (topology mode)
        self.comp_prec_sums: Dict[str, float] = {"crack": 0.0, "spalling": 0.0}
        self.comp_prec_counts: Dict[str, int] = {"crack": 0, "spalling": 0}
        self.path_cont_sums: Dict[str, float] = {"crack": 0.0, "spalling": 0.0}
        self.path_cont_counts: Dict[str, int] = {"crack": 0, "spalling": 0}

    def reset(self) -> None:
        super().reset()
        for k in self.cldice_sums:
            self.cldice_sums[k] = 0.0
            self.cldice_counts[k] = 0
            self.conn_sums[k] = 0.0
            self.conn_counts[k] = 0
            self.comp_prec_sums[k] = 0.0
            self.comp_prec_counts[k] = 0
            self.path_cont_sums[k] = 0.0
            self.path_cont_counts[k] = 0

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
                if self.compute_topology:
                    cp = _component_precision(pred[b], tgt[b], cid)
                    if cp is not None:
                        self.comp_prec_sums[name] += cp
                        self.comp_prec_counts[name] += 1
                    pc = _path_continuity(pred[b], tgt[b], cid)
                    if pc is not None:
                        self.path_cont_sums[name] += pc
                        self.path_cont_counts[name] += 1

    def compute(self) -> Dict[str, float]:
        m = super().compute()
        for name in ("crack", "spalling"):
            cnt = self.cldice_counts[name]
            m[f"clDice_{name}"] = (self.cldice_sums[name] / cnt) if cnt > 0 else 0.0
            cnt_c = self.conn_counts[name]
            m[f"ConnR_{name}"] = (self.conn_sums[name] / cnt_c) if cnt_c > 0 else 0.0
        m["clDice_fg_mean"] = 0.5 * (m["clDice_crack"] + m["clDice_spalling"])
        m["ConnR_fg_mean"] = 0.5 * (m["ConnR_crack"] + m["ConnR_spalling"])

        if self.compute_topology:
            for name in ("crack", "spalling"):
                cnt_p = self.comp_prec_counts[name]
                m[f"CompP_{name}"] = (self.comp_prec_sums[name] / cnt_p) if cnt_p > 0 else 0.0
                cnt_r = self.conn_counts[name]
                cr = m[f"ConnR_{name}"]
                cp = m[f"CompP_{name}"]
                m[f"CompF1_{name}"] = (2 * cp * cr / (cp + cr)) if (cp + cr) > 0 else 0.0
                cnt_pc = self.path_cont_counts[name]
                m[f"PathCont_{name}"] = (self.path_cont_sums[name] / cnt_pc) if cnt_pc > 0 else 0.0
            m["CompP_fg_mean"] = 0.5 * (m["CompP_crack"] + m["CompP_spalling"])
            m["CompF1_fg_mean"] = 0.5 * (m["CompF1_crack"] + m["CompF1_spalling"])
            m["PathCont_fg_mean"] = 0.5 * (m["PathCont_crack"] + m["PathCont_spalling"])
        return m

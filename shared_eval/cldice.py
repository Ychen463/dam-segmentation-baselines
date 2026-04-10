"""clDice (centreline Dice) metric for thin-structure segmentation.

Algorithm
---------
Tprec = |skel(pred) ∩ gt| / |skel(pred)|
Tsens = |skel(gt) ∩ pred| / |skel(gt)|
clDice = 2 · Tprec · Tsens / (Tprec + Tsens)

Uses hard skeletonization via ``skimage.morphology.skeletonize``.
Returns ``None`` when either skeleton is empty (class absent or fully
covering), consistent with the BF1 skip policy.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from skimage.morphology import skeletonize


def cldice_single(
    pred: np.ndarray, gt: np.ndarray, class_id: int
) -> Optional[float]:
    """Compute clDice for a single (H, W) int-label pair and a given class.

    Returns None if either skeleton is empty so the caller can skip in the
    outer mean (same policy as ``boundary_f1_single``).
    """
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)

    # Skip if both are absent
    if not pred_mask.any() and not gt_mask.any():
        return None

    skel_pred = skeletonize(pred_mask)
    skel_gt = skeletonize(gt_mask)

    n_skel_pred = int(skel_pred.sum())
    n_skel_gt = int(skel_gt.sum())

    if n_skel_pred == 0 or n_skel_gt == 0:
        return None

    tprec = float((skel_pred & gt_mask).sum()) / float(n_skel_pred)
    tsens = float((skel_gt & pred_mask).sum()) / float(n_skel_gt)

    if tprec + tsens <= 0.0:
        return 0.0
    return 2.0 * tprec * tsens / (tprec + tsens)

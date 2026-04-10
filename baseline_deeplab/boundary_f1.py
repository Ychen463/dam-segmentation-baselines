"""Boundary F1 score with Euclidean distance-transform tolerance matching.

Standard DeepLab / PASCAL style BF1:
  - pred_boundary = dilate(pred_c) XOR erode(pred_c)
  - gt_boundary   = dilate(gt_c)   XOR erode(gt_c)
  - dt_gt   = EDT of (NOT gt_boundary)   -> distance from each pixel to nearest gt edge
  - dt_pred = EDT of (NOT pred_boundary) -> distance from each pixel to nearest pred edge
  - precision = | pred_boundary AND (dt_gt   <= tol) | / |pred_boundary|
  - recall    = | gt_boundary   AND (dt_pred <= tol) | / |gt_boundary|
  - BF1       = 2PR / (P+R)

If either boundary is empty the image is skipped (returns None) so the caller
aggregates with a `fg_present`-style policy, consistent with Dice handling.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
    import cv2  # type: ignore


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """3x3 structuring-element boundary = dilation XOR erosion."""
    if _HAVE_SCIPY:
        d = binary_dilation(mask, iterations=1)
        e = binary_erosion(mask, iterations=1)
        return np.logical_xor(d, e)
    # OpenCV fallback
    m = mask.astype(np.uint8)
    import cv2  # type: ignore
    k = np.ones((3, 3), np.uint8)
    d = cv2.dilate(m, k, iterations=1).astype(bool)
    e = cv2.erode(m, k, iterations=1).astype(bool)
    return np.logical_xor(d, e)


def _edt(non_boundary: np.ndarray) -> np.ndarray:
    """EDT where input is True OUTSIDE the boundary (we want distance to boundary)."""
    if _HAVE_SCIPY:
        return distance_transform_edt(non_boundary)
    import cv2  # type: ignore
    # cv2.distanceTransform needs foreground=255 where we want the distance FROM zero-pixels.
    src = non_boundary.astype(np.uint8) * 255
    return cv2.distanceTransform(src, cv2.DIST_L2, 3).astype(np.float64)


def boundary_f1_single(
    pred: np.ndarray, gt: np.ndarray, class_id: int, tol_px: float
) -> Optional[float]:
    """Compute BF1 for a single (H, W) int-label pair and a given foreground class.

    Returns None if either pred_boundary or gt_boundary is empty (class absent or
    fully-covering), so the image is skipped in the outer mean.
    """
    pb_mask = (pred == class_id)
    gb_mask = (gt == class_id)
    if not pb_mask.any() and not gb_mask.any():
        return None

    pred_bd = _binary_boundary(pb_mask)
    gt_bd = _binary_boundary(gb_mask)
    np_pred = int(pred_bd.sum())
    ng_gt = int(gt_bd.sum())
    if np_pred == 0 or ng_gt == 0:
        return None

    # Distance from any pixel to the nearest *boundary* pixel.
    dt_gt = _edt(~gt_bd)
    dt_pred = _edt(~pred_bd)

    prec = float((pred_bd & (dt_gt <= tol_px)).sum()) / float(np_pred)
    rec = float((gt_bd & (dt_pred <= tol_px)).sum()) / float(ng_gt)
    if prec + rec <= 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)

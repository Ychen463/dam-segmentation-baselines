"""CRF post-processing for segmentation refinement.

Uses DenseCRF (Krähenbühl & Koltun, NIPS 2011) to refine predictions
using image appearance cues (color + spatial).  Requires ``pydensecrf``.
"""
from __future__ import annotations

import numpy as np


def crf_refine(
    image_rgb: np.ndarray,
    pred: np.ndarray,
    n_classes: int = 3,
    n_iters: int = 5,
    sxy_gauss: int = 3,
    compat_gauss: float = 3.0,
    sxy_bilat: int = 80,
    srgb_bilat: int = 13,
    compat_bilat: float = 10.0,
) -> np.ndarray:
    """Refine an argmax prediction map with DenseCRF.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        pred: (H, W) int32/int64 argmax prediction (0..n_classes-1).
        n_classes: number of semantic classes.
        n_iters: CRF inference iterations.
        sxy_gauss: spatial std for smoothness kernel.
        compat_gauss: compatibility for smoothness kernel.
        sxy_bilat: spatial std for appearance kernel.
        srgb_bilat: color std for appearance kernel.
        compat_bilat: compatibility for appearance kernel.

    Returns:
        Refined (H, W) argmax prediction.
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    H, W = pred.shape
    d = dcrf.DenseCRF2D(W, H, n_classes)

    # Unary potentials from hard labels (with small uncertainty)
    unary = unary_from_labels(pred.astype(np.int32).ravel(), n_classes,
                              gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(unary)

    # Pairwise: smoothness kernel (spatial only)
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss)

    # Pairwise: appearance kernel (spatial + color)
    d.addPairwiseBilateral(
        sxy=sxy_bilat, srgb=srgb_bilat,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=compat_bilat,
    )

    Q = d.inference(n_iters)
    refined = np.argmax(Q, axis=0).reshape(H, W).astype(pred.dtype)
    return refined


def crf_refine_from_logits(
    image_rgb: np.ndarray,
    logits: np.ndarray,
    n_classes: int = 3,
    n_iters: int = 5,
    sxy_gauss: int = 3,
    compat_gauss: float = 3.0,
    sxy_bilat: int = 80,
    srgb_bilat: int = 13,
    compat_bilat: float = 10.0,
) -> np.ndarray:
    """Refine predictions using softmax logits as unary potentials.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        logits: (C, H, W) float32 logits (before or after softmax).
        n_classes: number of semantic classes.
        n_iters: CRF inference iterations.

    Returns:
        Refined (H, W) argmax prediction.
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import softmax_to_unary
    from scipy.special import softmax as sp_softmax

    C, H, W = logits.shape
    # Convert logits to probabilities
    probs = sp_softmax(logits, axis=0).astype(np.float32)
    # Flatten to (C, H*W) and convert to unary
    unary = softmax_to_unary(probs.reshape(C, -1))

    d = dcrf.DenseCRF2D(W, H, n_classes)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss)
    d.addPairwiseBilateral(
        sxy=sxy_bilat, srgb=srgb_bilat,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=compat_bilat,
    )

    Q = d.inference(n_iters)
    return np.argmax(Q, axis=0).reshape(H, W).astype(np.int64)

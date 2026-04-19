"""Morphology-aware crack feature extraction for MAC difficulty scoring.

Precomputes per-sample crack morphology features from GT masks:
- crack_mean_width: mean width via distance transform on skeleton
- crack_width_cv: coefficient of variation of crack width
- junction_density: skeleton junction points / skeleton length
- crack_components: number of independent crack connected components
- crack_spalling_proximity: fraction of crack pixels near spalling boundary
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class MorphFeatures:
    crack_mean_width: float = 0.0
    crack_width_cv: float = 0.0
    junction_density: float = 0.0
    crack_components: int = 0
    crack_spalling_proximity: float = 0.0
    crack_ratio: float = 0.0
    spalling_ratio: float = 0.0
    has_crack: bool = False
    has_spalling: bool = False


def compute_morph_features(mask: np.ndarray) -> MorphFeatures:
    """Extract crack morphology features from a GT mask.

    Args:
        mask: (H, W) int array, 0=background, 1=crack, 2=spalling.

    Returns:
        MorphFeatures dataclass.
    """
    from scipy.ndimage import distance_transform_edt, label
    from skimage.morphology import skeletonize

    h, w = mask.shape
    total_px = h * w

    crack_mask = (mask == 1)
    spalling_mask = (mask == 2)
    crack_px = crack_mask.sum()
    spalling_px = spalling_mask.sum()

    feat = MorphFeatures(
        crack_ratio=float(crack_px) / total_px,
        spalling_ratio=float(spalling_px) / total_px,
        has_crack=bool(crack_px > 0),
        has_spalling=bool(spalling_px > 0),
    )

    if crack_px < 10:
        return feat

    # 1. Skeleton + distance transform → crack width
    skel = skeletonize(crack_mask)
    skel_px = skel.sum()
    if skel_px < 2:
        return feat

    dt = distance_transform_edt(crack_mask)
    widths = dt[skel]  # distance from skeleton to boundary = half-width
    widths = widths * 2.0  # full width estimate
    feat.crack_mean_width = float(widths.mean())
    if feat.crack_mean_width > 0:
        feat.crack_width_cv = float(widths.std() / (widths.mean() + 1e-6))

    # 2. Junction density: convolve skeleton with 3x3 ones, junctions have >= 4 neighbors
    skel_u8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    from scipy.signal import fftconvolve
    neighbor_count = fftconvolve(skel_u8, kernel, mode='same')
    # Junction = skeleton pixel with 4+ neighbors (including itself)
    junctions = skel & (neighbor_count >= 4)
    feat.junction_density = float(junctions.sum()) / float(skel_px)

    # 3. Connected components
    labeled, n_comp = label(crack_mask)
    feat.crack_components = int(n_comp)

    # 4. Crack-spalling proximity
    if spalling_px > 0 and crack_px > 0:
        dt_crack = distance_transform_edt(~crack_mask)
        near_crack = dt_crack[spalling_mask] < 10.0
        feat.crack_spalling_proximity = float(near_crack.sum()) / (float(spalling_px) + 1e-6)

    return feat


def precompute_morph_cache(
    records: List[Dict],
    data_root: Path,
    cache_path: Path,
) -> Dict[str, MorphFeatures]:
    """Batch-precompute morphology features for all training records.

    Uses pickle cache. If cache exists and has matching record count, loads it.

    Args:
        records: list of record dicts with "id" and "rel" keys.
        data_root: dataset root directory.
        cache_path: path to pickle cache file.

    Returns:
        Dict mapping sample_id -> MorphFeatures.
    """
    from baseline_unet.dataset import decode_mask, mask_path, read_mask_rgb

    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if len(cached) == len(records):
            print(f"[morph] loaded cache ({len(cached)} samples) from {cache_path}")
            return cached
        print(f"[morph] cache size mismatch ({len(cached)} vs {len(records)}), recomputing")

    print(f"[morph] computing morphology features for {len(records)} samples ...")
    cache: Dict[str, MorphFeatures] = {}
    for i, rec in enumerate(records):
        sid = rec["id"]
        rel = rec["rel"]
        mask_rgb = read_mask_rgb(mask_path(data_root, rel))
        label_mask, _ = decode_mask(mask_rgb)
        cache[sid] = compute_morph_features(label_mask)
        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            print(f"  [{i+1}/{len(records)}]", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[morph] saved cache to {cache_path}")
    return cache

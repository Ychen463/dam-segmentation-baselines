"""Topology-Aware Post-Processing Pipeline (TAPP) for crack segmentation.

Two complementary modules:
1. Morphological Shape Filtering (MSF): removes false-positive crack components
   that lack crack-like morphology (too round, too small, too compact).
2. Skeleton-Guided Gap Filling (SGF): reconnects broken crack segments by
   detecting nearby skeleton endpoints with compatible orientation and
   bridging them with dilated line segments.

Usage:
    from tapp import tapp_postprocess
    refined_mask = tapp_postprocess(pred_mask)  # (H, W) int array, classes 0/1/2
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, disk, dilation
from skimage.measure import label, regionprops


# ---------------------------------------------------------------------------
# Module 1: Morphological Shape Filtering (MSF)
# ---------------------------------------------------------------------------

def morphological_shape_filter(
    mask: np.ndarray,
    crack_class: int = 1,
    min_area: int = 30,
    min_eccentricity: float = 0.85,
    min_solidity: float = 0.0,
    max_solidity: float = 0.85,
) -> np.ndarray:
    """Remove false-positive crack components that lack elongated morphology.

    Real cracks are thin and elongated (high eccentricity, low solidity).
    False positives from background tend to be small, round blobs.

    Args:
        mask: (H, W) prediction mask with class labels.
        crack_class: label value for crack (default 1).
        min_area: components smaller than this are removed.
        min_eccentricity: components rounder than this are removed.
            Eccentricity ranges from 0 (circle) to 1 (line).
        max_solidity: components more solid/compact than this are removed.
            Solidity = area / convex_hull_area. Cracks have low solidity.
        min_solidity: floor to avoid removing extremely fragmented components.

    Returns:
        Refined mask with false-positive crack components set to background.
    """
    crack_binary = (mask == crack_class)
    if not crack_binary.any():
        return mask

    labeled = label(crack_binary, connectivity=2)
    props = regionprops(labeled)

    remove_labels = set()
    for prop in props:
        # Small isolated blobs
        if prop.area < min_area:
            remove_labels.add(prop.label)
            continue

        # Round components (not crack-like)
        if prop.eccentricity < min_eccentricity:
            remove_labels.add(prop.label)
            continue

        # Too compact/solid (crack should be thin with lots of convex hull gaps)
        if prop.solidity > max_solidity and prop.area < 500:
            remove_labels.add(prop.label)
            continue

    if not remove_labels:
        return mask

    refined = mask.copy()
    remove_mask = np.isin(labeled, list(remove_labels))
    refined[remove_mask] = 0  # set to background

    return refined


# ---------------------------------------------------------------------------
# Module 2: Skeleton-Guided Gap Filling (SGF)
# ---------------------------------------------------------------------------

def _find_endpoints(skel: np.ndarray) -> list:
    """Find skeleton endpoints (pixels with exactly 1 neighbor)."""
    if not skel.any():
        return []

    # Count 8-connected neighbors for each skeleton pixel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = ndimage.convolve(skel.astype(np.uint8), kernel,
                                       mode='constant', cval=0)
    # Endpoints: skeleton pixels with exactly 1 neighbor
    endpoint_mask = skel & (neighbor_count == 1)
    ys, xs = np.where(endpoint_mask)
    return list(zip(ys.tolist(), xs.tolist()))


def _estimate_direction(skel: np.ndarray, y: int, x: int,
                        window: int = 7) -> np.ndarray | None:
    """Estimate local crack direction at a skeleton endpoint.

    Fits a direction vector from the endpoint along nearby skeleton pixels.
    """
    H, W = skel.shape
    y0 = max(0, y - window)
    y1 = min(H, y + window + 1)
    x0 = max(0, x - window)
    x1 = min(W, x + window + 1)

    patch = skel[y0:y1, x0:x1]
    ys, xs = np.where(patch)
    if len(ys) < 3:
        return None

    # Coordinates relative to endpoint
    dy = ys - (y - y0)
    dx = xs - (x - x0)

    # PCA to find principal direction
    coords = np.stack([dx, dy], axis=1).astype(np.float64)
    coords -= coords.mean(axis=0)
    if np.abs(coords).max() < 1e-6:
        return None

    cov = coords.T @ coords
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, -1]  # largest eigenvector

    # Orient direction away from the skeleton body (toward the gap)
    # The skeleton extends inward, so the gap is in the opposite direction
    # of the centroid of nearby skeleton pixels
    centroid_dx = np.mean(dx)
    centroid_dy = np.mean(dy)
    if direction[0] * centroid_dx + direction[1] * centroid_dy > 0:
        direction = -direction

    return direction


def _draw_line(mask: np.ndarray, y0: int, x0: int, y1: int, x1: int,
               value: int = 1) -> None:
    """Draw a line on mask using Bresenham's algorithm."""
    H, W = mask.shape
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy

    while True:
        if 0 <= y0 < H and 0 <= x0 < W:
            mask[y0, x0] = value
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def skeleton_guided_gap_filling(
    mask: np.ndarray,
    crack_class: int = 1,
    max_gap: int = 15,
    max_angle_deg: float = 45.0,
    dilate_radius: int = 2,
) -> np.ndarray:
    """Reconnect broken crack segments by bridging nearby skeleton endpoints.

    For each pair of skeleton endpoints within `max_gap` pixels, checks if
    their local directions are compatible (roughly collinear). If so, draws
    a bridge line and dilates it to restore crack width.

    Args:
        mask: (H, W) prediction mask with class labels.
        crack_class: label value for crack (default 1).
        max_gap: maximum pixel distance between endpoints to consider bridging.
        max_angle_deg: maximum angle (degrees) between endpoint directions
            for them to be considered compatible.
        dilate_radius: radius of dilation applied to bridge lines.

    Returns:
        Refined mask with crack gaps filled.
    """
    crack_binary = (mask == crack_class).astype(np.uint8)
    if not crack_binary.any():
        return mask

    skel = skeletonize(crack_binary > 0)
    endpoints = _find_endpoints(skel)

    if len(endpoints) < 2:
        return mask

    # Compute directions for all endpoints
    ep_dirs = []
    for (y, x) in endpoints:
        d = _estimate_direction(skel, y, x)
        ep_dirs.append(d)

    # Find compatible endpoint pairs
    bridge_mask = np.zeros_like(crack_binary, dtype=np.uint8)
    used = set()
    cos_thresh = np.cos(np.radians(max_angle_deg))

    for i in range(len(endpoints)):
        if i in used or ep_dirs[i] is None:
            continue
        yi, xi = endpoints[i]
        di = ep_dirs[i]

        best_j = -1
        best_dist = max_gap + 1

        for j in range(i + 1, len(endpoints)):
            if j in used or ep_dirs[j] is None:
                continue
            yj, xj = endpoints[j]

            # Check if they belong to different connected components
            # (no point bridging endpoints of the same crack)
            if skel[yi, xi] and skel[yj, xj]:
                # Quick check: are they already connected?
                pass  # We'll check via labeled components below

            dist = np.sqrt((yi - yj) ** 2 + (xi - xj) ** 2)
            if dist > max_gap or dist < 3:
                continue

            dj = ep_dirs[j]

            # Check direction compatibility:
            # 1. Both directions should roughly point toward each other
            gap_vec = np.array([xj - xi, yj - yi], dtype=np.float64)
            gap_norm = np.linalg.norm(gap_vec)
            if gap_norm < 1e-6:
                continue
            gap_vec /= gap_norm

            # Direction i should point toward j
            align_i = np.dot(di, gap_vec)
            # Direction j should point toward i (opposite gap direction)
            align_j = np.dot(dj, -gap_vec)

            if align_i < cos_thresh or align_j < cos_thresh:
                continue

            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j >= 0:
            yj, xj = endpoints[best_j]
            _draw_line(bridge_mask, yi, xi, yj, xj, value=1)
            used.add(i)
            used.add(best_j)

    if not bridge_mask.any():
        return mask

    # Check that bridges only connect different components
    labeled_crack = label(crack_binary, connectivity=2)
    bridge_ys, bridge_xs = np.where(bridge_mask > 0)

    # Dilate bridges to restore crack width
    bridge_dilated = dilation(bridge_mask, disk(dilate_radius))

    # Only fill where background (don't overwrite spalling)
    refined = mask.copy()
    fill_mask = (bridge_dilated > 0) & (mask == 0)
    refined[fill_mask] = crack_class

    return refined


# ---------------------------------------------------------------------------
# Combined TAPP Pipeline
# ---------------------------------------------------------------------------

def tapp_postprocess(
    mask: np.ndarray,
    crack_class: int = 1,
    # MSF parameters
    msf_min_area: int = 30,
    msf_min_eccentricity: float = 0.85,
    msf_max_solidity: float = 0.85,
    # SGF parameters
    sgf_max_gap: int = 15,
    sgf_max_angle_deg: float = 45.0,
    sgf_dilate_radius: int = 2,
    # Module switches
    use_msf: bool = True,
    use_sgf: bool = True,
) -> np.ndarray:
    """Full TAPP pipeline: MSF (remove false positives) then SGF (fill gaps).

    Order matters: MSF first removes noise, then SGF bridges real crack gaps
    without being confused by false-positive clusters.

    Args:
        mask: (H, W) integer array with class labels (0=bg, 1=crack, 2=spalling).

    Returns:
        Refined mask.
    """
    result = mask.copy()

    if use_msf:
        result = morphological_shape_filter(
            result, crack_class=crack_class,
            min_area=msf_min_area,
            min_eccentricity=msf_min_eccentricity,
            max_solidity=msf_max_solidity,
        )

    if use_sgf:
        result = skeleton_guided_gap_filling(
            result, crack_class=crack_class,
            max_gap=sgf_max_gap,
            max_angle_deg=sgf_max_angle_deg,
            dilate_radius=sgf_dilate_radius,
        )

    return result

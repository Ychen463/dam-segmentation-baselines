"""Per-group evaluation for feature-cluster-aware training.

Tracks per-image confusion matrices, aggregates by group to compute
robustness metrics (worst-group, CVaR, percentiles).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class PerGroupEvaluator:
    """Track per-image confusion matrices, aggregate by group."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.image_data: List[Tuple[int, np.ndarray]] = []

    def update(self, seg_logits: torch.Tensor, masks: torch.Tensor,
               group_ids: List[int]) -> None:
        """Record per-image CM for each sample in batch."""
        preds = seg_logits.argmax(1).cpu().numpy()  # (B, H, W)
        gt = masks.cpu().numpy()
        nc = self.num_classes
        for i in range(preds.shape[0]):
            p_flat = preds[i].ravel()
            g_flat = gt[i].ravel()
            # Fast confusion matrix via bincount
            cm = np.bincount(nc * g_flat + p_flat,
                             minlength=nc * nc).reshape(nc, nc).astype(np.int64)
            self.image_data.append((group_ids[i], cm))

    def compute_group_metrics(self) -> Dict:
        """Aggregate per-group, compute robustness metrics."""
        nc = self.num_classes

        # Group CMs by group_id
        group_cms: Dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros((nc, nc), dtype=np.int64))
        group_counts: Dict[int, int] = defaultdict(int)
        for gid, cm in self.image_data:
            group_cms[gid] += cm
            group_counts[gid] += 1

        # Per-group metrics
        per_group: Dict[int, Dict] = {}
        all_miou_fg = []
        eligible_miou_fg = []  # groups with >= 5 images
        all_crack_recall = []
        all_spalling_recall = []

        for gid in sorted(group_cms.keys()):
            cm = group_cms[gid]
            n_images = group_counts[gid]

            # Per-class IoU (present-class rule: NaN if class absent in GT)
            ious = {}
            present_fg_ious = []
            for c in range(nc):
                gt_pixels = cm[c, :].sum()
                if gt_pixels == 0:
                    ious[c] = float('nan')
                else:
                    tp = cm[c, c]
                    fp = cm[:, c].sum() - tp
                    fn = gt_pixels - tp
                    iou = tp / (tp + fp + fn + 1e-8)
                    ious[c] = float(iou)
                    if c > 0:  # foreground
                        present_fg_ious.append(float(iou))

            miou_fg = float(np.mean(present_fg_ious)) if present_fg_ious else float('nan')

            # Crack recall (class 1)
            crack_gt = cm[1, :].sum()
            crack_recall = float(cm[1, 1] / (crack_gt + 1e-8)) if crack_gt > 0 else float('nan')

            # Spalling recall (class 2)
            spall_gt = cm[2, :].sum()
            spall_recall = float(cm[2, 2] / (spall_gt + 1e-8)) if spall_gt > 0 else float('nan')

            per_group[gid] = {
                'n_images': n_images,
                'iou_bg': ious[0],
                'iou_crack': ious.get(1, float('nan')),
                'iou_spalling': ious.get(2, float('nan')),
                'mIoU_fg': miou_fg,
                'crack_recall': crack_recall,
                'spalling_recall': spall_recall,
            }

            if not np.isnan(miou_fg):
                all_miou_fg.append(miou_fg)
                if n_images >= 5:
                    eligible_miou_fg.append(miou_fg)
            if not np.isnan(crack_recall):
                all_crack_recall.append(crack_recall)
            if not np.isnan(spall_recall):
                all_spalling_recall.append(spall_recall)

        # Summary metrics
        arr = np.array(all_miou_fg) if all_miou_fg else np.array([0.0])
        n_groups = len(arr)

        worst_group_mIoU_fg = float(arr.min()) if len(arr) > 0 else 0.0
        eligible_arr = np.array(eligible_miou_fg) if eligible_miou_fg else arr
        eligible_worst_group_mIoU_fg = float(eligible_arr.min()) if len(eligible_arr) > 0 else 0.0

        p10_mIoU_fg = float(np.percentile(arr, 10)) if len(arr) > 0 else 0.0

        # CVaR20: mean of worst 20% groups
        k = max(1, int(np.ceil(0.2 * n_groups)))
        sorted_miou = np.sort(arr)
        cvar20_mIoU_fg = float(sorted_miou[:k].mean())

        std_mIoU_fg = float(arr.std()) if len(arr) > 1 else 0.0
        mean_mIoU_fg = float(arr.mean())
        avg_worst_gap = mean_mIoU_fg - worst_group_mIoU_fg

        worst_crack_recall = float(np.min(all_crack_recall)) if all_crack_recall else float('nan')
        worst_spalling_recall = float(np.min(all_spalling_recall)) if all_spalling_recall else float('nan')

        summary = {
            'n_groups_evaluated': n_groups,
            'mean_mIoU_fg': mean_mIoU_fg,
            'worst_group_mIoU_fg': worst_group_mIoU_fg,
            'eligible_worst_group_mIoU_fg': eligible_worst_group_mIoU_fg,
            'p10_mIoU_fg': p10_mIoU_fg,
            'cvar20_mIoU_fg': cvar20_mIoU_fg,
            'std_mIoU_fg': std_mIoU_fg,
            'avg_worst_gap': avg_worst_gap,
            'worst_crack_recall': worst_crack_recall,
            'worst_spalling_recall': worst_spalling_recall,
        }

        return {'summary': summary, 'per_group': per_group}

    def reset(self) -> None:
        self.image_data.clear()

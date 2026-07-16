"""Group-aware samplers for feature-cluster-aware training.

Provides weighted sampling based on group membership (135 ResNet-50 feature
clusters), plus GroupDRO and JTT managers for Phase 2 experiments.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

import numpy as np
import torch
from torch.utils.data import Sampler


def load_group_assignments(path: Path) -> Dict[str, int]:
    """Load group_assignments.json -> {image_filename: group_id}."""
    with open(path) as f:
        return json.load(f)


class GroupAwareSampler(Sampler[int]):
    """Weighted random sampler based on group membership.

    Modes:
    - "random_replace": equal weight per sample, with replacement (for G0R).
    - "group_uniform_capped": w[i] = 1/max(group_size, cap). With replacement.
    - "inverse_sqrt": w[i] = 1/sqrt(group_size + k). With replacement.

    All modes sample len(dataset) items WITH replacement.
    """

    def __init__(self, records: List[Dict], group_map: Dict[str, int],
                 mode: str = "random_replace",
                 cap: int = 5, smooth_k: int = 3,
                 num_samples: Optional[int] = None):
        self.records = records
        self.group_map = group_map
        self.mode = mode
        self.cap = cap
        self.smooth_k = smooth_k
        self.n = num_samples or len(records)

        # Compute group sizes (within the records provided)
        self._group_sizes: Dict[int, int] = Counter()
        self._sample_groups: List[int] = []
        for rec in records:
            gid = group_map.get(rec["id"], group_map.get(rec["rel"], -1))
            self._sample_groups.append(gid)
            self._group_sizes[gid] += 1

        # Compute weights
        self._weights = self._compute_weights()
        self._last_indices: List[int] = []

    def _compute_weights(self) -> np.ndarray:
        n = len(self.records)
        w = np.ones(n, dtype=np.float64)

        if self.mode == "random_replace":
            # Equal weight, sampling with replacement
            pass

        elif self.mode == "group_uniform_capped":
            for i, gid in enumerate(self._sample_groups):
                gs = self._group_sizes[gid]
                w[i] = 1.0 / max(gs, self.cap)

        elif self.mode == "inverse_sqrt":
            for i, gid in enumerate(self._sample_groups):
                gs = self._group_sizes[gid]
                w[i] = 1.0 / math.sqrt(gs + self.smooth_k)

        else:
            raise ValueError(f"Unknown GroupAwareSampler mode: {self.mode}")

        # Normalize
        w /= w.sum()
        return w

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[int]:
        indices = np.random.choice(len(self.records), size=self.n,
                                   replace=True, p=self._weights)
        self._last_indices = indices.tolist()
        return iter(self._last_indices)

    def get_sampling_stats(self) -> Dict:
        """Per-epoch sampling statistics for logging."""
        if not self._last_indices:
            return {}

        # Per-group sample count
        group_counts: Dict[int, int] = Counter()
        image_counts: Dict[int, int] = Counter()
        for idx in self._last_indices:
            gid = self._sample_groups[idx]
            group_counts[gid] += 1
            image_counts[idx] += 1

        gc_vals = list(group_counts.values()) if group_counts else [0]
        unique_images = len(set(self._last_indices))
        total = len(self._last_indices)

        # Effective sample size: (sum w)^2 / sum w^2
        w_selected = self._weights[self._last_indices]
        ess = (w_selected.sum() ** 2) / (w_selected ** 2).sum() if w_selected.sum() > 0 else 0

        return {
            'mode': self.mode,
            'group_sample_min': min(gc_vals),
            'group_sample_max': max(gc_vals),
            'group_sample_mean': float(np.mean(gc_vals)),
            'max_image_repeat': max(image_counts.values()) if image_counts else 0,
            'unique_images': unique_images,
            'total_samples': total,
            'coverage_ratio': unique_images / len(self.records) if self.records else 0,
            'effective_sample_size': float(ess),
        }


class GroupDROTracker:
    """Corrected GroupDRO: optimizes sum q_g * L_bar_g.

    Per-sample weight: w_i = q_{g_i} / p_{g_i}
    where p_g = n_g / N (group's natural frequency in train set).

    Maintains log_q[g], updated epoch-end:
        log_q[g] += eta * ema_loss[g]   (only for groups seen this epoch)
        q = softmax(log_q)
    """

    def __init__(self, group_sizes: Dict[int, int], eta: float = 0.1,
                 max_weight: float = 10.0, ema_decay: float = 0.9):
        self.group_ids = sorted(group_sizes.keys())
        self.n_groups = len(self.group_ids)
        self.gid_to_idx = {g: i for i, g in enumerate(self.group_ids)}
        total = sum(group_sizes.values())
        self.p = np.array([group_sizes[g] / total for g in self.group_ids],
                          dtype=np.float64)
        self.eta = eta
        self.max_weight = max_weight
        self.ema_decay = ema_decay

        # State
        self.log_q = np.zeros(self.n_groups, dtype=np.float64)
        self.q = np.ones(self.n_groups, dtype=np.float64) / self.n_groups
        self.ema_loss = np.zeros(self.n_groups, dtype=np.float64)
        self._epoch_loss_sum = np.zeros(self.n_groups, dtype=np.float64)
        self._epoch_loss_count = np.zeros(self.n_groups, dtype=np.int64)

    def accumulate(self, group_ids: List[int], per_sample_loss: torch.Tensor) -> None:
        """Accumulate per-sample losses within an epoch."""
        losses = per_sample_loss.detach().cpu().numpy()
        for gid, loss_val in zip(group_ids, losses):
            if gid not in self.gid_to_idx:
                continue
            idx = self.gid_to_idx[gid]
            self._epoch_loss_sum[idx] += loss_val
            self._epoch_loss_count[idx] += 1

    def step(self) -> None:
        """End-of-epoch update: update EMA losses and q distribution."""
        for i in range(self.n_groups):
            if self._epoch_loss_count[i] > 0:
                epoch_mean = self._epoch_loss_sum[i] / self._epoch_loss_count[i]
                self.ema_loss[i] = (self.ema_decay * self.ema_loss[i] +
                                    (1 - self.ema_decay) * epoch_mean)
                self.log_q[i] += self.eta * self.ema_loss[i]

        # Softmax for q
        log_q_shifted = self.log_q - self.log_q.max()
        exp_q = np.exp(log_q_shifted)
        self.q = exp_q / exp_q.sum()

        # Reset epoch accumulators
        self._epoch_loss_sum[:] = 0
        self._epoch_loss_count[:] = 0

    def get_sample_weight(self, group_id: int) -> float:
        """Get per-sample weight for a given group."""
        if group_id not in self.gid_to_idx:
            return 1.0
        idx = self.gid_to_idx[group_id]
        raw = self.q[idx] / (self.p[idx] + 1e-10)
        return min(raw, self.max_weight)

    def get_sample_weights_batch(self, group_ids: List[int]) -> torch.Tensor:
        """Get weights for a batch of samples."""
        weights = torch.tensor([self.get_sample_weight(g) for g in group_ids],
                               dtype=torch.float32)
        return weights


class JTTManager:
    """True two-stage JTT with re-initialization.

    Stage 1: Train from T1-init for stage1_epochs, then identify errors.
    Stage 2: Re-initialize from T1, train with upweighted error samples.
    """

    def __init__(self, records: List[Dict], group_map: Dict[str, int],
                 stage1_epochs: int = 10, upweight: float = 3.0,
                 error_quantile: float = 0.2, num_classes: int = 3):
        self.records = records
        self.group_map = group_map
        self.stage1_epochs = stage1_epochs
        self.upweight = upweight
        self.error_quantile = error_quantile
        self.num_classes = num_classes
        self.error_set: Set[str] = set()
        self._identified = False

    @property
    def stage1_done(self) -> bool:
        return self._identified

    def identify_errors(self, model, loader, device) -> Set[str]:
        """Run inference on train set, rank by present-class mIoU.

        Bottom error_quantile fraction = error set.
        """
        import torch.nn.functional as F

        model.eval()
        sample_ious: List[tuple] = []  # (sample_id, present_class_miou)

        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device, non_blocking=True).float()
                masks = batch["mask"].to(device, non_blocking=True).long()
                sample_ids = batch["sample_id"]

                outputs = model(imgs)
                seg_logits = F.interpolate(outputs["seg_logits"].float(),
                                           masks.shape[-2:], mode="bilinear",
                                           align_corners=False)
                preds = seg_logits.argmax(1)  # (B, H, W)

                for i in range(preds.shape[0]):
                    p = preds[i].cpu().numpy().ravel()
                    g = masks[i].cpu().numpy().ravel()
                    nc = self.num_classes
                    cm = np.bincount(nc * g + p, minlength=nc * nc).reshape(nc, nc)

                    present_ious = []
                    for c in range(nc):
                        gt_c = cm[c, :].sum()
                        if gt_c == 0:
                            continue
                        tp = cm[c, c]
                        fp = cm[:, c].sum() - tp
                        fn = gt_c - tp
                        iou = tp / (tp + fp + fn + 1e-8)
                        present_ious.append(float(iou))

                    miou = float(np.mean(present_ious)) if present_ious else 1.0
                    sample_ious.append((sample_ids[i], miou))

        # Sort by mIoU (ascending = worst first)
        sample_ious.sort(key=lambda x: x[1])
        n_error = max(1, int(len(sample_ious) * self.error_quantile))
        self.error_set = {sid for sid, _ in sample_ious[:n_error]}
        self._identified = True

        # Stats
        error_ious = [m for _, m in sample_ious[:n_error]]
        clean_ious = [m for _, m in sample_ious[n_error:]]
        print(f"[JTT] Identified {n_error}/{len(sample_ious)} error samples "
              f"(bottom {self.error_quantile:.0%})")
        print(f"[JTT] Error mIoU: mean={np.mean(error_ious):.4f} "
              f"max={np.max(error_ious):.4f}")
        print(f"[JTT] Clean mIoU: mean={np.mean(clean_ious):.4f} "
              f"min={np.min(clean_ious):.4f}")

        return self.error_set

    def get_sampler_weights(self, records: List[Dict]) -> List[float]:
        """Return per-sample weights: upweight for error set, 1.0 for others."""
        weights = []
        for rec in records:
            if rec["id"] in self.error_set:
                weights.append(self.upweight)
            else:
                weights.append(1.0)
        return weights

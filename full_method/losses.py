"""Loss functions for the full method.

- Per-sample CE/Entropy (for difficulty estimation, not for backward)
- TverskyLoss (crack-specific, per-sample then mean)
- BoundaryBCELoss (crack boundary only)
- Foreground Dice loss (skip absent classes)
- CompositeLoss (combines all + schedules weights)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


# ---------------------------------------------------------------------------
# Per-sample helpers (for difficulty estimation — detach before use!)
# ---------------------------------------------------------------------------

def per_sample_ce(logits: torch.Tensor, targets: torch.Tensor,
                  class_weights: torch.Tensor) -> torch.Tensor:
    """Per-sample mean cross-entropy. Returns (B,) tensor."""
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')  # (B,H,W)
    return ce.mean(dim=(1, 2))  # (B,)


def per_sample_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample mean prediction entropy. Returns (B,) tensor."""
    prob = torch.softmax(logits, dim=1)
    ent = -(prob * prob.clamp_min(1e-8).log()).sum(dim=1)  # (B, H, W)
    return ent.mean(dim=(1, 2))  # (B,)


def per_sample_class_ce(logits: torch.Tensor, targets: torch.Tensor,
                        class_idx: int) -> torch.Tensor:
    """Per-sample mean CE on pixels of a specific class. Returns (B,) tensor."""
    mask = (targets == class_idx).float()                    # (B, H, W)
    ce = F.cross_entropy(logits, targets, reduction='none')  # (B, H, W)
    return (ce * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)


# ---------------------------------------------------------------------------
# Tversky loss (crack-specific)
# ---------------------------------------------------------------------------

class TverskyLoss(nn.Module):
    """Binary Tversky for crack (class=1). Per-sample aggregation, then batch mean."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        crack_prob = torch.softmax(logits, dim=1)[:, 1:2]      # (B,1,H,W)
        crack_gt = (targets == 1).float().unsqueeze(1)           # (B,1,H,W)
        # Per-sample aggregation
        tp = (crack_prob * crack_gt).sum(dim=(1, 2, 3))          # (B,)
        fp = (crack_prob * (1 - crack_gt)).sum(dim=(1, 2, 3))    # (B,)
        fn = ((1 - crack_prob) * crack_gt).sum(dim=(1, 2, 3))    # (B,)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - tversky.mean()


# ---------------------------------------------------------------------------
# Boundary BCE loss (crack boundary only)
# ---------------------------------------------------------------------------

def boundary_bce_loss(boundary_logits: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """BCE loss on crack boundary (dilated - eroded crack mask)."""
    crack = (mask == 1).float().unsqueeze(1)       # (B,1,H,W)
    dil = F.max_pool2d(crack, 3, 1, 1)
    ero = -F.max_pool2d(-crack, 3, 1, 1)
    boundary_gt = (dil != ero).float()              # (B,1,H,W)
    return F.binary_cross_entropy_with_logits(boundary_logits, boundary_gt)


# ---------------------------------------------------------------------------
# Soft-clDice (topology-preserving loss for thin structures)
# ---------------------------------------------------------------------------

def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)

def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), stride=1, padding=1)

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iters: int):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

def soft_cldice_loss(prob, target, iters=7, eps=1e-6):
    """Soft clDice loss on (B,1,H,W) probability and binary target."""
    sp = soft_skel(prob, iters)
    sl = soft_skel(target, iters)
    tprec = (sp * target).sum(dim=(1, 2, 3)) / (sp.sum(dim=(1, 2, 3)) + eps)
    tsens = (sl * prob).sum(dim=(1, 2, 3)) / (sl.sum(dim=(1, 2, 3)) + eps)
    cl = 2 * tprec * tsens / (tprec + tsens + eps)
    return 1 - cl.mean()


# ---------------------------------------------------------------------------
# Skeleton Recall Loss (SRL) — Kirchhoff et al., ECCV 2024
# ---------------------------------------------------------------------------

def skeleton_recall_loss(pred_prob, skel_gt, eps=1e-6):
    """Skeleton Recall Loss: mean predicted probability at GT skeleton pixels.

    Much lighter than soft-clDice (no online skeletonization needed).
    GT skeletons are precomputed offline in the dataset.

    Args:
        pred_prob: (B,1,H,W) soft probabilities for the target class.
        skel_gt:   (B,1,H,W) binary skeleton of GT mask (precomputed).

    Returns:
        Scalar loss = 1 - mean(pred_prob at skeleton pixels).
        Returns 0 for samples with empty skeletons (no crack present).
    """
    numer = (pred_prob * skel_gt).sum(dim=(1, 2, 3))
    denom = skel_gt.sum(dim=(1, 2, 3))                # (B,)
    has_skel = denom > 0                               # mask out empty skeletons
    if not has_skel.any():
        return pred_prob.new_zeros(())
    skel_recall = numer[has_skel] / (denom[has_skel] + eps)
    return 1.0 - skel_recall.mean()


# ---------------------------------------------------------------------------
# Snake branch auxiliary loss (crack-only focal BCE + Dice)
# ---------------------------------------------------------------------------

def snake_aux_loss(crack_enhance: torch.Tensor, targets: torch.Tensor,
                   alpha: float = 0.75, gamma: float = 2.0,
                   eps: float = 1e-6) -> torch.Tensor:
    """Auxiliary loss on the snake branch crack enhancement logits.

    Focal BCE + Dice, both applied to the binary crack mask.
    """
    crack_gt = (targets == 1).float().unsqueeze(1)  # (B,1,H,W)
    logits = F.interpolate(crack_enhance, size=targets.shape[-2:],
                           mode="bilinear", align_corners=False)
    # Focal BCE
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, crack_gt, reduction='none')
    pt = crack_gt * p + (1 - crack_gt) * (1 - p)
    alpha_t = crack_gt * alpha + (1 - crack_gt) * (1 - alpha)
    focal = (alpha_t * (1 - pt) ** gamma * bce).mean()
    # Dice
    inter = (p * crack_gt).sum()
    card = p.sum() + crack_gt.sum()
    dice = 1.0 - (2.0 * inter + eps) / (card + eps)
    return 0.5 * focal + 0.5 * dice


# ---------------------------------------------------------------------------
# Foreground Dice loss (reusable, matches baseline_unet.losses logic)
# ---------------------------------------------------------------------------

def fg_dice_loss(logits: torch.Tensor, targets: torch.Tensor,
                 num_classes: int = C.NUM_CLASSES, eps: float = 1e-6) -> torch.Tensor:
    """Foreground-only Dice loss with absent-class skip."""
    probs = logits.softmax(dim=1)
    onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    card = probs.sum(dims) + onehot.sum(dims)
    dice = (2.0 * inter + eps) / (card + eps)  # [C]

    fg_dice = dice[1:]
    fg_present = (onehot[:, 1:].sum(dims) > 0)
    if fg_present.any():
        return 1.0 - fg_dice[fg_present].mean()
    return logits.new_zeros(())


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """Full method composite loss with scheduled crack/boundary weights."""

    def __init__(self, ce_weight: torch.Tensor, cfg: C.RunCfg = None):
        super().__init__()
        if cfg is None:
            cfg = C.RunCfg()
        self.cfg = cfg
        self.register_buffer("ce_weight", ce_weight.float())
        self.tversky = TverskyLoss(alpha=cfg.loss_tversky_alpha,
                                   beta=cfg.loss_tversky_beta)

    def forward(self, outputs: dict, targets: torch.Tensor,
                scheduler, epoch: int,
                sample_weights: torch.Tensor = None,
                crack_skel: torch.Tensor = None,
                mac_ce_multipliers: tuple = None,
                mac_topo_multiplier: float = 1.0) -> tuple:
        seg_logits = F.interpolate(outputs["seg_logits"], targets.shape[-2:],
                                   mode="bilinear", align_corners=False)
        bd_logits = F.interpolate(outputs["boundary_logits"], targets.shape[-2:],
                                  mode="bilinear", align_corners=False)

        # CE weight: apply MAC class-conditional multipliers if provided
        ce_w = self.ce_weight
        if mac_ce_multipliers is not None:
            crack_mult, spalling_mult = mac_ce_multipliers
            ce_w = self.ce_weight.clone()
            ce_w[1] = ce_w[1] * crack_mult
            ce_w[2] = ce_w[2] * spalling_mult

        # CE loss: per-sample weighted when sample_weights provided
        if sample_weights is not None:
            ce_unreduced = F.cross_entropy(seg_logits, targets, weight=ce_w,
                                           reduction='none')  # (B,H,W)
            ce_per_sample = ce_unreduced.mean(dim=(1, 2))  # (B,)
            loss_ce = (ce_per_sample * sample_weights).mean()
        else:
            loss_ce = F.cross_entropy(seg_logits, targets, weight=ce_w)

        loss_dice = fg_dice_loss(seg_logits, targets)

        # Tversky loss: only when enabled, with optional per-sample weighting
        if self.cfg.use_tversky_loss:
            if sample_weights is not None:
                loss_tversky = self._tversky_weighted(seg_logits, targets, sample_weights)
            else:
                loss_tversky = self.tversky(seg_logits, targets)
        else:
            loss_tversky = seg_logits.new_zeros(())

        # Boundary BCE loss: only when enabled (batch-level, not per-sample weighted)
        if self.cfg.use_boundary_loss:
            loss_bd = boundary_bce_loss(bd_logits, targets)
        else:
            loss_bd = seg_logits.new_zeros(())

        # Topology loss: SRL (preferred) or soft-clDice
        loss_topo = seg_logits.new_zeros(())
        topo_active = epoch >= self.cfg.cldice_start_epoch
        if self.cfg.use_srl_loss and topo_active:
            probs = seg_logits.softmax(dim=1)
            p_crack = probs[:, 1:2, :, :]                       # (B,1,H,W)
            if crack_skel is not None and crack_skel.any():
                loss_topo = skeleton_recall_loss(p_crack, crack_skel)
        elif self.cfg.use_cldice_loss and topo_active:
            probs = seg_logits.softmax(dim=1)
            p_crack = probs[:, 1:2, :, :]                       # (B,1,H,W)
            y_crack = (targets == 1).float().unsqueeze(1)        # (B,1,H,W)
            loss_topo = soft_cldice_loss(p_crack, y_crack,
                                         iters=self.cfg.cldice_iters)

        # Loss weights: scheduled vs constant
        if self.cfg.use_class_loss_schedule:
            lam_crack = scheduler.crack_weight(epoch)
            lam_bd = scheduler.boundary_weight(epoch)
        else:
            lam_crack = 1.0
            lam_bd = 1.0

        # Snake branch auxiliary loss
        loss_snake = seg_logits.new_zeros(())
        if self.cfg.use_snake_aux_loss and "crack_enhance" in outputs:
            loss_snake = snake_aux_loss(outputs["crack_enhance"], targets)

        total = (self.cfg.loss_ce_w * loss_ce
                 + self.cfg.loss_dice_w * loss_dice
                 + lam_crack * loss_tversky
                 + lam_bd * loss_bd
                 + self.cfg.cldice_weight * mac_topo_multiplier * loss_topo
                 + self.cfg.snake_aux_weight * loss_snake)

        # Per-sample signals for difficulty estimation (detach!)
        ps_ce = per_sample_ce(seg_logits, targets, self.ce_weight).detach()
        ps_ent = per_sample_entropy(seg_logits).detach()

        info = {
            "loss_ce": loss_ce.item(),
            "loss_dice": loss_dice.item(),
            "loss_tversky": loss_tversky.item(),
            "loss_bd": loss_bd.item(),
            "loss_cldice": loss_topo.item(),
            "loss_snake": loss_snake.item(),
            "per_sample_ce": ps_ce.cpu(),
            "per_sample_ent": ps_ent.cpu(),
        }

        # MAC: per-class CE for class-specific difficulty tracking
        if self.cfg.use_mac:
            info["per_sample_ce_crack"] = per_sample_class_ce(
                seg_logits, targets, 1).detach().cpu()
            info["per_sample_ce_spalling"] = per_sample_class_ce(
                seg_logits, targets, 2).detach().cpu()

        return total, info

    def _tversky_weighted(self, logits: torch.Tensor, targets: torch.Tensor,
                          sample_weights: torch.Tensor) -> torch.Tensor:
        """Tversky loss with per-sample weighting (multiply before mean)."""
        crack_prob = torch.softmax(logits, dim=1)[:, 1:2]
        crack_gt = (targets == 1).float().unsqueeze(1)
        tp = (crack_prob * crack_gt).sum(dim=(1, 2, 3))
        fp = (crack_prob * (1 - crack_gt)).sum(dim=(1, 2, 3))
        fn = ((1 - crack_prob) * crack_gt).sum(dim=(1, 2, 3))
        tversky = (tp + self.tversky.eps) / (
            tp + self.tversky.alpha * fp + self.tversky.beta * fn + self.tversky.eps)
        per_sample_loss = 1.0 - tversky  # (B,)
        return (per_sample_loss * sample_weights).mean()

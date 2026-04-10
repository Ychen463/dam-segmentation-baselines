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
                scheduler, epoch: int) -> tuple:
        seg_logits = F.interpolate(outputs["seg_logits"], targets.shape[-2:],
                                   mode="bilinear", align_corners=False)
        bd_logits = F.interpolate(outputs["boundary_logits"], targets.shape[-2:],
                                  mode="bilinear", align_corners=False)

        loss_ce = F.cross_entropy(seg_logits, targets, weight=self.ce_weight)
        loss_dice = fg_dice_loss(seg_logits, targets)
        loss_tversky = self.tversky(seg_logits, targets)
        loss_bd = boundary_bce_loss(bd_logits, targets)

        lam_crack = scheduler.crack_weight(epoch)
        lam_bd = scheduler.boundary_weight(epoch)

        total = (self.cfg.loss_ce_w * loss_ce
                 + self.cfg.loss_dice_w * loss_dice
                 + lam_crack * loss_tversky
                 + lam_bd * loss_bd)

        # Per-sample signals for difficulty estimation (detach!)
        ps_ce = per_sample_ce(seg_logits, targets, self.ce_weight).detach()
        ps_ent = per_sample_entropy(seg_logits).detach()

        return total, {
            "loss_ce": loss_ce.item(),
            "loss_dice": loss_dice.item(),
            "loss_tversky": loss_tversky.item(),
            "loss_bd": loss_bd.item(),
            "per_sample_ce": ps_ce.cpu(),
            "per_sample_ent": ps_ent.cpu(),
        }

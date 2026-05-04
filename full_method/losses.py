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
# Skeleton Prediction Loss + Consistency Loss (Dual-Task, G3 preset)
# ---------------------------------------------------------------------------

def skeleton_prediction_loss(skel_logits, skel_gt, eps=1e-6):
    """BCE + Dice loss for skeleton prediction head.

    Args:
        skel_logits: (B, 1, H, W) raw logits from skeleton head.
        skel_gt: (B, 1, H, W) binary GT skeleton.

    Returns:
        Scalar loss combining BCE and Dice for skeleton prediction.
    """
    # Resize if needed
    if skel_logits.shape[-2:] != skel_gt.shape[-2:]:
        skel_logits = F.interpolate(skel_logits, size=skel_gt.shape[-2:],
                                    mode="bilinear", align_corners=False)
    # BCE
    bce = F.binary_cross_entropy_with_logits(skel_logits, skel_gt, reduction='mean')
    # Dice
    p = torch.sigmoid(skel_logits)
    inter = (p * skel_gt).sum()
    card = p.sum() + skel_gt.sum()
    dice = 1.0 - (2.0 * inter + eps) / (card + eps)
    return 0.5 * bce + 0.5 * dice


def skeleton_consistency_loss(seg_logits, skel_logits, targets, eps=1e-6):
    """Topology consistency loss: predicted skeleton should match skeleton of
    predicted segmentation.

    The idea: if the model predicts a crack region, the skeleton of that region
    should match what the skeleton head predicts. This enforces structural
    coherence between the two tasks.

    Implementation: soft Dice between (skeleton_head output) and
    (soft morphological skeleton of predicted crack probability).

    Args:
        seg_logits: (B, C, H, W) segmentation logits.
        skel_logits: (B, 1, H, W) skeleton prediction logits.
        targets: (B, H, W) ground truth masks (for size reference).

    Returns:
        Scalar consistency loss.
    """
    H, W = targets.shape[-2:]
    # Get predicted crack probability
    seg_up = F.interpolate(seg_logits, size=(H, W), mode="bilinear", align_corners=False)
    p_crack = seg_up.softmax(dim=1)[:, 1:2]  # (B, 1, H, W)

    # Compute soft skeleton of predicted crack (lightweight: 5 iterations)
    pred_skel = soft_skel(p_crack, iters=5)  # (B, 1, H, W)

    # Get skeleton head prediction
    skel_pred = torch.sigmoid(F.interpolate(skel_logits, size=(H, W),
                                            mode="bilinear", align_corners=False))

    # Soft Dice between the two skeleton representations
    inter = (pred_skel * skel_pred).sum(dim=(1, 2, 3))
    card = pred_skel.sum(dim=(1, 2, 3)) + skel_pred.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (card + eps)

    # Only compute for samples that have crack pixels
    has_crack = (p_crack.sum(dim=(1, 2, 3)) > 1.0)
    if has_crack.any():
        return 1.0 - dice[has_crack].mean()
    return seg_logits.new_zeros(())


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
# Lovász-Softmax loss (Berman et al., CVPR 2018)
# ---------------------------------------------------------------------------

def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def ohem_cross_entropy(logits, targets, weight, ratio=0.25):
    """Online Hard Example Mining CE: keep only top-k% hardest pixels."""
    ce = F.cross_entropy(logits, targets, weight=weight, reduction='none')  # (B,H,W)
    B = ce.shape[0]
    ce_flat = ce.reshape(B, -1)
    k = max(1, int(ce_flat.shape[1] * ratio))
    topk, _ = ce_flat.topk(k, dim=1)
    return topk.mean()


def lovasz_softmax(probas, labels, classes=(1, 2)):
    """Lovász-Softmax loss for specified foreground classes.

    Args:
        probas: (B, C, H, W) class probabilities (after softmax).
        labels: (B, H, W) integer ground truth.
        classes: tuple of class indices to compute loss for.
    """
    losses = []
    for c in classes:
        fg = (labels == c).float()  # (B, H, W)
        fg_flat = fg.reshape(-1)
        errors = (fg_flat - probas[:, c].reshape(-1)).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg_flat[perm]
        grad = _lovasz_grad(fg_sorted)
        loss_c = torch.dot(F.relu(errors_sorted), grad)
        losses.append(loss_c)
    return sum(losses) / len(losses) if losses else probas.new_zeros(())


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

        # CE loss: OHEM, per-sample weighted, or standard
        if self.cfg.use_ohem:
            loss_ce = ohem_cross_entropy(seg_logits, targets, ce_w,
                                         ratio=self.cfg.ohem_ratio)
        elif sample_weights is not None:
            ce_unreduced = F.cross_entropy(seg_logits, targets, weight=ce_w,
                                           reduction='none')  # (B,H,W)
            ce_per_sample = ce_unreduced.mean(dim=(1, 2))  # (B,)
            loss_ce = (ce_per_sample * sample_weights).mean()
        else:
            loss_ce = F.cross_entropy(seg_logits, targets, weight=ce_w)

        # Dice or Lovász loss
        if self.cfg.use_lovasz_loss:
            probs_lv = seg_logits.softmax(dim=1)
            loss_dice = lovasz_softmax(probs_lv, targets, classes=(1, 2))
        else:
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

        # Dual-task skeleton losses (G3 preset)
        loss_skel_pred = seg_logits.new_zeros(())
        loss_skel_consist = seg_logits.new_zeros(())
        if self.cfg.use_skeleton_head and "skel_logits" in outputs:
            skel_logits = outputs["skel_logits"]
            # Skeleton prediction loss (supervised by GT skeleton)
            if crack_skel is not None and crack_skel.any():
                loss_skel_pred = skeleton_prediction_loss(skel_logits, crack_skel)
            # Skeleton consistency loss (active after skel_consist_start_epoch)
            if epoch >= self.cfg.skel_consist_start_epoch:
                loss_skel_consist = skeleton_consistency_loss(
                    seg_logits, skel_logits, targets)

        dice_w = self.cfg.lovasz_weight if self.cfg.use_lovasz_loss else self.cfg.loss_dice_w
        total = (self.cfg.loss_ce_w * loss_ce
                 + dice_w * loss_dice
                 + lam_crack * loss_tversky
                 + lam_bd * loss_bd
                 + self.cfg.cldice_weight * mac_topo_multiplier * loss_topo
                 + self.cfg.snake_aux_weight * loss_snake
                 + self.cfg.skel_pred_weight * loss_skel_pred
                 + self.cfg.skel_consist_weight * loss_skel_consist)

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
            "loss_skel_pred": loss_skel_pred.item(),
            "loss_skel_consist": loss_skel_consist.item(),
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

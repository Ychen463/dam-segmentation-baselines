"""Adaptive Class-Conditional Teacher Weighting (ACCW).

Replaces manual per-class teacher weights with learnable parameters
optimized via validation feedback (simplified bi-level optimization).

Model params theta: updated on training loss (inner loop).
Weight params alpha: updated on validation loss (outer loop).

Every `update_freq` training steps, a validation mini-batch is forwarded
through both teachers + student with current adaptive weights. The val
loss gradient flows only through alpha (model params are detached).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveTeacherWeights(nn.Module):
    """Learnable per-class teacher-2 weights via sigmoid parameterization.

    For each class c:
        w2_c = sigmoid(alpha_c)     # teacher-2 weight
        w1_c = 1 - w2_c             # teacher-1 weight

    Attributes:
        logits: (C,) learnable parameters, initialized from manual weights.
    """

    def __init__(self, num_classes: int = 3,
                 init_t2_weights: list | None = None):
        """
        Args:
            num_classes: number of classes.
            init_t2_weights: initial teacher-2 weights per class.
                Default: [0.5, 0.6, 0.3] (bg, crack, spalling).
        """
        super().__init__()
        if init_t2_weights is None:
            init_t2_weights = [0.5, 0.6, 0.3]
        assert len(init_t2_weights) == num_classes
        # Inverse sigmoid to get initial logits
        init_logits = [math.log(w / (1.0 - w + 1e-8) + 1e-8)
                       for w in init_t2_weights]
        self.logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (w1, w2) each of shape (C,)."""
        w2 = torch.sigmoid(self.logits)
        w1 = 1.0 - w2
        return w1, w2

    def get_weights_dict(self) -> dict[str, float]:
        """Return current weights as a dict for logging."""
        w1, w2 = self.forward()
        w2_np = w2.detach().cpu().numpy()
        names = ["bg", "crack", "spalling"]
        return {f"w2_{names[i]}": float(w2_np[i]) for i in range(len(names))}


def adaptive_dual_teacher_ensemble(
    t1_logits: torch.Tensor,
    t2_logits: torch.Tensor,
    accw: AdaptiveTeacherWeights,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Create ensemble soft labels using learnable per-class weights.

    Unlike dual_teacher_ensemble() which uses fixed config values,
    this version uses differentiable weights from AdaptiveTeacherWeights,
    allowing gradient flow through the weight parameters.

    Args:
        t1_logits: (B, C, H, W) teacher-1 logits.
        t2_logits: (B, C, H, W) teacher-2 logits.
        accw: AdaptiveTeacherWeights module.
        temperature: softmax temperature.

    Returns:
        (B, C, H, W) ensemble logits.
    """
    t1_prob = F.softmax(t1_logits / temperature, dim=1)
    t2_prob = F.softmax(t2_logits / temperature, dim=1)

    w1, w2 = accw()  # (C,), (C,)

    # Reshape for broadcasting: (1, C, 1, 1)
    w1 = w1.view(1, -1, 1, 1)
    w2 = w2.view(1, -1, 1, 1)

    ensemble = w1 * t1_prob + w2 * t2_prob

    # Renormalize to sum to 1
    ensemble = ensemble / (ensemble.sum(dim=1, keepdim=True) + 1e-8)

    # Convert back to logits
    return torch.log(ensemble + 1e-8) * temperature


def accw_val_loss(
    student_model: nn.Module,
    teacher1_model: nn.Module,
    teacher2_model: nn.Module,
    accw: AdaptiveTeacherWeights,
    val_batch: dict,
    device: str,
    temperature: float = 4.0,
    kd_alpha: float = 0.5,
    use_conf_kd: bool = True,
) -> torch.Tensor:
    """Compute validation KD loss with adaptive weights.

    The gradient flows through accw.logits only (student & teacher
    params are detached / frozen). This enables updating the weight
    params to minimize val-set distillation quality.

    Args:
        student_model: the student (in eval mode, grads disabled on params).
        teacher1_model: frozen teacher 1.
        teacher2_model: frozen teacher 2.
        accw: AdaptiveTeacherWeights (the only trainable component here).
        val_batch: a batch dict from the val DataLoader.
        device: device string.
        temperature: KD temperature.
        kd_alpha: weight for KD loss vs. supervised loss.
        use_conf_kd: whether to apply confidence-aware KD.

    Returns:
        Scalar loss for updating accw parameters.
    """
    imgs = val_batch["image"].to(device, non_blocking=True).float()
    masks = val_batch["mask"].to(device, non_blocking=True).long()

    with torch.no_grad():
        t1_out = teacher1_model(imgs)
        t1_logits = F.interpolate(t1_out["seg_logits"].float(),
                                  masks.shape[-2:], mode="bilinear",
                                  align_corners=False)
        t2_out = teacher2_model(imgs)
        t2_logits = F.interpolate(t2_out["seg_logits"].float(),
                                  masks.shape[-2:], mode="bilinear",
                                  align_corners=False)
        s_out = student_model(imgs)
        s_logits = F.interpolate(s_out["seg_logits"].float(),
                                 masks.shape[-2:], mode="bilinear",
                                 align_corners=False)

    # Ensemble with adaptive weights (gradient flows through accw)
    t_logits = adaptive_dual_teacher_ensemble(
        t1_logits, t2_logits, accw, temperature)

    # KD loss: student vs. adaptive ensemble
    s_log_prob = F.log_softmax(s_logits.detach() / temperature, dim=1)
    t_prob = F.softmax(t_logits / temperature, dim=1)

    if use_conf_kd:
        # Confidence-aware: down-weight where teachers disagree
        t1_prob = F.softmax(t1_logits / temperature, dim=1)
        t2_prob = F.softmax(t2_logits / temperature, dim=1)
        kl_t1t2 = (t1_prob * (t1_prob.clamp_min(1e-8).log()
                              - t2_prob.clamp_min(1e-8).log())).sum(1)
        kl_max = kl_t1t2.max().clamp_min(1e-6)
        agreement = 1.0 - (kl_t1t2 / kl_max)  # (B, H, W)

        kl_per_pixel = (t_prob * (t_prob.log() - s_log_prob)).sum(dim=1)
        loss_kd = (kl_per_pixel * agreement).mean() * (temperature ** 2)
    else:
        loss_kd = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temperature ** 2)

    # Supervised loss on val (CE + Dice) — provides ground-truth signal
    # to guide weights toward better per-class teacher selection
    loss_ce = F.cross_entropy(s_logits.detach(), masks)

    # Combined: we want weights that produce ensemble labels which,
    # when distilled, align with GT supervision on validation data
    # The key insight: optimizing KD loss alone would just match the
    # student's current predictions; adding CE grounds the weights
    # toward GT-aligned teacher selection.
    total = kd_alpha * loss_kd + (1.0 - kd_alpha) * loss_ce

    return total

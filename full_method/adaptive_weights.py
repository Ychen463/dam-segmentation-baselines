"""Adaptive Class-Conditional Teacher Weighting (ACCW).

Replaces manual per-class teacher weights with learnable parameters
optimized via validation feedback (simplified bi-level optimization).

Model params theta: updated on training loss (inner loop).
Weight params alpha: updated on validation loss (outer loop).

At the end of each epoch, the full validation set is forwarded through
both teachers + student. The val loss gradient flows only through alpha
(model params are detached), providing a stable signal to learn
per-class teacher expertise.
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


def accw_epoch_update(
    student_model: nn.Module,
    teacher1_model: nn.Module,
    teacher2_model: nn.Module,
    accw: AdaptiveTeacherWeights,
    accw_optimizer: torch.optim.Optimizer,
    val_loader,
    device: str,
    temperature: float = 4.0,
    kd_alpha: float = 0.5,
    use_conf_kd: bool = True,
) -> dict:
    """Update adaptive weights using the full validation set (once per epoch).

    Iterates over all val batches, accumulates gradient on accw.logits,
    then takes a single optimizer step. This gives a stable gradient
    signal from all 150 val images rather than a noisy 4-image estimate.

    Args:
        student_model: the student (eval mode, params frozen for this step).
        teacher1_model: frozen teacher 1.
        teacher2_model: frozen teacher 2.
        accw: AdaptiveTeacherWeights (the only trainable component).
        accw_optimizer: optimizer for accw parameters.
        val_loader: full validation DataLoader.
        device: device string.
        temperature: KD temperature.
        kd_alpha: weight for KD loss vs. supervised loss.
        use_conf_kd: whether to apply confidence-aware KD.

    Returns:
        Dict with val loss and current weights for logging.
    """
    student_model.eval()
    accw_optimizer.zero_grad()

    total_loss = torch.tensor(0.0, device=device)
    n_batches = 0

    for val_batch in val_loader:
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
            t1_prob = F.softmax(t1_logits / temperature, dim=1)
            t2_prob = F.softmax(t2_logits / temperature, dim=1)
            kl_t1t2 = (t1_prob * (t1_prob.clamp_min(1e-8).log()
                                  - t2_prob.clamp_min(1e-8).log())).sum(1)
            kl_max = kl_t1t2.max().clamp_min(1e-6)
            agreement = 1.0 - (kl_t1t2 / kl_max)

            kl_per_pixel = (t_prob * (t_prob.log() - s_log_prob)).sum(dim=1)
            loss_kd = (kl_per_pixel * agreement).mean() * (temperature ** 2)
        else:
            loss_kd = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temperature ** 2)

        loss_ce = F.cross_entropy(s_logits.detach(), masks)
        batch_loss = kd_alpha * loss_kd + (1.0 - kd_alpha) * loss_ce

        # Accumulate gradients across all val batches
        (batch_loss / len(val_loader)).backward()
        total_loss = total_loss + batch_loss.detach()
        n_batches += 1

    # Single optimizer step with accumulated gradients from full val set
    accw_optimizer.step()

    avg_loss = total_loss.item() / max(n_batches, 1)
    result = {"accw_val_loss": avg_loss}
    result.update(accw.get_weights_dict())
    return result

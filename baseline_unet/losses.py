"""CE + multiclass Dice loss. Foreground-only Dice with absent-class skip."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight: torch.Tensor, ce_w: float = C.LOSS_CE_W,
                 dice_w: float = C.LOSS_DICE_W, num_classes: int = C.NUM_CLASSES,
                 eps: float = 1e-6):
        super().__init__()
        self.register_buffer("ce_weight", ce_weight.float())
        self.ce_w = ce_w
        self.dice_w = dice_w
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.ce_weight)

        probs = logits.softmax(dim=1)
        onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        card = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * inter + self.eps) / (card + self.eps)  # [C]

        fg_dice = dice[1:]
        fg_present = (onehot[:, 1:].sum(dims) > 0)
        if fg_present.any():
            dice_loss = 1.0 - fg_dice[fg_present].mean()
        else:
            dice_loss = logits.new_zeros(())

        return self.ce_w * ce + self.dice_w * dice_loss

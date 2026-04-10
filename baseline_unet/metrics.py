"""Confusion-matrix based segmentation metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from . import config as C


class SegMetrics:
    def __init__(self, num_classes: int = C.NUM_CLASSES):
        self.n = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.cm[:] = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        p = pred.detach().cpu().numpy().ravel().astype(np.int64)
        t = target.detach().cpu().numpy().ravel().astype(np.int64)
        mask = (t >= 0) & (t < self.n)
        p = p[mask]; t = t[mask]
        idx = t * self.n + p
        binc = np.bincount(idx, minlength=self.n * self.n)
        self.cm += binc.reshape(self.n, self.n)

    def compute(self) -> Dict[str, float]:
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        iou = np.where(denom_iou > 0, tp / np.maximum(denom_iou, 1e-9), float("nan"))
        dice = np.where(denom_dice > 0, 2 * tp / np.maximum(denom_dice, 1e-9), float("nan"))
        pix_acc = tp.sum() / max(cm.sum(), 1.0)

        names = C.CLASS_NAMES
        out: Dict[str, float] = {}
        for i, name in enumerate(names):
            out[f"IoU_{name}"] = float(iou[i]) if not np.isnan(iou[i]) else 0.0
            out[f"Dice_{name}"] = float(dice[i]) if not np.isnan(dice[i]) else 0.0

        fg_iou = iou[1:]
        fg_iou_valid = fg_iou[~np.isnan(fg_iou)]
        out["mIoU_fg"] = float(fg_iou_valid.mean()) if fg_iou_valid.size else 0.0
        all_iou_valid = iou[~np.isnan(iou)]
        out["mIoU_all"] = float(all_iou_valid.mean()) if all_iou_valid.size else 0.0
        out["pixel_acc"] = float(pix_acc)
        return out


def format_metrics(m: Dict[str, float]) -> str:
    lines = []
    lines.append(
        "IoU   bg={IoU_background:.4f}  crack={IoU_crack:.4f}  spalling={IoU_spalling:.4f}"
        "  | mIoU_fg={mIoU_fg:.4f}  pix_acc={pixel_acc:.4f}".format(**m)
    )
    lines.append(
        "Dice  bg={Dice_background:.4f}  crack={Dice_crack:.4f}  spalling={Dice_spalling:.4f}".format(**m)
    )
    return "\n".join(lines)

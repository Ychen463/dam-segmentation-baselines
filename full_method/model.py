"""SegFormerWithBoundary: SegFormer-B2 + crack boundary head via forward hook.

The boundary head branches from the fused decoder features (linear_fuse output)
and predicts a per-pixel crack boundary map.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class SegFormerWithBoundary(nn.Module):
    """SegFormer with an auxiliary crack boundary prediction head."""

    def __init__(self, pretrained: str, num_classes: int = C.NUM_CLASSES):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.hf_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            id2label={0: "background", 1: "crack", 2: "spalling"},
            label2id={"background": 0, "crack": 1, "spalling": 2},
        )

        # Auto-infer decoder feature dimension from HF config
        feat_dim = self.hf_model.config.decoder_hidden_size  # B2 = 768

        self.boundary_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, 1, 1),
        )

        self._fuse_feat = None
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self._fuse_feat = output
        self.hf_model.decode_head.linear_fuse.register_forward_hook(hook_fn)

    def forward(self, pixel_values: torch.Tensor) -> dict:
        self._fuse_feat = None  # Clear stale feature from previous batch

        outputs = self.hf_model(pixel_values)
        seg_logits = outputs.logits  # (B, C, H/4, W/4)

        assert self._fuse_feat is not None, "linear_fuse hook did not fire"
        boundary_logits = self.boundary_head(self._fuse_feat)  # (B, 1, H/4, W/4)

        return {
            "seg_logits": seg_logits,
            "boundary_logits": boundary_logits,
        }


class _PreviewWrapper(nn.Module):
    """Wraps SegFormerWithBoundary so save_preview() gets a raw logits tensor."""

    def __init__(self, model: SegFormerWithBoundary):
        super().__init__()
        self.m = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.m(x)
        logits = out["seg_logits"]
        return F.interpolate(logits, size=x.shape[-2:],
                             mode="bilinear", align_corners=False)

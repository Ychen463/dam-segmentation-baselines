"""DINOv2-LoRA: DINOv2 ViT-B/14 with LoRA adapters for crack/spalling segmentation.

Frozen DINOv2 encoder + LoRA (rank=16) into attention Q/K/V + MLP layers.
Lightweight FPN decoder for 3-class dense prediction.
Only ~5% parameters are trainable.

DINOv2 provides stronger dense prediction features than SAM, as it was
trained with self-supervised objectives on diverse visual data.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .sam_model import LoRALinear, FPNDecoder


# ---------------------------------------------------------------------------
# LoRA injection for DINOv2
# ---------------------------------------------------------------------------

def inject_lora_dinov2(model: nn.Module, rank: int = 16, alpha: float = 16.0):
    """Inject LoRA into DINOv2 ViT attention and MLP layers.

    Targets: blocks[i].attn.qkv, blocks[i].attn.proj,
             blocks[i].mlp.fc1, blocks[i].mlp.fc2
    """
    target_names = ("qkv", "proj", "fc1", "fc2")
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in target_names:
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
                replaced += 1
    return replaced


# ---------------------------------------------------------------------------
# DINOv2-LoRA Model
# ---------------------------------------------------------------------------

class DINOv2LoRA(nn.Module):
    """DINOv2 ViT-B/14 encoder (frozen + LoRA) with FPN decoder.

    Architecture:
        1. DINOv2 ViT-B/14 (frozen, loaded from torch.hub)
        2. LoRA adapters injected into attention Q/K/V + MLP layers
        3. FPN decoder on features from blocks [2, 5, 8, 11]
        4. Output: 3-class segmentation logits

    Only LoRA params + decoder are trainable (~5% of total).
    """

    def __init__(self, num_classes: int = C.NUM_CLASSES,
                 lora_rank: int = 16, lora_alpha: float = 16.0,
                 fpn_dim: int = 256, img_size: int = 518,
                 feature_indices: Tuple[int, ...] = (2, 5, 8, 11)):
        super().__init__()
        self.img_size = img_size
        self.feature_indices = feature_indices

        # Load DINOv2 ViT-B/14
        self.encoder = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True
        )
        self.patch_size = self.encoder.patch_size  # 14
        self.embed_dim = self.encoder.embed_dim    # 768

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Inject LoRA
        n_lora = inject_lora_dinov2(self.encoder, rank=lora_rank, alpha=lora_alpha)
        print(f"[DINOv2-LoRA] injected LoRA into {n_lora} layers (rank={lora_rank})")

        # FPN decoder
        self.decoder = FPNDecoder(
            embed_dim=self.embed_dim, num_classes=num_classes, fpn_dim=fpn_dim
        )

        # Print param stats
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DINOv2-LoRA] total params: {total/1e6:.1f}M, "
              f"trainable: {trainable/1e6:.1f}M ({100*trainable/total:.1f}%)")

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run DINOv2 encoder and extract intermediate block features.

        DINOv2 ViT-B/14 has 12 transformer blocks. We tap features at
        [2, 5, 8, 11] for multi-depth representation.
        """
        B = x.shape[0]

        # Patch embed: (B, 3, H, W) → (B, N, C) with CLS token prepended
        x = self.encoder.prepare_tokens_with_masks(x)

        # Compute spatial dims from number of patch tokens
        n_tokens = x.shape[1] - 1  # exclude CLS token
        h = w = int(n_tokens ** 0.5)

        features = []
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in self.feature_indices:
                # Remove CLS token and reshape to spatial
                tokens = x[:, 1:]  # (B, N, C)
                feat = tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)  # (B, C, h, w)
                features.append(feat)

        return features

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Args:
            pixel_values: (B, 3, H, W) input images
        Returns:
            dict with "seg_logits": (B, num_classes, H, W)
        """
        B, _, H_in, W_in = pixel_values.shape

        # Resize to DINOv2-friendly size (multiple of patch_size=14)
        if H_in != self.img_size or W_in != self.img_size:
            x = F.interpolate(pixel_values, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        else:
            x = pixel_values

        # Extract multi-depth features
        features = self._extract_features(x)

        # Decode
        seg_logits = self.decoder(features)  # (B, C, feat_h, feat_w)

        # Upsample to original input resolution
        seg_logits = F.interpolate(seg_logits, size=(H_in, W_in),
                                   mode="bilinear", align_corners=False)

        # Dummy boundary logits for CompositeLoss compatibility
        boundary_logits = torch.zeros(B, 1, H_in, W_in,
                                      device=seg_logits.device, dtype=seg_logits.dtype)

        return {"seg_logits": seg_logits, "boundary_logits": boundary_logits}

    def trainable_parameters(self):
        """Return only trainable parameters (LoRA + decoder)."""
        return [p for p in self.parameters() if p.requires_grad]

    def lora_parameters(self):
        """Return LoRA parameters (for separate LR group)."""
        params = []
        for name, p in self.named_parameters():
            if p.requires_grad and "lora_" in name:
                params.append(p)
        return params

    def decoder_parameters(self):
        """Return decoder parameters (for separate LR group)."""
        return list(self.decoder.parameters())

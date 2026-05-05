"""TopoLoRA-SAM: SAM ViT-B with LoRA adapters for crack/spalling segmentation.

Frozen SAM encoder + LoRA (rank=16) into attention Q/K/V projections.
Lightweight FPN decoder for 3-class dense prediction.
Only ~5% parameters are trainable.

Reference: TopoLoRA-SAM (arXiv 2601.02273)
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adaptation."""

    def __init__(self, orig: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.orig = orig
        self.rank = rank
        self.scale = alpha / rank

        # Freeze original weights
        orig.weight.requires_grad_(False)
        if orig.bias is not None:
            orig.bias.requires_grad_(False)

        # LoRA A/B matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, orig.in_features))
        self.lora_B = nn.Parameter(torch.zeros(orig.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B starts at zero → LoRA starts as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.orig(x)
        lora = (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return base + lora


def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 16.0,
                target_modules: Tuple[str, ...] = ("qkv", "proj", "lin1", "lin2")):
    """Inject LoRA into all matching Linear layers in a SAM ViT encoder."""
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(t in child_name for t in target_modules):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
                replaced += 1
    return replaced


# ---------------------------------------------------------------------------
# FPN Decoder
# ---------------------------------------------------------------------------

class FPNDecoder(nn.Module):
    """Lightweight Feature Pyramid Network decoder for multi-class segmentation.

    Takes multi-scale features from SAM ViT blocks and produces dense predictions.
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 3, fpn_dim: int = 256):
        super().__init__()
        # Lateral connections (project each scale to fpn_dim)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, fpn_dim, 1),
                nn.GroupNorm(32, fpn_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])

        # Top-down fusion convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
                nn.GroupNorm(32, fpn_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])

        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
            nn.GroupNorm(32, fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of 4 tensors, each (B, C, H_i, W_i) from different encoder depths
        Returns:
            (B, num_classes, H_0, W_0) segmentation logits at highest feature resolution
        """
        # Lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway (from deepest to shallowest)
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:],
                               mode="bilinear", align_corners=False)
            laterals[i] = laterals[i] + up

        # Apply FPN convolutions
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Use highest resolution feature for prediction
        # Upsample all to highest resolution and sum
        target_size = fpn_outs[0].shape[-2:]
        fused = fpn_outs[0]
        for f in fpn_outs[1:]:
            fused = fused + F.interpolate(f, size=target_size,
                                          mode="bilinear", align_corners=False)

        return self.seg_head(fused)


# ---------------------------------------------------------------------------
# TopoLoRA-SAM
# ---------------------------------------------------------------------------

class TopoLoRASAM(nn.Module):
    """SAM ViT-B encoder (frozen + LoRA) with FPN decoder for segmentation.

    Architecture:
        1. SAM ViT-B image encoder (frozen)
        2. LoRA adapters injected into attention Q/K/V + MLP layers
        3. FPN decoder on features from blocks [2, 5, 8, 11]
        4. Output: 3-class segmentation logits

    Only LoRA params + decoder are trainable (~5% of total).
    """

    def __init__(self, sam_checkpoint: str, num_classes: int = C.NUM_CLASSES,
                 lora_rank: int = 16, lora_alpha: float = 16.0,
                 fpn_dim: int = 256, sam_img_size: int = 1024,
                 feature_indices: Tuple[int, ...] = (2, 5, 8, 11)):
        super().__init__()
        self.sam_img_size = sam_img_size
        self.feature_indices = feature_indices

        # Load SAM ViT-B encoder
        self.encoder = self._load_sam_encoder(sam_checkpoint, sam_img_size)
        embed_dim = self.encoder.pos_embed.shape[-1] if hasattr(self.encoder, 'pos_embed') else 768

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Inject LoRA
        n_lora = inject_lora(self.encoder, rank=lora_rank, alpha=lora_alpha)
        print(f"[TopoLoRA-SAM] injected LoRA into {n_lora} layers (rank={lora_rank})")

        # FPN decoder
        self.decoder = FPNDecoder(embed_dim=embed_dim, num_classes=num_classes,
                                  fpn_dim=fpn_dim)

        # Print param stats
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[TopoLoRA-SAM] total params: {total/1e6:.1f}M, "
              f"trainable: {trainable/1e6:.1f}M ({100*trainable/total:.1f}%)")

    @staticmethod
    def _load_sam_encoder(checkpoint_path: str, img_size: int = 1024):
        """Load SAM ViT-B image encoder from checkpoint."""
        try:
            from segment_anything import sam_model_registry
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            encoder = sam.image_encoder
            print(f"[TopoLoRA-SAM] loaded SAM ViT-B from {checkpoint_path}")
            return encoder
        except ImportError:
            raise ImportError(
                "segment_anything not installed. "
                "Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run encoder and extract intermediate features from specified blocks.

        SAM ViT-B has 12 transformer blocks. We tap features at indices
        [2, 5, 8, 11] to get 4 scales of representation.

        Since SAM ViT doesn't have natural multi-scale (all features are same
        resolution), we use the features from different depths as "scales"
        for the FPN decoder.
        """
        # SAM encoder: patch_embed → blocks → neck
        # We need to hook into intermediate blocks

        # Patch embedding
        x = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed

        features = []
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in self.feature_indices:
                # x shape: (B, H, W, C) for SAM ViT
                features.append(x.permute(0, 3, 1, 2))  # → (B, C, H, W)

        return features

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Args:
            pixel_values: (B, 3, H, W) input images (any resolution)
        Returns:
            dict with "seg_logits": (B, num_classes, H, W)
        """
        B, _, H_in, W_in = pixel_values.shape

        # Resize to SAM's expected input size
        if H_in != self.sam_img_size or W_in != self.sam_img_size:
            x = F.interpolate(pixel_values, size=(self.sam_img_size, self.sam_img_size),
                              mode="bilinear", align_corners=False)
        else:
            x = pixel_values

        # Extract multi-depth features
        features = self._extract_features(x)

        # Decode
        seg_logits = self.decoder(features)  # (B, C, feat_H, feat_W)

        # Upsample to input resolution
        seg_logits = F.interpolate(seg_logits, size=(H_in, W_in),
                                   mode="bilinear", align_corners=False)

        # Dummy boundary logits for CompositeLoss compatibility
        boundary_logits = torch.zeros(B, 1, H_in, W_in,
                                      device=seg_logits.device, dtype=seg_logits.dtype)

        return {"seg_logits": seg_logits, "boundary_logits": boundary_logits}

    def trainable_parameters(self):
        """Return only trainable parameters (LoRA + decoder) for optimizer."""
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

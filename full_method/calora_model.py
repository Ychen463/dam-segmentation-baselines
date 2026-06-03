"""Crack-Aware Adaptive LoRA (CALoRA) for SAM ViT-B.

Key novelty: each attention layer has multiple LoRA experts at different ranks
(e.g., rank 8 for simple regions, rank 32 for complex cracks). A lightweight
crack-complexity router estimates per-image gating weights from patch embeddings,
dynamically allocating more adaptation capacity to images with thin, topologically
complex crack patterns.

This is fundamentally different from:
  - Standard LoRA: fixed rank, input-independent
  - AdaLoRA (Zhang et al. 2023): prunes rank during training but fixed at inference
  - Our CALoRA: input-dependent rank mixing at both train and inference time

Architecture:
    1. SAM ViT-B image encoder (frozen)
    2. MoE-LoRA adapters in attention Q/K/V + MLP layers
    3. CrackComplexityRouter on patch embeddings -> per-image expert gates
    4. FPN decoder for dense prediction
    5. Optional: auxiliary routing loss supervised by GT crack skeleton density
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .sam_model import FPNDecoder


# ---------------------------------------------------------------------------
# MoE-LoRA layer
# ---------------------------------------------------------------------------

class MoELoRALinear(nn.Module):
    """Linear layer with Mixture-of-Expert LoRA adaptation.

    Maintains multiple (A_i, B_i) pairs at different ranks. The final LoRA
    output is gate-weighted: sum_i gate_i * (x @ A_i^T) @ B_i^T * scale_i.
    Gates are set externally by the router before each forward pass.
    """

    def __init__(self, orig: nn.Linear, ranks: Tuple[int, ...] = (8, 32),
                 alpha: float = 16.0):
        super().__init__()
        self.orig = orig
        self.num_experts = len(ranks)
        self.ranks = ranks

        # Freeze original weights
        orig.weight.requires_grad_(False)
        if orig.bias is not None:
            orig.bias.requires_grad_(False)

        # Expert LoRA pairs
        self.lora_As = nn.ParameterList()
        self.lora_Bs = nn.ParameterList()
        self.scales = []
        for r in ranks:
            A = nn.Parameter(torch.zeros(r, orig.in_features))
            B = nn.Parameter(torch.zeros(orig.out_features, r))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            # B starts at zero -> LoRA starts as identity
            self.lora_As.append(A)
            self.lora_Bs.append(B)
            self.scales.append(alpha / r)

        # Gate set by router before forward pass
        self._gate = None  # (B, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.orig(x)
        gate = self._gate

        if gate is None:
            # Uniform fallback (e.g., during teacher loading)
            lora_out = sum(
                (x @ A.T) @ B.T * s
                for A, B, s in zip(self.lora_As, self.lora_Bs, self.scales)
            ) / self.num_experts
        else:
            lora_out = torch.zeros_like(base)
            for i, (A, B, s) in enumerate(zip(self.lora_As, self.lora_Bs, self.scales)):
                expert_out = (x @ A.T) @ B.T * s
                # gate[:, i] is (B,) -> expand to match x dims (B, H, W, D)
                g = gate[:, i]
                for _ in range(expert_out.dim() - 1):
                    g = g.unsqueeze(-1)
                lora_out = lora_out + g * expert_out

        return base + lora_out


# ---------------------------------------------------------------------------
# Crack-Complexity Router
# ---------------------------------------------------------------------------

class CrackComplexityRouter(nn.Module):
    """Estimates per-image crack complexity from patch embeddings.

    Produces soft gating weights for MoE-LoRA experts. Initialized to uniform
    gating so training starts from standard LoRA behavior.

    The router learns to allocate higher-rank experts to images with complex
    crack patterns (thin, branching, high junction density) via the main
    segmentation loss backpropagation.

    Optionally supervised by auxiliary loss: GT crack skeleton density
    correlates with desired high-rank gate activation.
    """

    def __init__(self, embed_dim: int = 768, num_experts: int = 2,
                 hidden_dim: int = 64):
        super().__init__()
        self.num_experts = num_experts
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )
        # Initialize to uniform gating (zero logits -> equal softmax)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, patch_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embed: (B, H, W, C) SAM patch embeddings
        Returns:
            gates: (B, num_experts) soft gating weights summing to 1
        """
        x = patch_embed.mean(dim=(1, 2))  # (B, C) global avg pool
        logits = self.mlp(x)              # (B, num_experts)
        return F.softmax(logits, dim=-1)

    def routing_entropy(self, gates: torch.Tensor) -> torch.Tensor:
        """Compute routing entropy for load-balancing regularization.

        Low entropy = one expert dominates (bad); high entropy = uniform (good).
        Returns negative entropy so minimizing this = maximizing entropy.
        """
        return (gates * gates.clamp(min=1e-8).log()).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Auxiliary routing loss
# ---------------------------------------------------------------------------

def compute_router_aux_loss(gates: torch.Tensor, masks: torch.Tensor,
                            crack_class: int = 1) -> torch.Tensor:
    """Auxiliary loss: encourage high-rank expert for crack-dense images.

    Computes crack pixel density per image, then pushes the last expert gate
    (highest rank) to correlate with it via MSE.

    Args:
        gates: (B, E) expert gating weights
        masks: (B, H, W) ground truth segmentation masks
        crack_class: label index for crack pixels
    Returns:
        scalar loss
    """
    B = masks.shape[0]
    total_pixels = masks.shape[1] * masks.shape[2]

    # Crack density per image: [0, 1]
    crack_density = (masks == crack_class).float().sum(dim=(1, 2)) / total_pixels

    # Target: higher crack density -> more weight on last expert (highest rank)
    # Normalize density to [0, 1] range for regression target
    target_high_rank = crack_density.clamp(0, 1).to(gates.device)

    # MSE between highest-rank gate and crack density
    high_rank_gate = gates[:, -1]  # last expert = highest rank
    return F.mse_loss(high_rank_gate, target_high_rank)


# ---------------------------------------------------------------------------
# MoE-LoRA injection
# ---------------------------------------------------------------------------

def inject_moe_lora(model: nn.Module, ranks: Tuple[int, ...] = (8, 32),
                    alpha: float = 16.0,
                    target_modules: Tuple[str, ...] = ("qkv", "proj",
                                                       "lin1", "lin2")):
    """Replace matching Linear layers with MoELoRALinear."""
    replaced = 0
    moe_layers: List[MoELoRALinear] = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(
                    t in child_name for t in target_modules):
                moe_layer = MoELoRALinear(child, ranks=ranks, alpha=alpha)
                setattr(module, child_name, moe_layer)
                moe_layers.append(moe_layer)
                replaced += 1
    return replaced, moe_layers


# ---------------------------------------------------------------------------
# CrackAdaptiveLoRASAM (main model)
# ---------------------------------------------------------------------------

class CrackAdaptiveLoRASAM(nn.Module):
    """SAM ViT-B with Crack-Aware Adaptive LoRA (CALoRA).

    Architecture:
        1. SAM ViT-B (frozen)
        2. MoE-LoRA in all attention layers (multiple rank experts)
        3. CrackComplexityRouter (patch embeddings -> expert gates)
        4. FPN decoder -> 3-class segmentation

    The router runs once per image, producing per-image gates that are
    broadcast to all MoE-LoRA layers. This means the entire encoder
    adapts its effective rank based on input crack complexity.
    """

    def __init__(self, sam_checkpoint: str, num_classes: int = C.NUM_CLASSES,
                 calora_ranks: Tuple[int, ...] = (8, 32),
                 lora_alpha: float = 16.0, fpn_dim: int = 256,
                 sam_img_size: int = 1024, router_hidden: int = 64,
                 feature_indices: Tuple[int, ...] = (2, 5, 8, 11)):
        super().__init__()
        self.sam_img_size = sam_img_size
        self.feature_indices = feature_indices
        self.calora_ranks = calora_ranks

        # Load SAM ViT-B encoder
        self.encoder = self._load_sam_encoder(sam_checkpoint, sam_img_size)
        embed_dim = (self.encoder.pos_embed.shape[-1]
                     if hasattr(self.encoder, 'pos_embed') else 768)

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Inject MoE-LoRA (replaces standard LoRA)
        n_moe, self._moe_layers = inject_moe_lora(
            self.encoder, ranks=calora_ranks, alpha=lora_alpha)
        print(f"[CALoRA-SAM] injected MoE-LoRA into {n_moe} layers "
              f"(ranks={calora_ranks})")

        # Crack complexity router
        self.router = CrackComplexityRouter(
            embed_dim=embed_dim,
            num_experts=len(calora_ranks),
            hidden_dim=router_hidden,
        )

        # FPN decoder (same as TopoLoRASAM)
        self.decoder = FPNDecoder(embed_dim=embed_dim, num_classes=num_classes,
                                  fpn_dim=fpn_dim)

        # Print param stats
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_router = sum(p.numel() for p in self.router.parameters())
        n_moe_params = sum(
            p.numel() for layer in self._moe_layers
            for p in list(layer.lora_As) + list(layer.lora_Bs)
        )
        print(f"[CALoRA-SAM] total: {total/1e6:.1f}M, "
              f"trainable: {trainable/1e6:.1f}M ({100*trainable/total:.1f}%)")
        print(f"[CALoRA-SAM] router: {n_router/1e3:.1f}K, "
              f"MoE-LoRA: {n_moe_params/1e6:.2f}M")

    @staticmethod
    def _load_sam_encoder(checkpoint_path: str, img_size: int = 1024):
        """Load SAM ViT-B image encoder from checkpoint."""
        try:
            from segment_anything import sam_model_registry
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            encoder = sam.image_encoder
            print(f"[CALoRA-SAM] loaded SAM ViT-B from {checkpoint_path}")
            return encoder
        except ImportError:
            raise ImportError(
                "segment_anything not installed. "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

    def _set_gates(self, gates: torch.Tensor):
        """Broadcast router gates to all MoE-LoRA layers."""
        for layer in self._moe_layers:
            layer._gate = gates

    def _clear_gates(self):
        """Clear gates after forward pass."""
        for layer in self._moe_layers:
            layer._gate = None

    def _extract_features(self, x: torch.Tensor,
                          gates: torch.Tensor) -> List[torch.Tensor]:
        """Run encoder with MoE-LoRA gating and extract intermediate features."""
        # Set gates on all MoE-LoRA layers
        self._set_gates(gates)

        # Patch embedding
        x = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed

        features = []
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in self.feature_indices:
                features.append(x.permute(0, 3, 1, 2))  # (B, C, H, W)

        self._clear_gates()
        return features

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Args:
            pixel_values: (B, 3, H, W)
        Returns:
            dict with "seg_logits", "boundary_logits", "router_gates"
        """
        B, _, H_in, W_in = pixel_values.shape

        # Resize to SAM input size
        if H_in != self.sam_img_size or W_in != self.sam_img_size:
            x = F.interpolate(pixel_values,
                              size=(self.sam_img_size, self.sam_img_size),
                              mode="bilinear", align_corners=False)
        else:
            x = pixel_values

        # Patch embedding (for router)
        patch_embed = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            patch_embed_with_pos = patch_embed + self.encoder.pos_embed
        else:
            patch_embed_with_pos = patch_embed

        # Router: estimate crack complexity -> expert gates
        gates = self.router(patch_embed_with_pos.detach())  # (B, num_experts)

        # Run encoder blocks with MoE-LoRA gating
        # (We re-use patch_embed_with_pos to avoid recomputing)
        self._set_gates(gates)
        feat_x = patch_embed_with_pos
        features = []
        for i, block in enumerate(self.encoder.blocks):
            feat_x = block(feat_x)
            if i in self.feature_indices:
                features.append(feat_x.permute(0, 3, 1, 2))
        self._clear_gates()

        # Decode
        seg_logits = self.decoder(features)
        seg_logits = F.interpolate(seg_logits, size=(H_in, W_in),
                                   mode="bilinear", align_corners=False)

        # Dummy boundary logits for CompositeLoss compatibility
        boundary_logits = torch.zeros(B, 1, H_in, W_in,
                                      device=seg_logits.device,
                                      dtype=seg_logits.dtype)

        return {
            "seg_logits": seg_logits,
            "boundary_logits": boundary_logits,
            "router_gates": gates,
        }

    def trainable_parameters(self):
        """All trainable parameters (MoE-LoRA + router + decoder)."""
        return [p for p in self.parameters() if p.requires_grad]

    def lora_parameters(self):
        """MoE-LoRA parameters (for separate LR group)."""
        params = []
        for name, p in self.named_parameters():
            if p.requires_grad and "lora_" in name:
                params.append(p)
        return params

    def router_parameters(self):
        """Router parameters."""
        return list(self.router.parameters())

    def decoder_parameters(self):
        """Decoder parameters."""
        return list(self.decoder.parameters())

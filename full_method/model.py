"""SegFormerWithBoundary & DSCformerDam models.

SegFormerWithBoundary: SegFormer-B2 + crack boundary head via forward hook.
DSCformerDam: SegFormer-B2 + Dynamic Snake Convolution crack branch (ICCV 2023).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class SegFormerWithBoundary(nn.Module):
    """SegFormer with auxiliary outputs (boundary head removed)."""

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

    def forward(self, pixel_values: torch.Tensor) -> dict:
        outputs = self.hf_model(pixel_values)
        seg_logits = outputs.logits  # (B, C, H/4, W/4)

        # Dummy boundary_logits for backward compatibility with loss/train code
        boundary_logits = torch.zeros(
            seg_logits.shape[0], 1, seg_logits.shape[2], seg_logits.shape[3],
            device=seg_logits.device, dtype=seg_logits.dtype)

        return {
            "seg_logits": seg_logits,
            "boundary_logits": boundary_logits,
        }


# ---------------------------------------------------------------------------
# Dynamic Snake Convolution (DSConv, ICCV 2023)
# ---------------------------------------------------------------------------

class DSC(nn.Module):
    """Deformable Sampling Core — learns cumulative offsets along one axis.

    morph=0: snake along x-axis (offsets in y)
    morph=1: snake along y-axis (offsets in x)
    """

    def __init__(self, in_ch: int, kernel_size: int = 9, morph: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.morph = morph
        # Predict 2*K-1 raw offsets (center + left/up + right/down)
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size - 1, 3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        K = self.kernel_size
        center = K // 2  # e.g. 4 for K=9

        # Raw offsets constrained to [-1, 1], then cumsum for continuity
        raw = self.offset_conv(x).tanh()  # (B, 2K-1, H, W)

        # Split into left/up (reversed) and right/down halves around center
        # Indices: 0..center-1 = left/up, center = center, center+1..2K-2 = right/down
        offset = torch.zeros(B, 2 * K - 1, H, W, device=x.device)
        # Right/down cumulative offsets
        for i in range(1, K):
            offset[:, center + i] = offset[:, center + i - 1] + raw[:, center + i]
        # Left/up cumulative offsets (go in reverse)
        for i in range(1, K):
            offset[:, center - i] = offset[:, center - i + 1] + raw[:, center - i]

        # Build sampling grid
        # Base grid: (H, W) normalized to [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )

        results = []
        for i in range(2 * K - 1):
            off = offset[:, i]  # (B, H, W) — pixel-space offset
            if self.morph == 0:
                # Snake along x: offset in y direction
                off_norm = off * (2.0 / max(H - 1, 1))
                g = torch.stack([grid_x.expand(B, -1, -1),
                                 grid_y.expand(B, -1, -1) + off_norm], dim=-1)
            else:
                # Snake along y: offset in x direction
                off_norm = off * (2.0 / max(W - 1, 1))
                g = torch.stack([grid_x.expand(B, -1, -1) + off_norm,
                                 grid_y.expand(B, -1, -1)], dim=-1)
            sampled = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros",
                                    align_corners=True)  # (B, C, H, W)
            results.append(sampled)

        # Stack and reshape for directional conv: (B, C*(2K-1), H, W)
        return torch.cat(results, dim=1)


class DSConv(nn.Module):
    """Dynamic Snake Convolution block: DSC deformation + directional conv.

    morph=0: horizontal snake → Conv2d(*, *, (1, K))
    morph=1: vertical snake   → Conv2d(*, *, (K, 1))
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9, morph: int = 0):
        super().__init__()
        self.dsc = DSC(in_ch, kernel_size, morph)
        expanded = in_ch * (2 * kernel_size - 1)
        if morph == 0:
            self.conv = nn.Conv2d(expanded, out_ch, (1, kernel_size),
                                  padding=(0, kernel_size // 2))
        else:
            self.conv = nn.Conv2d(expanded, out_ch, (kernel_size, 1),
                                  padding=(kernel_size // 2, 0))
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(self.dsc(x))))


class StandardConvBranch(nn.Module):
    """Standard Conv control for CrackSnakeBranch.

    Same projections + head structure, but replaces DSConv with standard
    directional convolutions (1xK and Kx1). Note: parameter count is
    ~1.93M vs CrackSnakeBranch's ~1.32M due to the dense Conv2d(64->1088, 3x3).
    """

    def __init__(self, s1_ch: int = 128, s2_ch: int = 320,
                 hidden: int = 64, kernel_size: int = 9):
        super().__init__()
        self.proj_s1 = nn.Conv2d(s1_ch, hidden, 1)
        self.proj_s2 = nn.Conv2d(s2_ch, hidden, 1)
        expanded = hidden * (2 * kernel_size - 1)  # match DSConv channel expansion
        self.expand = nn.Conv2d(hidden, expanded, 3, padding=1)
        self.conv_x = nn.Sequential(
            nn.Conv2d(expanded, hidden, (1, kernel_size), padding=(0, kernel_size // 2)),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_y = nn.Sequential(
            nn.Conv2d(expanded, hidden, (kernel_size, 1), padding=(kernel_size // 2, 0)),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        f1 = self.proj_s1(s1)
        f2 = F.interpolate(self.proj_s2(s2), size=s1.shape[-2:],
                           mode="bilinear", align_corners=False)
        fused = F.relu(self.expand(f1 + f2))
        out = self.conv_x(fused) + self.conv_y(fused)
        return self.head(out)


class CrackSnakeBranch(nn.Module):
    """Lightweight dual-DSConv branch for crack-specific geometric features.

    Consumes SegFormer encoder hidden_states[1] (128ch, H/8) and
    hidden_states[2] (320ch, H/16), produces a single-channel crack
    enhancement map at H/8 resolution.
    Output is zero-initialized so the branch starts as a no-op.
    """

    def __init__(self, s1_ch: int = 128, s2_ch: int = 320,
                 hidden: int = 64, kernel_size: int = 9):
        super().__init__()
        self.proj_s1 = nn.Conv2d(s1_ch, hidden, 1)
        self.proj_s2 = nn.Conv2d(s2_ch, hidden, 1)
        self.dsconv_x = DSConv(hidden, hidden, kernel_size, morph=0)
        self.dsconv_y = DSConv(hidden, hidden, kernel_size, morph=1)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 1, 1),
        )
        # Zero-init output so branch starts as no-op
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s1: (B, 128, H/8, W/8) — hidden_states[1]
            s2: (B, 320, H/16, W/16) — hidden_states[2]
        Returns:
            (B, 1, H/8, W/8) crack enhancement logits
        """
        f1 = self.proj_s1(s1)
        f2 = F.interpolate(self.proj_s2(s2), size=s1.shape[-2:],
                           mode="bilinear", align_corners=False)
        fused = f1 + f2  # (B, hidden, H/8, W/8)
        out = self.dsconv_x(fused) + self.dsconv_y(fused)
        return self.head(out)


class MultiScaleCrackSnakeBranch(nn.Module):
    """Multi-Scale Dynamic Snake Convolution branch for crack enhancement.

    Uses parallel DSConv paths at multiple kernel sizes (5, 9, 15) to capture
    cracks of varying widths — from hairline (1-3px) to wide structural (20+px).
    Each scale has bidirectional (x+y) snake convolutions. Outputs are fused
    via a learned 1x1 convolution, allowing the model to adaptively weight
    each scale per spatial location.

    This is the key architectural novelty: standard DSConv uses a single kernel
    size, which limits its receptive field to one characteristic crack width.
    Multi-scale DSConv provides scale-adaptive deformable sampling.
    """

    def __init__(self, s1_ch: int = 128, s2_ch: int = 320,
                 hidden: int = 64, kernel_sizes: tuple = (5, 9, 15)):
        super().__init__()
        self.proj_s1 = nn.Conv2d(s1_ch, hidden, 1)
        self.proj_s2 = nn.Conv2d(s2_ch, hidden, 1)

        # Multi-scale DSConv paths
        self.scales = nn.ModuleList()
        for ks in kernel_sizes:
            self.scales.append(nn.ModuleDict({
                'dsconv_x': DSConv(hidden, hidden, ks, morph=0),
                'dsconv_y': DSConv(hidden, hidden, ks, morph=1),
            }))

        # Fuse multi-scale outputs: N_scales * hidden → hidden
        n_scales = len(kernel_sizes)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * n_scales, hidden, 1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 1, 1),
        )
        # Zero-init output so branch starts as no-op
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s1: (B, 128, H/8, W/8) — hidden_states[1]
            s2: (B, 320, H/16, W/16) — hidden_states[2]
        Returns:
            (B, 1, H/8, W/8) crack enhancement logits
        """
        f1 = self.proj_s1(s1)
        f2 = F.interpolate(self.proj_s2(s2), size=s1.shape[-2:],
                           mode="bilinear", align_corners=False)
        fused = f1 + f2  # (B, hidden, H/8, W/8)

        # Parallel multi-scale DSConv
        scale_outs = []
        for scale in self.scales:
            out_s = scale['dsconv_x'](fused) + scale['dsconv_y'](fused)
            scale_outs.append(out_s)

        # Concatenate and fuse
        multi = torch.cat(scale_outs, dim=1)  # (B, hidden*N, H/8, W/8)
        fused_multi = self.fuse(multi)         # (B, hidden, H/8, W/8)

        return self.head(fused_multi)


class AttentionGate(nn.Module):
    """Learnable gate for DSConv fusion (Plan B).

    Instead of simple additive fusion (z_crack + E_crack), learns when to
    trust the DSConv branch based on local context:
        gate = sigmoid(W_g * [z_crack; E_crack])
        output = z_crack + gate * E_crack

    This allows the model to suppress DSConv in regions where it's not helpful
    (e.g., spalling boundaries) and amplify it along crack paths.
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()
        # Input: concatenation of [decoder_crack_logit, dsconv_enhancement]
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        # Initialize near-identity (gate ≈ 1) so it starts like simple addition
        nn.init.zeros_(self.gate_conv[-1].weight)
        nn.init.constant_(self.gate_conv[-1].bias, 2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, z_crack: torch.Tensor, e_crack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_crack: (B, 1, H, W) decoder crack logits
            e_crack: (B, 1, H, W) DSConv enhancement
        Returns:
            (B, 1, H, W) gated crack logits
        """
        gate_input = torch.cat([z_crack, e_crack], dim=1)  # (B, 2, H, W)
        gate = torch.sigmoid(self.gate_conv(gate_input))    # (B, 1, H, W)
        return z_crack + gate * e_crack


class SkeletonPredictionHead(nn.Module):
    """Lightweight head that predicts crack skeleton from decoder features.

    Used in the Dual-Task Skeleton Consistency framework (G3 preset):
    the model must simultaneously predict segmentation AND skeleton,
    with a consistency loss enforcing that the predicted skeleton matches
    the skeleton of the predicted segmentation.
    """

    def __init__(self, feat_dim: int = 768, hidden: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 1, 1),
        )

    def forward(self, fuse_feat: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1, H/4, W/4) skeleton logits."""
        return self.head(fuse_feat)


class DSCformerDam(nn.Module):
    """SegFormer-B2 + Dynamic Snake Convolution crack branch.

    The snake branch gradually learns to enhance the crack channel.

    When cfg.use_skeleton_head=True, adds a SkeletonPredictionHead for
    dual-task skeleton consistency learning.
    """

    def __init__(self, pretrained: str, num_classes: int = C.NUM_CLASSES,
                 cfg: C.RunCfg = None):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.hf_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            id2label={0: "background", 1: "crack", 2: "spalling"},
            label2id={"background": 0, "crack": 1, "spalling": 2},
        )

        feat_dim = self.hf_model.config.decoder_hidden_size  # B2 = 768

        # Snake branch for crack enhancement
        hidden = cfg.snake_channels if cfg is not None else 64
        ks = cfg.snake_kernel_size if cfg is not None else 9
        # SegFormer-B2 encoder dims: [64, 128, 320, 512]
        encoder_dims = self.hf_model.config.hidden_sizes  # [64, 128, 320, 512]

        # Multi-scale DSConv branch (E1 preset) or single-scale (default)
        self.use_multiscale_snake = cfg is not None and getattr(cfg, 'use_multiscale_snake', False)
        self.use_standard_conv_branch = cfg is not None and getattr(cfg, 'use_standard_conv_branch', False)
        if self.use_standard_conv_branch:
            self.snake_branch = StandardConvBranch(
                s1_ch=encoder_dims[1], s2_ch=encoder_dims[2],
                hidden=hidden, kernel_size=ks,
            )
        elif self.use_multiscale_snake:
            snake_kernel_sizes = getattr(cfg, 'snake_kernel_sizes', (5, 9, 15))
            self.snake_branch = MultiScaleCrackSnakeBranch(
                s1_ch=encoder_dims[1], s2_ch=encoder_dims[2],
                hidden=hidden, kernel_sizes=snake_kernel_sizes,
            )
        else:
            self.snake_branch = CrackSnakeBranch(
                s1_ch=encoder_dims[1], s2_ch=encoder_dims[2],
                hidden=hidden, kernel_size=ks,
            )

        # Attention-gated fusion (Plan B, G4 preset)
        self.use_attention_gate = cfg is not None and getattr(cfg, 'use_attention_gate', False)
        if self.use_attention_gate:
            self.attn_gate = AttentionGate(in_channels=2)

        # Skeleton prediction head (dual-task, G3/G4 preset)
        self.use_skeleton_head = cfg is not None and getattr(cfg, 'use_skeleton_head', False)
        if self.use_skeleton_head:
            self.skeleton_head = SkeletonPredictionHead(
                feat_dim=feat_dim,
                hidden=getattr(cfg, 'skeleton_head_hidden', 128),
            )

        # Contrastive learning projection (Plan C, G4 preset)
        self.use_contrastive = cfg is not None and getattr(cfg, 'use_contrastive', False)
        if self.use_contrastive:
            contrast_dim = getattr(cfg, 'contrastive_dim', 64)
            self.contrast_proj = nn.Sequential(
                nn.Conv2d(feat_dim, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, contrast_dim, 1),
            )

        self._fuse_feat = None
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self._fuse_feat = output
        self.hf_model.decode_head.linear_fuse.register_forward_hook(hook_fn)

    def forward(self, pixel_values: torch.Tensor) -> dict:
        self._fuse_feat = None

        outputs = self.hf_model(pixel_values, output_hidden_states=True)
        seg_logits = outputs.logits  # (B, C, H/4, W/4)

        assert self._fuse_feat is not None, "linear_fuse hook did not fire"

        # Dummy boundary_logits for backward compatibility with loss/train code
        boundary_logits = torch.zeros(
            seg_logits.shape[0], 1, seg_logits.shape[2], seg_logits.shape[3],
            device=seg_logits.device, dtype=seg_logits.dtype)

        # Extract encoder hidden states and reshape from (B, N, C) → (B, C, H, W)
        hidden_states = outputs.hidden_states  # tuple of (B, N_i, C_i)
        B = pixel_values.shape[0]
        H_in, W_in = pixel_values.shape[2], pixel_values.shape[3]
        # Stage resolutions: H/4, H/8, H/16, H/32
        s1 = hidden_states[1]  # (B, N, 128) at H/8
        s2 = hidden_states[2]  # (B, N, 320) at H/16
        H1, W1 = H_in // 8, W_in // 8
        H2, W2 = H_in // 16, W_in // 16
        s1 = s1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        s2 = s2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # Crack enhancement
        crack_enhance = self.snake_branch(s1, s2)  # (B, 1, H/8, W/8)
        crack_enhance = F.interpolate(crack_enhance, size=seg_logits.shape[-2:],
                                      mode="bilinear", align_corners=False)

        # Fusion: attention-gated (Plan B) or simple additive
        seg_logits = seg_logits.clone()
        if self.use_attention_gate:
            seg_logits[:, 1:2] = self.attn_gate(seg_logits[:, 1:2], crack_enhance)
        else:
            seg_logits[:, 1:2] = seg_logits[:, 1:2] + crack_enhance

        result = {
            "seg_logits": seg_logits,
            "boundary_logits": boundary_logits,
            "crack_enhance": crack_enhance,
        }

        # Skeleton prediction (dual-task, Plan A)
        if self.use_skeleton_head:
            result["skel_logits"] = self.skeleton_head(self._fuse_feat)

        # Contrastive features (Plan C)
        if self.use_contrastive:
            result["contrast_feat"] = self.contrast_proj(self._fuse_feat)

        return result


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

"""VMamba-Tiny/Small segmentation model.

Architecture:
  - VMamba encoder (hierarchical VSS blocks with 2D selective scan)
  - SegFormer-style ALL-MLP decode head

References:
  - Liu et al., "VMamba: Visual State Space Model", arXiv 2401.10166
  - Zhu et al., "Vision Mamba", arXiv 2401.09417
  - Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

Requires: mamba_ssm (CUDA, pip install mamba-ssm) for fast selective scan.
Falls back to pure PyTorch sequential scan if unavailable (slow but correct).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA selective scan kernel
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False


# ---------------------------------------------------------------------------
# Selective scan: PyTorch fallback
# ---------------------------------------------------------------------------

def selective_scan_ref(
    u: torch.Tensor,       # (B, D, L)
    delta: torch.Tensor,   # (B, D, L)
    A: torch.Tensor,       # (D, N)
    B: torch.Tensor,       # (B, N, L)
    C: torch.Tensor,       # (B, N, L)
    D_param: Optional[torch.Tensor] = None,  # (D,)
    delta_softplus: bool = False,
) -> torch.Tensor:
    """Pure PyTorch sequential selective scan (reference implementation)."""
    if delta_softplus:
        delta = F.softplus(delta)

    B_batch, dim, L = u.shape
    N = A.shape[1]

    u_f = u.float()
    delta_f = delta.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()

    h = torch.zeros(B_batch, dim, N, device=u.device, dtype=torch.float32)
    ys = []

    for t in range(L):
        dt = delta_f[:, :, t]                           # (B, D)
        dA = torch.exp(dt.unsqueeze(-1) * A_f)          # (B, D, N)
        dB = dt.unsqueeze(-1) * B_f[:, :, t].unsqueeze(1)  # (B, D, N)
        h = dA * h + dB * u_f[:, :, t].unsqueeze(-1)    # (B, D, N)
        y = (h * C_f[:, :, t].unsqueeze(1)).sum(-1)      # (B, D)
        ys.append(y)

    y = torch.stack(ys, dim=-1)  # (B, D, L)
    if D_param is not None:
        y = y + u_f * D_param.float().unsqueeze(0).unsqueeze(-1)
    return y.to(u.dtype)


def _selective_scan(u, delta, A, B, C, D_param=None, delta_softplus=False):
    """Dispatch to CUDA kernel or PyTorch fallback."""
    if HAS_MAMBA_CUDA:
        return selective_scan_fn(u, delta, A, B, C, D_param,
                                delta_softplus=delta_softplus)
    return selective_scan_ref(u, delta, A, B, C, D_param,
                             delta_softplus=delta_softplus)


# ---------------------------------------------------------------------------
# Cross-scan: 4-directional 2D scan
# ---------------------------------------------------------------------------

def cross_scan_forward(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, 4, C, L) with 4 scan directions."""
    B, C, H, W = x.shape
    L = H * W
    x0 = x.reshape(B, C, L)                                   # row-major
    x1 = x0.flip(-1)                                           # reverse row-major
    x2 = x.permute(0, 1, 3, 2).reshape(B, C, L)               # column-major
    x3 = x2.flip(-1)                                           # reverse column-major
    return torch.stack([x0, x1, x2, x3], dim=1)


def cross_merge(y: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(B, 4, C, L) -> (B, C, H, W) by summing 4 scan directions."""
    B, K, C, L = y.shape
    y0 = y[:, 0]
    y1 = y[:, 1].flip(-1)
    y2 = y[:, 2].reshape(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, L)
    y3 = y[:, 3].flip(-1).reshape(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, L)
    return (y0 + y1 + y2 + y3).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# SS2D: 2D Selective Scan module
# ---------------------------------------------------------------------------

K_DIRS = 4


class SS2D(nn.Module):
    """2D Selective Scan with cross-scan (4 directions)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3,
                 expand: int = 2, dt_rank: str = "auto", dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = int(d_model * expand)
        self.d_inner = d_inner
        dt_rank_val = math.ceil(d_inner / 16) if dt_rank == "auto" else int(dt_rank)
        self.dt_rank = dt_rank_val

        # Input projection → (x, z)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Depthwise conv on x (2D)
        self.conv2d = nn.Conv2d(d_inner, d_inner, d_conv,
                                padding=d_conv // 2, groups=d_inner, bias=True)
        self.act = nn.SiLU()

        # Per-direction SSM parameter projections
        # Projects x → (dt_rank, d_state_B, d_state_C) for each of K directions
        self.x_proj = nn.Parameter(
            torch.empty(K_DIRS, d_inner, dt_rank_val + d_state * 2))

        # Per-direction dt low-rank projection: dt_rank → d_inner
        self.dt_projs_weight = nn.Parameter(torch.empty(K_DIRS, dt_rank_val, d_inner))
        self.dt_projs_bias = nn.Parameter(torch.empty(K_DIRS, d_inner))

        # SSM parameters A, D (per-direction)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_logs = nn.Parameter(
            torch.log(A).unsqueeze(0).repeat(K_DIRS, 1, 1))  # (K, d_inner, N)
        self.Ds = nn.Parameter(torch.ones(K_DIRS, d_inner))

        # Output
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.x_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dt_projs_weight, a=math.sqrt(5))
        # dt bias: initialize so that softplus(bias) ≈ uniform(0.001, 0.1)
        dt_init = torch.exp(
            torch.rand(K_DIRS, self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        inv_softplus = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_projs_bias.data.copy_(inv_softplus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) -> (B, H, W, C)"""
        B, H, W, C = x.shape
        L = H * W

        # Project to x, z
        xz = self.in_proj(x)                          # (B, H, W, 2E)
        x_inner, z = xz.chunk(2, dim=-1)              # each (B, H, W, E)

        # DWConv + activation
        x_inner = x_inner.permute(0, 3, 1, 2).contiguous()  # (B, E, H, W)
        x_inner = self.act(self.conv2d(x_inner))

        # Cross-scan into 4 directional sequences
        xs = cross_scan_forward(x_inner)               # (B, 4, E, L)

        # Project to SSM params (dt, B, C) per direction
        # xs: (B, 4, E, L) → (4, B, L, E) for projection
        xs_t = xs.permute(1, 0, 3, 2).contiguous()    # (4, B, L, E)

        # x_proj: (4, E, dt_rank + 2N)
        dbc = torch.einsum('kble,ker->kblr', xs_t, self.x_proj)
        dts, Bs, Cs = dbc.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # dt low-rank → full: (4, B, L, dt_rank) → (4, B, L, E)
        dts = torch.einsum('kblr,kre->kble', dts, self.dt_projs_weight)
        dts = dts + self.dt_projs_bias[:, None, None, :]

        # Reshape for selective_scan: need (B, E, L) and (B, N, L)
        dts = dts.permute(0, 1, 3, 2).contiguous()    # (4, B, E, L)
        Bs = Bs.permute(0, 1, 3, 2).contiguous()      # (4, B, N, L)
        Cs = Cs.permute(0, 1, 3, 2).contiguous()      # (4, B, N, L)

        # A = -exp(A_log)
        As = -torch.exp(self.A_logs.float())           # (4, E, N)

        # Selective scan per direction
        ys = []
        for k in range(K_DIRS):
            y_k = _selective_scan(
                xs[:, k].contiguous(),                 # (B, E, L)
                dts[k],                                # (B, E, L)
                As[k],                                 # (E, N)
                Bs[k],                                 # (B, N, L)
                Cs[k],                                 # (B, N, L)
                self.Ds[k],                            # (E,)
                delta_softplus=True,
            )
            ys.append(y_k)

        y = torch.stack(ys, dim=1)                     # (B, 4, E, L)
        y = cross_merge(y, H, W)                       # (B, E, H, W)
        y = y.permute(0, 2, 3, 1).contiguous()         # (B, H, W, E)
        y = self.out_norm(y)

        # Gate with SiLU(z)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


# ---------------------------------------------------------------------------
# VSSBlock: LN → SS2D → residual → LN → MLP → residual
# ---------------------------------------------------------------------------

class VSSBlock(nn.Module):
    """VSS block: LN → SS2D → residual, optionally + LN → MLP → residual.

    Default mlp_ratio=0 matches VMamba-T (SS2D's internal gating acts as FFN).
    Set mlp_ratio>0 for VMamba-v2 style with explicit MLP.
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 3,
                 expand: int = 2, mlp_ratio: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state, d_conv=d_conv,
                         expand=expand, dropout=dropout)
        self.has_mlp = mlp_ratio > 0
        if self.has_mlp:
            self.norm2 = nn.LayerNorm(dim)
            mlp_hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, dim),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) -> (B, H, W, C)"""
        x = x + self.ss2d(self.norm1(x))
        if self.has_mlp:
            x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch embedding & merging
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    """Image to patch embedding: (B, 3, H, W) → (B, H/4, W/4, C)."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 96,
                 patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                              # (B, C, H/4, W/4)
        x = x.permute(0, 2, 3, 1).contiguous()        # (B, H/4, W/4, C)
        x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    """Downsample 2× by merging 2×2 patches: (B, H, W, C) → (B, H/2, W/2, 2C)."""
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"H={H}, W={W} must be even"
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)       # (B, H/2, W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)                          # (B, H/2, W/2, 2C)
        return x


# ---------------------------------------------------------------------------
# VMamba encoder
# ---------------------------------------------------------------------------

class VSSLayer(nn.Module):
    """Stack of VSSBlocks at a single resolution."""
    def __init__(self, dim: int, depth: int, d_state: int = 16,
                 d_conv: int = 3, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(dim, d_state=d_state, d_conv=d_conv,
                     expand=expand, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class VMambaEncoder(nn.Module):
    """Hierarchical VMamba encoder producing multi-scale features.

    Output feature maps at 1/4, 1/8, 1/16, 1/32 of input resolution,
    same as SegFormer/Swin Transformer.
    """
    def __init__(self, dims: List[int] = None, depths: List[int] = None,
                 d_state: int = 16, d_conv: int = 3, expand: int = 2,
                 in_channels: int = 3, dropout: float = 0.0):
        super().__init__()
        if dims is None:
            dims = [96, 192, 384, 768]
        if depths is None:
            depths = [2, 2, 9, 2]

        self.num_stages = len(dims)
        self.dims = dims

        # Patch embedding: 4× downsample
        self.patch_embed = PatchEmbed2D(in_channels, dims[0], patch_size=4)

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(self.num_stages):
            self.stages.append(VSSLayer(
                dims[i], depths[i], d_state=d_state,
                d_conv=d_conv, expand=expand, dropout=dropout))
            # Add downsample between stages (not after last)
            if i < self.num_stages - 1:
                self.downsamples.append(PatchMerging2D(dims[i]))
            else:
                self.downsamples.append(nn.Identity())

        # Final norms for each stage output
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: (B, 3, H, W) -> list of (B, C_i, H_i, W_i) for i in 0..3."""
        x = self.patch_embed(x)  # (B, H/4, W/4, C0)

        features = []
        for i in range(self.num_stages):
            x = self.stages[i](x)                     # (B, H_i, W_i, C_i)
            feat = self.norms[i](x)
            # Convert to BCHW for decoder
            features.append(feat.permute(0, 3, 1, 2).contiguous())
            if i < self.num_stages - 1:
                x = self.downsamples[i](x)             # (B, H/2, W/2, 2C)

        return features


# ---------------------------------------------------------------------------
# Decode head (SegFormer ALL-MLP style)
# ---------------------------------------------------------------------------

class MLPDecodeHead(nn.Module):
    """Multi-scale feature fusion decoder (SegFormer-style).

    Projects each scale to a common dim, upsamples to 1/4 resolution,
    concatenates, and projects to num_classes.
    """
    def __init__(self, in_dims: List[int], decode_dim: int = 256,
                 num_classes: int = 3):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, decode_dim, 1),
                nn.BatchNorm2d(decode_dim),
                nn.ReLU(inplace=True),
            )
            for d in in_dims
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(decode_dim * len(in_dims), decode_dim, 1),
            nn.BatchNorm2d(decode_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(decode_dim, num_classes, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """features: list of (B, C_i, H_i, W_i) -> (B, num_classes, H0, W0)."""
        target_h, target_w = features[0].shape[2:]
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.projs)):
            p = proj(feat)
            if p.shape[2:] != (target_h, target_w):
                p = F.interpolate(p, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
            projected.append(p)
        x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class VMambaSeg(nn.Module):
    """VMamba encoder-decoder for semantic segmentation.

    VMamba-Tiny: dims=[96,192,384,768], depths=[2,2,9,2], ~31M encoder params
    VMamba-Small: dims=[96,192,384,768], depths=[2,2,27,2], ~50M encoder params

    Output: (B, num_classes, H, W) logits at input resolution.
    """
    def __init__(self, num_classes: int = 3, dims: List[int] = None,
                 depths: List[int] = None, d_state: int = 16,
                 d_conv: int = 3, expand: int = 2, decode_dim: int = 256,
                 dropout: float = 0.0, pretrained_ckpt: str = None):
        super().__init__()
        if dims is None:
            dims = [96, 192, 384, 768]
        if depths is None:
            depths = [2, 2, 9, 2]

        self.encoder = VMambaEncoder(
            dims=dims, depths=depths, d_state=d_state,
            d_conv=d_conv, expand=expand, dropout=dropout)
        self.decoder = MLPDecodeHead(dims, decode_dim, num_classes)

        if pretrained_ckpt:
            self._load_pretrained(pretrained_ckpt)

    def _load_pretrained(self, ckpt_path: str):
        """Load pretrained VMamba encoder weights (classification checkpoint)."""
        import warnings
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Handle different checkpoint formats
        if 'model' in state:
            state = state['model']
        elif 'state_dict' in state:
            state = state['state_dict']

        # Filter to encoder keys only and remap
        encoder_state = {}
        for k, v in state.items():
            # Remove common prefixes from classification checkpoints
            for prefix in ('backbone.', 'encoder.', ''):
                if k.startswith(prefix):
                    new_k = k[len(prefix):]
                    break
            encoder_state[new_k] = v

        missing, unexpected = self.encoder.load_state_dict(
            encoder_state, strict=False)
        if missing:
            warnings.warn(f"VMamba pretrained: {len(missing)} missing keys "
                          f"(first 5: {missing[:5]})")
        if unexpected:
            warnings.warn(f"VMamba pretrained: {len(unexpected)} unexpected keys "
                          f"(first 5: {unexpected[:5]})")
        print(f"[VMamba] Loaded pretrained encoder from {ckpt_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, num_classes, H, W)"""
        input_h, input_w = x.shape[2:]
        features = self.encoder(x)
        logits = self.decoder(features)                # (B, C, H/4, W/4)
        logits = F.interpolate(logits, size=(input_h, input_w),
                               mode='bilinear', align_corners=False)
        return logits


def build_vmamba_seg(cfg) -> VMambaSeg:
    """Build VMambaSeg from a RunCfg."""
    backend = "mamba_ssm (CUDA)" if HAS_MAMBA_CUDA else "PyTorch (sequential, slow)"
    print(f"[VMamba] Selective scan backend: {backend}")
    return VMambaSeg(
        num_classes=3,
        dims=cfg.dims,
        depths=cfg.depths,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        decode_dim=cfg.decode_dim,
        pretrained_ckpt=cfg.pretrained_ckpt,
    )

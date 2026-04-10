"""Unified model registry: load any baseline checkpoint with a consistent interface.

Each registered model returns ``(B, C, H, W)`` logits from ``model(images)``,
where ``images`` is a ``(B, 3, H, W)`` float tensor.  Mask2Former needs
special wrapping because its native output is post-processed prediction maps.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_unet import config as C


# ---------------------------------------------------------------------------
# Registry data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    name: str
    checkpoint: Path
    img_size: int
    build_fn: Callable[[], nn.Module]
    # If set, wraps the raw model so that __call__ returns (B,C,H,W) logits.
    inference_wrapper: Optional[Callable] = None
    # Extra objects needed at inference (e.g. Mask2Former processor)
    extras: Dict = field(default_factory=dict)


_REGISTRY: Dict[str, ModelEntry] = {}


def register(entry: ModelEntry) -> None:
    _REGISTRY[entry.name] = entry


def get(name: str) -> ModelEntry:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_models() -> List[str]:
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_unet_r34() -> nn.Module:
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=C.NUM_CLASSES,
    )


def _build_deeplabv3p(encoder: str) -> nn.Module:
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=C.NUM_CLASSES,
    )


def _build_segformer_b2() -> nn.Module:
    from baseline_segformer import config as seg_C
    from transformers import SegformerForSemanticSegmentation
    return SegformerForSemanticSegmentation.from_pretrained(
        seg_C.PRETRAINED_MODEL_ID,
        num_labels=C.NUM_CLASSES,
        ignore_mismatched_sizes=True,
        id2label={0: "background", 1: "crack", 2: "spalling"},
        label2id={"background": 0, "crack": 1, "spalling": 2},
    )


def _build_mask2former():
    """Returns (model, processor) tuple."""
    from baseline_mask2former import config as m2f_C
    from baseline_mask2former.train import build_model_and_processor
    cfg = m2f_C.CFG
    return build_model_and_processor(cfg)


def _build_full_method() -> nn.Module:
    from full_method.model import SegFormerWithBoundary
    from full_method import config as fm_C
    cfg = fm_C.RunCfg()
    return SegFormerWithBoundary(cfg.pretrained, fm_C.NUM_CLASSES)


# ---------------------------------------------------------------------------
# Inference wrappers
# ---------------------------------------------------------------------------

class SegformerLogitsWrapper(nn.Module):
    """Wraps SegformerForSemanticSegmentation to return (B,C,H,W) logits."""

    def __init__(self, model: nn.Module, target_size: int):
        super().__init__()
        self.model = model
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits  # (B, C, H/4, W/4)
        return F.interpolate(
            logits, size=(self.target_size, self.target_size),
            mode="bilinear", align_corners=False,
        )

    def parameters(self, recurse=True):
        return self.model.parameters(recurse=recurse)


class FullMethodLogitsWrapper(nn.Module):
    """Wraps SegFormerWithBoundary (dict output) to return (B,C,H,W) logits."""

    def __init__(self, model: nn.Module, target_size: int):
        super().__init__()
        self.model = model
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)  # dict with seg_logits, boundary_logits
        logits = out["seg_logits"]  # (B, C, H/4, W/4)
        return F.interpolate(
            logits, size=(self.target_size, self.target_size),
            mode="bilinear", align_corners=False,
        )

    def parameters(self, recurse=True):
        return self.model.parameters(recurse=recurse)


class Mask2FormerLogitsWrapper(nn.Module):
    """Wraps Mask2Former to return (B,C,H,W) fake logits (one-hot from pred maps)."""

    def __init__(self, model: nn.Module, processor, num_classes: int,
                 target_size: int):
        super().__init__()
        self.model = model
        self.processor = processor
        self.num_classes = num_classes
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        target_sizes = [(self.target_size, self.target_size)] * x.shape[0]
        pred_maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes,
        )
        # Convert pred maps to one-hot fake logits
        device = x.device
        stacked = torch.stack(pred_maps).to(device)  # (B,H,W)
        B, H, W = stacked.shape
        fake = torch.zeros(B, self.num_classes, H, W, device=device)
        fake.scatter_(1, stacked.unsqueeze(1).long(), 1.0)
        return fake

    def parameters(self, recurse=True):
        return self.model.parameters(recurse=recurse)


# ---------------------------------------------------------------------------
# Load checkpoint into model
# ---------------------------------------------------------------------------

def load_model(name: str, device: str = "cpu") -> nn.Module:
    """Build model, load checkpoint, apply wrapper, move to device."""
    entry = get(name)
    ckpt_path = entry.checkpoint

    if "mask2former" in name:
        model, processor = _build_mask2former()
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
        wrapper = Mask2FormerLogitsWrapper(
            model, processor, C.NUM_CLASSES, entry.img_size,
        )
        wrapper.to(device).eval()
        return wrapper

    model = entry.build_fn()
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])

    if entry.inference_wrapper is not None:
        model = entry.inference_wrapper(model, entry.img_size)

    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Register all known models
# ---------------------------------------------------------------------------

_CODES = C.CODES_DIR

# U-Net ResNet34 320
register(ModelEntry(
    name="unet_r34_320",
    checkpoint=_CODES / "baseline_unet" / "runs" / "unet_r34_ce_dice" / "best.pt",
    img_size=320,
    build_fn=_build_unet_r34,
))

# DeepLabV3+ ResNet50 512
register(ModelEntry(
    name="deeplabv3p_r50_512",
    checkpoint=_CODES / "baseline_deeplab" / "runs" / "deeplabv3p_r50_512" / "best.pt",
    img_size=512,
    build_fn=functools.partial(_build_deeplabv3p, encoder="resnet50"),
))

# DeepLabV3+ ResNet34 320
register(ModelEntry(
    name="deeplabv3p_r34_320",
    checkpoint=_CODES / "baseline_deeplab" / "runs" / "deeplabv3p_r34_320" / "best.pt",
    img_size=320,
    build_fn=functools.partial(_build_deeplabv3p, encoder="resnet34"),
))

# SegFormer-B2 plain 512
register(ModelEntry(
    name="segformer_b2_plain_512",
    checkpoint=_CODES / "baseline_segformer" / "runs" / "segformer_b2_plain_512" / "best.pt",
    img_size=512,
    build_fn=_build_segformer_b2,
    inference_wrapper=SegformerLogitsWrapper,
))

# SegFormer-B2 static curriculum 512
register(ModelEntry(
    name="segformer_b2_staticcurr_512",
    checkpoint=_CODES / "baseline_segformer" / "runs" / "segformer_b2_staticcurr_512" / "best.pt",
    img_size=512,
    build_fn=_build_segformer_b2,
    inference_wrapper=SegformerLogitsWrapper,
))

# Mask2Former Swin-Small 512
register(ModelEntry(
    name="mask2former_swin_small_512",
    checkpoint=_CODES / "baseline_mask2former" / "runs" / "mask2former_swin_small_512" / "best.pt",
    img_size=512,
    build_fn=lambda: None,  # handled specially in load_model
))

# Full Method: SegFormer-B2 + Boundary Head + Dynamic Curriculum
register(ModelEntry(
    name="segformer_b2_full_512",
    checkpoint=_CODES / "full_method" / "runs" / "segformer_b2_full_512" / "best.pt",
    img_size=512,
    build_fn=_build_full_method,
    inference_wrapper=FullMethodLogitsWrapper,
))

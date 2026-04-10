"""Per-epoch qualitative visualisations from a fixed val sample set."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from . import config as C
from .dataset import (
    build_transforms,
    decode_mask,
    image_path,
    mask_path,
    read_image_rgb,
    read_mask_rgb,
)

PALETTE = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255]], dtype=np.uint8)


def pick_viz_samples(val_files: List[str], root: Path, seed: int = C.SEED) -> List[str]:
    """Return 6 fixed samples: 2 crack-only, 2 crack+spalling, 2 Hard."""
    rng = random.Random(seed)
    crack_only: List[str] = []
    crack_and_sp: List[str] = []
    hard: List[str] = []
    for rel in val_files:
        m = read_mask_rgb(mask_path(root, rel))
        label, _ = decode_mask(m)
        has_sp = bool((label == 2).any())
        has_cr = bool((label == 1).any())
        if has_cr and not has_sp:
            crack_only.append(rel)
        elif has_cr and has_sp:
            crack_and_sp.append(rel)
        if rel.startswith("Hard/"):
            hard.append(rel)

    rng.shuffle(crack_only); rng.shuffle(crack_and_sp); rng.shuffle(hard)
    picks: List[str] = []
    picks += crack_only[:2]
    picks += crack_and_sp[:2]
    # 2 Hard that are not already picked
    for rel in hard:
        if rel not in picks:
            picks.append(rel)
        if len(picks) >= 6:
            break
    # Top up with any remaining val files if we're still short
    if len(picks) < 6:
        for rel in val_files:
            if rel not in picks:
                picks.append(rel)
            if len(picks) >= 6:
                break
    return picks[:6]


def _denorm(img_t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = img_t.detach().cpu() * std + mean
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255).astype(np.uint8)


def _colorise(label: np.ndarray) -> np.ndarray:
    return PALETTE[label]


def save_preview(model: Optional[torch.nn.Module], files: List[str], root: Path,
                 out_path: Path, device: str, img_size: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tfm = build_transforms(img_size, train=False)
    n = len(files)
    fig, axs = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axs = np.expand_dims(axs, 0)

    was_training = False
    if model is not None:
        was_training = model.training
        model.eval()

    with torch.no_grad():
        for i, rel in enumerate(files):
            img = read_image_rgb(image_path(root, rel))
            mask_rgb = read_mask_rgb(mask_path(root, rel))
            label, _ = decode_mask(mask_rgb)
            out = tfm(image=img, mask=label)
            img_t = out["image"]
            mask_t = out["mask"]
            if not torch.is_tensor(mask_t):
                mask_t = torch.from_numpy(mask_t)

            img_np = _denorm(img_t)
            gt_np = _colorise(mask_t.numpy().astype(np.int64))

            if model is not None:
                logits = model(img_t.unsqueeze(0).to(device))
                pred = logits.argmax(1)[0].cpu().numpy().astype(np.int64)
                pred_np = _colorise(pred)
            else:
                pred_np = np.zeros_like(gt_np)

            axs[i, 0].imshow(img_np); axs[i, 0].set_title(rel, fontsize=8); axs[i, 0].axis("off")
            axs[i, 1].imshow(gt_np); axs[i, 1].set_title("gt"); axs[i, 1].axis("off")
            axs[i, 2].imshow(pred_np); axs[i, 2].set_title("pred"); axs[i, 2].axis("off")

    if model is not None and was_training:
        model.train()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

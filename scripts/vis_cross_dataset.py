"""Generate qualitative cross-dataset visualisation (DamSegment -> S2DS).

Picks representative S2DS test images and produces a grid:
  input | GT | SegFormer-B2 | Mask2Former | HeteroDistill | HeteroDistill+DTKD

Usage (on RunPod):
    python scripts/vis_cross_dataset.py --device cuda
    python scripts/vis_cross_dataset.py --device cuda --num-images 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_unet.dataset import build_transforms
from shared_eval.model_registry import load_model
from scripts.eval_cross_dataset import S2DSDataset

CODES_DIR = Path(__file__).resolve().parent.parent
S2DS_DIR = CODES_DIR / "Dataset" / "S2DS"

MODELS = [
    ("SegFormer-B2", "segformer_b2_plain_512"),
    ("Mask2Former", "mask2former_swin_small_512"),
    ("DSConv+SRL", "dscformer_srl_G1"),
    ("HeteroDistill", "dual_kd_classaware_DKD2"),
]


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = img.copy()
    cr = mask == 1
    sp = mask == 2
    if cr.any():
        out[cr] = (np.array([255, 60, 60]) * alpha + out[cr] * (1 - alpha)).astype(np.uint8)
    if sp.any():
        out[sp] = (np.array([60, 200, 60]) * alpha + out[sp] * (1 - alpha)).astype(np.uint8)
    return out


def pick_images(data_dir: Path, num: int = 4) -> list[str]:
    """Pick test images that contain crack or spalling (most interesting)."""
    list_path = data_dir / "test_files.txt"
    with open(list_path) as f:
        files = [ln.strip() for ln in f if ln.strip()]

    # Score by defect pixel count
    scored = []
    for stem in files:
        mask = cv2.imread(str(data_dir / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        crack_px = int((mask == 1).sum())
        spall_px = int((mask == 2).sum())
        if crack_px + spall_px == 0:
            continue
        has_both = int(crack_px > 0 and spall_px > 0)
        scored.append((stem, has_both, crack_px + spall_px))

    # Prefer images with both classes, then by defect area
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    selected = [s[0] for s in scored[:num]]
    return selected


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def generate_vis(data_dir: Path, device: str, num_images: int, output_path: Path):
    # Pick images
    stems = pick_images(data_dir, num_images)
    print(f"[vis] Selected {len(stems)} images: {stems}")

    # Build dataset for transforms
    transform = build_transforms(512, train=False)
    dataset = S2DSDataset(data_dir, stems, transform=transform)

    # Load models
    loaded_models = {}
    col_labels = ["Input", "GT"]
    for label, name in MODELS:
        try:
            loaded_models[label] = load_model(name, device=device)
            col_labels.append(label)
            print(f"[vis] Loaded {label} ({name})")
        except (KeyError, FileNotFoundError) as e:
            print(f"[vis] Skipping {label}: {e}")

    n_cols = len(col_labels)
    n_rows = len(stems)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx in range(len(stems)):
        stem = stems[row_idx]
        img_t, mask_t, _ = dataset[row_idx]

        # Raw image for display
        img_raw = cv2.imread(str(data_dir / "images" / f"{stem}.png"), cv2.IMREAD_COLOR)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # GT mask
        gt_mask = cv2.imread(str(data_dir / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt_mask, (img_raw.shape[1], img_raw.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

        # Col 0: Input
        axes[row_idx, 0].imshow(img_raw)
        axes[row_idx, 0].axis("off")

        # Col 1: GT overlay
        axes[row_idx, 1].imshow(overlay(img_raw, gt_resized))
        axes[row_idx, 1].axis("off")

        # Model predictions
        x_batch = img_t.unsqueeze(0).to(device)
        col_idx = 2
        for label, _ in MODELS:
            if label not in loaded_models:
                continue
            model = loaded_models[label]
            logits = model(x_batch)
            pred = logits.argmax(dim=1)[0].cpu().numpy()
            pred_resized = cv2.resize(pred.astype(np.uint8),
                                      (img_raw.shape[1], img_raw.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            axes[row_idx, col_idx].imshow(overlay(img_raw, pred_resized))
            axes[row_idx, col_idx].axis("off")
            col_idx += 1

    # Column titles
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=13, fontweight="bold", pad=10)

    # Legend
    legend_patches = [
        mpatches.Patch(color=(1, 0.24, 0.24), label="Crack"),
        mpatches.Patch(color=(0.24, 0.78, 0.24), label="Spalling"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=13, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"[vis] Saved: {output_path}")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight")
    print(f"[vis] Saved: {pdf_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset qualitative vis")
    parser.add_argument("--data-dir", type=str, default=str(S2DS_DIR))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--output", type=str,
                        default="workspace/final/figures/fig_cross_dataset.png")
    args = parser.parse_args()

    device = args.device or _pick_device()
    generate_vis(
        Path(args.data_dir), device, args.num_images,
        Path(args.output),
    )


if __name__ == "__main__":
    main()

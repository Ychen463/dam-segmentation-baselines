"""Generate qualitative cross-dataset visualisation (DamSegment -> S2DS).

Picks representative S2DS test images and produces a grid:
  input | GT | SegFormer-B2 | Mask2Former | DSCFormer | DSCFormer+DTKD

Usage (on RunPod):
    python scripts/vis_cross_dataset.py --device cuda
    python scripts/vis_cross_dataset.py --device cuda --num-images 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_unet.dataset import build_transforms
from shared_eval.model_registry import load_model
from scripts.eval_cross_dataset import S2DSDataset

CODES_DIR = Path(__file__).resolve().parent.parent
S2DS_DIR = CODES_DIR / "Dataset" / "S2DS"

# Colors: bg=black, crack=red, spalling=green
COLORS = np.array([
    [0, 0, 0],        # 0 bg
    [255, 0, 0],      # 1 crack
    [0, 255, 0],      # 2 spalling
], dtype=np.uint8)

MODELS = [
    ("SegFormer-B2", "segformer_b2_plain_512"),
    ("Mask2Former", "mask2former_swin_small_512"),
    ("DSCFormer", "dscformer_srl_G1"),
    ("DSCFormer+DTKD", "dual_kd_classaware_DKD2"),
]


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return COLORS[mask]


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = mask_to_rgb(mask)
    fg = mask > 0
    out = img.copy()
    out[fg] = (img[fg] * (1 - alpha) + colored[fg] * alpha).astype(np.uint8)
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
    models = {}
    for label, name in MODELS:
        try:
            models[label] = load_model(name, device=device)
            print(f"[vis] Loaded {label} ({name})")
        except (KeyError, FileNotFoundError) as e:
            print(f"[vis] Skipping {label}: {e}")

    # Generate predictions
    cell_size = 256  # display size per cell
    n_cols = 2 + len(models)  # input + GT + models
    n_rows = len(stems)

    grid = np.zeros((n_rows * cell_size, n_cols * cell_size, 3), dtype=np.uint8)

    for row_idx in range(len(stems)):
        stem = stems[row_idx]
        img_t, mask_t, _ = dataset[row_idx]

        # Raw image for display
        img_raw = cv2.imread(str(data_dir / "images" / f"{stem}.png"), cv2.IMREAD_COLOR)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_raw = cv2.resize(img_raw, (cell_size, cell_size))

        # GT mask
        gt_mask = cv2.imread(str(data_dir / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        gt_overlay = overlay(img_raw, cv2.resize(gt_mask, (cell_size, cell_size),
                                                  interpolation=cv2.INTER_NEAREST))

        # Place input and GT
        y0 = row_idx * cell_size
        grid[y0:y0 + cell_size, 0:cell_size] = img_raw
        grid[y0:y0 + cell_size, cell_size:2 * cell_size] = gt_overlay

        # Model predictions
        x_batch = img_t.unsqueeze(0).to(device)
        for col_idx, (label, _) in enumerate(MODELS):
            if label not in models:
                continue
            model = models[label]
            logits = model(x_batch)
            pred = logits.argmax(dim=1)[0].cpu().numpy()
            pred_resized = cv2.resize(pred.astype(np.uint8), (cell_size, cell_size),
                                       interpolation=cv2.INTER_NEAREST)
            pred_overlay = overlay(img_raw, pred_resized)

            x0 = (2 + col_idx) * cell_size
            grid[y0:y0 + cell_size, x0:x0 + cell_size] = pred_overlay

    # Add column headers
    header_h = 30
    header = np.ones((header_h, n_cols * cell_size, 3), dtype=np.uint8) * 255
    col_labels = ["Input", "GT"] + [label for label, _ in MODELS]
    for i, label in enumerate(col_labels):
        x = i * cell_size + 10
        cv2.putText(header, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    final = np.vstack([header, grid])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"[vis] Saved: {output_path}")


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

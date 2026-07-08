"""Generate supplementary qualitative gallery.

Produces predictions for six sample categories:
  1. Random samples (5 images)
  2. Median-performance samples (closest to median per-image mIoU_fg)
  3. Failure cases (lowest per-image mIoU_fg)
  4. Hairline/small crack samples (thinnest mean crack width)
  5. False-positive-heavy samples (lowest crack precision)
  6. Crack-spalling coexistence (both classes present, sorted by overlap)

Usage on RunPod:
    cd /workspace/dam-segmentation-baselines
    python scripts/gen_supplementary_qualitative.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from baseline_unet.dataset import DamSegmentDataset, build_transforms, read_split_file
from baseline_unet import config as base_C
from full_method import config as C
from full_method.model import DSCformerDam, SegFormerWithBoundary
from shared_eval.model_registry import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT / "figures" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors: 0=bg (transparent), 1=crack (red), 2=spalling (green)
CMAP = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.55), (0, 0.7, 0, 0.55)])


def load_models():
    """Load SegFormer-B2, DSConv+SRL (T1), and HeteroDistill."""
    models = {}

    # SegFormer-B2
    models["SegFormer-B2"] = load_model("segformer_b2_plain_512", DEVICE)

    # DSConv+SRL (Teacher 1)
    cfg = C.RunCfg()
    cfg.use_boundary_loss = False
    t1 = DSCformerDam(cfg.pretrained, cfg=cfg).to(DEVICE)
    t1_ckpt = ROOT / "full_method" / "runs" / "dscformer_srl_G1" / "best.pt"
    if t1_ckpt.exists():
        state = torch.load(t1_ckpt, map_location=DEVICE, weights_only=False)
        key = "ema_model" if "ema_model" in state else "model"
        t1.load_state_dict(state[key], strict=False)
    t1.eval()

    class T1Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return F.interpolate(self.m(x)["seg_logits"], x.shape[-2:],
                                 mode="bilinear", align_corners=False)
    models["DSConv+SRL"] = T1Wrapper(t1)

    # HeteroDistill
    topo = DSCformerDam(cfg.pretrained, cfg=cfg).to(DEVICE)
    topo_ckpt = ROOT / "full_method" / "runs" / "dkd10_no_srl_rerun" / "best.pt"
    if topo_ckpt.exists():
        state = torch.load(topo_ckpt, map_location=DEVICE, weights_only=False)
        key = "ema_model" if "ema_model" in state else "model"
        topo.load_state_dict(state[key], strict=False)
    topo.eval()

    class TopoWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return F.interpolate(self.m(x)["seg_logits"], x.shape[-2:],
                                 mode="bilinear", align_corners=False)
    models["HeteroDistill"] = TopoWrapper(topo)

    return models


def compute_per_image_stats(models, test_loader):
    """Compute per-image metrics for sample selection."""
    stats = []
    with torch.no_grad():
        for imgs, masks, meta in test_loader:
            imgs = imgs.to(DEVICE).float()
            masks = masks.numpy()
            B = imgs.shape[0]

            preds = {}
            for name, model in models.items():
                logits = model(imgs)
                if isinstance(logits, dict):
                    logits = logits["seg_logits"]
                    logits = F.interpolate(logits, (512, 512),
                                           mode="bilinear", align_corners=False)
                pred = logits.argmax(1).cpu().numpy()
                preds[name] = pred

            for i in range(B):
                mask = masks[i]
                topo_pred = preds["HeteroDistill"][i]

                # Per-class IoU for HeteroDistill
                ious = {}
                for cls, cname in [(1, "crack"), (2, "spalling")]:
                    gt_c = (mask == cls)
                    pr_c = (topo_pred == cls)
                    inter = (gt_c & pr_c).sum()
                    union = (gt_c | pr_c).sum()
                    ious[cname] = inter / max(union, 1)

                has_crack = (mask == 1).sum() > 0
                has_spalling = (mask == 2).sum() > 0

                # Crack width (mean distance transform value on crack skeleton)
                crack_width = 0
                if has_crack:
                    from scipy import ndimage
                    crack_mask = (mask == 1).astype(np.uint8)
                    dt = ndimage.distance_transform_edt(crack_mask)
                    from skimage.morphology import skeletonize
                    skel = skeletonize(crack_mask > 0)
                    if skel.sum() > 0:
                        crack_width = dt[skel].mean()

                # Crack precision
                crack_prec = 0
                if has_crack:
                    tp = ((topo_pred == 1) & (mask == 1)).sum()
                    fp = ((topo_pred == 1) & (mask != 1)).sum()
                    crack_prec = tp / max(tp + fp, 1)

                mIoU_fg = 0
                n_cls = 0
                if has_crack:
                    mIoU_fg += ious["crack"]
                    n_cls += 1
                if has_spalling:
                    mIoU_fg += ious["spalling"]
                    n_cls += 1
                if n_cls > 0:
                    mIoU_fg /= n_cls

                stats.append({
                    "idx": len(stats),
                    "fname": meta[i] if isinstance(meta, (list, tuple)) else str(meta),
                    "mIoU_fg": float(mIoU_fg),
                    "IoU_crack": float(ious.get("crack", 0)),
                    "IoU_spalling": float(ious.get("spalling", 0)),
                    "has_crack": bool(has_crack),
                    "has_spalling": bool(has_spalling),
                    "crack_width": float(crack_width),
                    "crack_prec": float(crack_prec),
                    "imgs": imgs[i].cpu(),
                    "mask": mask,
                    "preds": {k: v[i] for k, v in preds.items()},
                })
    return stats


def select_samples(stats):
    """Select samples for each category."""
    selections = {}

    # Filter to images with at least one FG class
    fg = [s for s in stats if s["has_crack"] or s["has_spalling"]]

    # 1. Random (5 images)
    rng = np.random.RandomState(42)
    selections["1_random"] = rng.choice(fg, min(5, len(fg)), replace=False).tolist()

    # 2. Median performance (3 images closest to median mIoU_fg)
    mious = np.array([s["mIoU_fg"] for s in fg])
    median = np.median(mious)
    dists = np.abs(mious - median)
    median_idxs = np.argsort(dists)[:3]
    selections["2_median"] = [fg[i] for i in median_idxs]

    # 3. Failure cases (5 lowest mIoU_fg)
    sorted_by_miou = sorted(fg, key=lambda s: s["mIoU_fg"])
    selections["3_failure"] = sorted_by_miou[:5]

    # 4. Hairline cracks (5 thinnest, must have crack)
    crack_imgs = [s for s in fg if s["has_crack"] and s["crack_width"] > 0]
    sorted_by_width = sorted(crack_imgs, key=lambda s: s["crack_width"])
    selections["4_hairline"] = sorted_by_width[:5]

    # 5. False-positive-heavy (5 lowest crack precision, must have crack)
    crack_pred = [s for s in fg if s["has_crack"]]
    sorted_by_prec = sorted(crack_pred, key=lambda s: s["crack_prec"])
    selections["5_false_positive"] = sorted_by_prec[:5]

    # 6. Crack-spalling coexistence (5 images with both classes)
    both = [s for s in fg if s["has_crack"] and s["has_spalling"]]
    sorted_both = sorted(both, key=lambda s: s["mIoU_fg"])
    selections["6_coexistence"] = sorted_both[:5]

    return selections


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def plot_gallery(selections, out_dir):
    """Plot and save each category as a separate figure."""
    model_names = ["SegFormer-B2", "DSConv+SRL", "HeteroDistill"]

    for cat_name, samples in selections.items():
        n = len(samples)
        if n == 0:
            continue

        ncols = 2 + len(model_names)  # input, GT, preds
        fig, axes = plt.subplots(n, ncols, figsize=(ncols * 2.5, n * 2.5))
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, s in enumerate(samples):
            # Denormalize image
            img = s["imgs"].permute(1, 2, 0).numpy()
            img = img * IMAGENET_STD + IMAGENET_MEAN
            img = np.clip(img, 0, 1)

            # Input
            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f"mIoU={s['mIoU_fg']:.2f}", fontsize=8)
            axes[row, 0].axis("off")

            # GT
            axes[row, 1].imshow(img)
            axes[row, 1].imshow(s["mask"], cmap=CMAP, vmin=0, vmax=2)
            axes[row, 1].set_title("GT", fontsize=8)
            axes[row, 1].axis("off")

            # Predictions
            for j, mname in enumerate(model_names):
                axes[row, 2 + j].imshow(img)
                axes[row, 2 + j].imshow(s["preds"][mname], cmap=CMAP, vmin=0, vmax=2)
                iou_cr = s["IoU_crack"] if mname == "HeteroDistill" else ""
                axes[row, 2 + j].set_title(mname, fontsize=8)
                axes[row, 2 + j].axis("off")

        # Column headers only on first row
        col_titles = ["Input", "GT"] + model_names
        for j, t in enumerate(col_titles):
            axes[0, j].set_title(t, fontsize=9, fontweight="bold")

        fig.suptitle(cat_name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_path = out_dir / f"supp_{cat_name}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    print(f"Device: {DEVICE}")

    cfg = C.RunCfg()
    cfg.use_boundary_loss = False

    test_files = read_split_file(C.SPLIT_FILES["test"])
    test_ds = DamSegmentDataset(base_C.DATA_ROOT, test_files,
                                 build_transforms(512, train=False))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                              num_workers=2, pin_memory=(DEVICE == "cuda"))
    print(f"Test set: {len(test_files)} images")

    print("Loading models...")
    models = load_models()

    print("Computing per-image stats...")
    stats = compute_per_image_stats(models, test_loader)

    print("Selecting samples...")
    selections = select_samples(stats)
    for cat, samples in selections.items():
        print(f"  {cat}: {len(samples)} samples")

    print("Generating gallery...")
    plot_gallery(selections, OUT_DIR)
    print(f"\nAll galleries saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

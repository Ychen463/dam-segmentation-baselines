"""Group-based split evaluation to assess patch-level spatial leakage.

Since DamSegment has no source-image metadata, we cluster patches by visual
similarity (ImageNet-pretrained ResNet-50 features + agglomerative clustering)
and create a group-level split where all patches in a cluster are assigned to
the same fold.  This prevents visually similar (potentially adjacent) patches
from appearing in both train and test.

Usage::

    # Step 1: Build the group-based split
    python scripts/eval_group_split.py --build-split

    # Step 2: Evaluate models on the new split's test set
    python scripts/eval_group_split.py --evaluate

    # Both steps:
    python scripts/eval_group_split.py --build-split --evaluate

    # Customise clustering:
    python scripts/eval_group_split.py --build-split --n-clusters 120
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_unet import config as C
from baseline_unet.dataset import (
    build_transforms,
    decode_mask,
    has_spalling_from_mask,
    image_path,
    mask_path,
    read_mask_rgb,
)
from shared_eval.metrics_full import SegMetricsFull
from shared_eval.model_registry import load_model, list_models

CODES_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = CODES_DIR / "baseline_unet" / "splits"
GROUP_SPLIT_DIR = SPLIT_DIR / "group_split"
BF1_TOLERANCE_PX = 2
SEED = 42

# Models to evaluate
DEFAULT_MODELS = [
    "segformer_b2_plain_512",
    "dscformer_srl_G1",
    "dual_kd_classaware_DKD2",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(data_root: Path, all_rels: List[str],
                     batch_size: int = 16) -> np.ndarray:
    """Extract ImageNet-pretrained ResNet-50 features for all patches."""
    import torchvision.models as models
    import torchvision.transforms as T

    device = _pick_device()
    print(f"[group-split] Extracting features on {device} ...")

    # Use ResNet-50 up to avgpool (2048-d)
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = torch.nn.Identity()
    backbone.to(device).eval()

    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for i in range(0, len(all_rels), batch_size):
        batch_rels = all_rels[i:i + batch_size]
        tensors = []
        for rel in batch_rels:
            img = cv2.imread(str(image_path(data_root, rel)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(preprocess(img))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feat = backbone(batch).cpu().numpy()
        features.append(feat)
        if (i // batch_size) % 10 == 0:
            print(f"  {i}/{len(all_rels)} ...")

    features = np.concatenate(features, axis=0)
    # L2-normalise for cosine-like clustering
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)
    print(f"[group-split] Extracted {features.shape[0]} x {features.shape[1]} features")
    return features


# ---------------------------------------------------------------------------
# Clustering and split creation
# ---------------------------------------------------------------------------

def cluster_patches(features: np.ndarray, n_clusters: int) -> np.ndarray:
    """Agglomerative clustering into n_clusters groups."""
    from sklearn.cluster import AgglomerativeClustering

    print(f"[group-split] Clustering {len(features)} patches into {n_clusters} groups ...")
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(features)
    sizes = np.bincount(labels)
    print(f"[group-split] Cluster sizes: min={sizes.min()}, max={sizes.max()}, "
          f"median={np.median(sizes):.0f}, mean={sizes.mean():.1f}")
    return labels


def build_group_split(
    all_rels: List[str],
    cluster_labels: np.ndarray,
    data_root: Path,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> Dict[str, List[str]]:
    """Split by cluster: entire clusters go to the same fold."""
    rng = random.Random(SEED)

    # Group rels by cluster
    clusters: Dict[int, List[str]] = {}
    for rel, cid in zip(all_rels, cluster_labels):
        clusters.setdefault(int(cid), []).append(rel)

    # Sort clusters for determinism, then shuffle
    cluster_ids = sorted(clusters.keys())
    rng.shuffle(cluster_ids)

    # Assign clusters to splits, tracking cumulative sample count
    total = len(all_rels)
    n_test_target = int(total * test_ratio)
    n_val_target = int(total * val_ratio)

    test_rels, val_rels, train_rels = [], [], []
    test_count, val_count = 0, 0

    for cid in cluster_ids:
        members = clusters[cid]
        if test_count < n_test_target:
            test_rels.extend(members)
            test_count += len(members)
        elif val_count < n_val_target:
            val_rels.extend(members)
            val_count += len(members)
        else:
            train_rels.extend(members)

    rng.shuffle(train_rels)
    rng.shuffle(val_rels)
    rng.shuffle(test_rels)

    # Print tier distribution per split
    for name, rels in [("train", train_rels), ("val", val_rels), ("test", test_rels)]:
        tiers = [r.split("/")[0] for r in rels]
        tier_counts = {t: tiers.count(t) for t in ("Easy", "Medium", "Hard")}
        # Check spalling distribution
        n_sp = sum(1 for r in rels
                   if has_spalling_from_mask(read_mask_rgb(mask_path(data_root, r))))
        print(f"[group-split] {name}: n={len(rels)}, tiers={tier_counts}, "
              f"has_spalling={n_sp}")

    return {"train": train_rels, "val": val_rels, "test": test_rels}


def save_group_split(splits: Dict[str, List[str]], out_dir: Path,
                     cluster_labels: np.ndarray, all_rels: List[str]) -> None:
    """Save split files and cluster metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rels in splits.items():
        path = out_dir / f"{name}.txt"
        with open(path, "w") as f:
            for rel in rels:
                f.write(rel + "\n")
        print(f"[group-split] Saved {path} ({len(rels)} samples)")

    # Save cluster assignments for reproducibility
    meta = {rel: int(cid) for rel, cid in zip(all_rels, cluster_labels)}
    with open(out_dir / "cluster_assignments.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[group-split] Saved cluster assignments to {out_dir / 'cluster_assignments.json'}")


# ---------------------------------------------------------------------------
# Dataset for evaluation using arbitrary split file
# ---------------------------------------------------------------------------

class DamSegDataset(Dataset):
    """Simple dataset that loads images + masks from a split file."""

    def __init__(self, data_root: Path, rels: List[str], img_size: int = 512):
        self.data_root = data_root
        self.rels = rels
        self.transform = build_transforms(img_size, train=False)

    def __len__(self) -> int:
        return len(self.rels)

    def __getitem__(self, idx: int):
        rel = self.rels[idx]
        img = cv2.imread(str(image_path(self.data_root, rel)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = read_mask_rgb(mask_path(self.data_root, rel))
        label, _ = decode_mask(m)
        out = self.transform(image=img, mask=label)
        image_t = out["image"]
        mask_t = out["mask"]
        if not torch.is_tensor(mask_t):
            mask_t = torch.from_numpy(mask_t)
        return image_t, mask_t.long(), rel


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate_model(model_name: str, test_rels: List[str],
                   data_root: Path, device: str) -> Dict[str, float]:
    """Evaluate a single model on the group-split test set."""
    from shared_eval.model_registry import get as get_entry
    entry = get_entry(model_name)
    model = load_model(model_name, device=device)

    dataset = DamSegDataset(data_root, test_rels, img_size=entry.img_size)
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        num_workers=2, pin_memory=(device == "cuda"))

    metrics = SegMetricsFull(C.NUM_CLASSES, BF1_TOLERANCE_PX)
    metrics.reset()

    for images, masks, rels in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        metrics.update(logits, masks)

    return metrics.compute()


def format_results(model_name: str, results: Dict[str, float]) -> str:
    r = results
    lines = [
        f"\n{'='*60}",
        f"  {model_name}  →  Group-split test",
        f"{'='*60}",
        f"  IoU     bg={r['IoU_background']:.4f}  crack={r['IoU_crack']:.4f}"
        f"  spalling={r['IoU_spalling']:.4f}  | mIoU_fg={r['mIoU_fg']:.4f}",
        f"  Dice    crack={r['Dice_crack']:.4f}  spalling={r['Dice_spalling']:.4f}",
        f"  BF1     crack={r['BF1_crack']:.4f}  spalling={r['BF1_spalling']:.4f}"
        f"  | BF1_fg={r['BF1_fg_mean']:.4f}",
        f"  clDice  crack={r['clDice_crack']:.4f}  spalling={r['clDice_spalling']:.4f}"
        f"  | clDice_fg={r['clDice_fg_mean']:.4f}",
    ]
    return "\n".join(lines)


def format_comparison(orig: Dict[str, float], group: Dict[str, float],
                      model_name: str) -> str:
    """Side-by-side comparison of original vs group-split metrics."""
    keys = ["mIoU_fg", "IoU_crack", "IoU_spalling",
            "BF1_fg_mean", "BF1_crack", "BF1_spalling",
            "clDice_crack", "clDice_spalling"]
    lines = [f"\n  {model_name}: original-split vs group-split"]
    lines.append(f"  {'metric':<18} {'orig':>8} {'group':>8} {'delta':>8}")
    lines.append(f"  {'-'*42}")
    for k in keys:
        o = orig.get(k, float("nan"))
        g = group.get(k, float("nan"))
        d = g - o
        lines.append(f"  {k:<18} {o*100:>7.2f}% {g*100:>7.2f}% {d*100:>+7.2f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluate on original split for comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_original_split(model_name: str, data_root: Path,
                            device: str) -> Dict[str, float]:
    """Evaluate model on the original test split for comparison."""
    test_file = SPLIT_DIR / "test.txt"
    with open(test_file) as f:
        test_rels = [ln.strip() for ln in f if ln.strip()]
    return evaluate_model(model_name, test_rels, data_root, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Group-based split evaluation")
    parser.add_argument("--build-split", action="store_true",
                        help="Build the group-based split (extract features + cluster)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate models on the group-split test set")
    parser.add_argument("--n-clusters", type=int, default=100,
                        help="Number of visual-similarity clusters (default: 100)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to evaluate (default: key models)")
    parser.add_argument("--compare", action="store_true",
                        help="Also evaluate on original split for side-by-side comparison")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for feature extraction")
    args = parser.parse_args()

    if not args.build_split and not args.evaluate:
        parser.error("Specify --build-split and/or --evaluate")

    data_root = C.DATA_ROOT

    # Scan all images
    all_rels = []
    for diff in C.DIFFICULTIES:
        img_dir = data_root / diff / "Images"
        for p in sorted(img_dir.glob("*.jpg")):
            all_rels.append(f"{diff}/{p.name}")
    print(f"[group-split] Total images: {len(all_rels)}")

    if args.build_split:
        features = extract_features(data_root, all_rels, args.batch_size)
        cluster_labels = cluster_patches(features, args.n_clusters)
        splits = build_group_split(all_rels, cluster_labels, data_root)
        save_group_split(splits, GROUP_SPLIT_DIR, cluster_labels, all_rels)

    if args.evaluate:
        # Load group-split test set
        test_file = GROUP_SPLIT_DIR / "test.txt"
        if not test_file.exists():
            print(f"[ERROR] {test_file} not found. Run with --build-split first.")
            sys.exit(1)
        with open(test_file) as f:
            test_rels = [ln.strip() for ln in f if ln.strip()]
        print(f"\n[group-split] Evaluating on {len(test_rels)} group-split test images")

        device = _pick_device()
        model_names = args.models or DEFAULT_MODELS

        # Filter to models with existing checkpoints
        available = list_models()
        model_names = [m for m in model_names if m in available]

        all_results = {}
        csv_rows = []

        for model_name in model_names:
            print(f"\n--- Evaluating: {model_name} ---")
            group_results = evaluate_model(model_name, test_rels, data_root, device)
            all_results[model_name] = group_results
            print(format_results(model_name, group_results))

            row = {"model": model_name}
            for k, v in group_results.items():
                row[k] = f"{v:.4f}"

            if args.compare:
                orig_results = evaluate_original_split(model_name, data_root, device)
                print(format_comparison(orig_results, group_results, model_name))
                for k, v in orig_results.items():
                    row[f"orig_{k}"] = f"{v:.4f}"

            csv_rows.append(row)

        # Save CSV
        out_csv = GROUP_SPLIT_DIR / "group_split_results.csv"
        if csv_rows:
            keys = list(csv_rows[0].keys())
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in csv_rows:
                    w.writerow(row)
            print(f"\n[group-split] Results saved to {out_csv}")


if __name__ == "__main__":
    main()

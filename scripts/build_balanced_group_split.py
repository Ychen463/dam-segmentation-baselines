"""Build a balanced group-based split that prevents patch-level leakage.

Problem with the original 100-cluster split:
  - 4 mega-clusters (sizes 424, 218, 174, 127) hold 63% of data
  - Assigning whole clusters to splits makes 80/10/10 impossible
  - Resulting 631/369/500 split has severe tier imbalance

Solution:
  1. Load existing cluster assignments (100 clusters from ResNet-50 features)
  2. Extract ResNet-50 features (or load from cache)
  3. Sub-divide large clusters (size > max_group_size) via secondary clustering
     on the same ResNet-50 features, creating smaller "groups"
  4. Allocate groups to train/val/test using constrained greedy optimisation
     targeting 80/10/10 ratio with balanced tier distribution

Two-step usage on RunPod::

    # Step 1: Extract features and cache them (only needed once)
    python scripts/build_balanced_group_split.py --extract-features

    # Step 2: Build the split (fast, no GPU needed)
    python scripts/build_balanced_group_split.py

    # Or do both in one go:
    python scripts/build_balanced_group_split.py --extract-features

    # Custom thresholds
    python scripts/build_balanced_group_split.py --max-group-size 40

    # Dry run (print stats without saving)
    python scripts/build_balanced_group_split.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

CODES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODES_DIR))

SPLIT_DIR = CODES_DIR / "baseline_unet" / "splits"
GROUP_SPLIT_DIR = SPLIT_DIR / "group_split"
BALANCED_SPLIT_DIR = SPLIT_DIR / "balanced_group_split"
FEATURE_CACHE = GROUP_SPLIT_DIR / "resnet50_features.npy"
SEED = 42


# ---------------------------------------------------------------------------
# Data scanning
# ---------------------------------------------------------------------------

def scan_all_rels() -> List[str]:
    """Scan dataset for all image relative paths (sorted, deterministic)."""
    from baseline_unet import config as C

    all_rels = []
    for diff in C.DIFFICULTIES:
        img_dir = C.DATA_ROOT / diff / "Images"
        for p in sorted(img_dir.glob("*.jpg")):
            all_rels.append(f"{diff}/{p.name}")
    return all_rels


def load_cluster_assignments() -> Dict[str, int]:
    """Load existing cluster_assignments.json from the original group split."""
    path = GROUP_SPLIT_DIR / "cluster_assignments.json"
    if not path.exists():
        print(f"[ERROR] {path} not found. "
              "Run eval_group_split.py --build-split first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Feature extraction (requires torch, torchvision, PIL — RunPod only)
# ---------------------------------------------------------------------------

def extract_and_cache_features(all_rels: List[str], batch_size: int = 16) -> None:
    """Extract ResNet-50 features for all images and save to .npy cache."""
    from PIL import Image
    import torch
    import torchvision.models as models
    import torchvision.transforms as T
    from baseline_unet import config as C

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[features] Extracting ResNet-50 features on {device} ...")

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = torch.nn.Identity()
    backbone.to(device).eval()

    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    data_root = C.DATA_ROOT
    features = []
    for i in range(0, len(all_rels), batch_size):
        batch_rels = all_rels[i:i + batch_size]
        tensors = []
        for rel in batch_rels:
            tier = rel.split("/")[0]
            img_path = data_root / tier / "Images" / rel.split("/")[1]
            img = Image.open(str(img_path)).convert("RGB")
            tensors.append(preprocess(img))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feat = backbone(batch).cpu().numpy()
        features.append(feat)
        if (i // batch_size) % 20 == 0:
            print(f"  {i}/{len(all_rels)} ...")

    features = np.concatenate(features, axis=0)
    # L2-normalise for cosine clustering
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)

    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(FEATURE_CACHE, features)
    print(f"[features] Saved {features.shape} features to {FEATURE_CACHE}")


def load_cached_features(n_expected: int) -> np.ndarray:
    """Load cached features, exit with message if not found."""
    if not FEATURE_CACHE.exists():
        print(f"[ERROR] Feature cache not found at {FEATURE_CACHE}")
        print("  Run with --extract-features first:")
        print("    python scripts/build_balanced_group_split.py --extract-features")
        sys.exit(1)
    features = np.load(FEATURE_CACHE)
    if features.shape[0] != n_expected:
        print(f"[ERROR] Feature cache has {features.shape[0]} entries, "
              f"expected {n_expected}. Re-run with --extract-features.")
        sys.exit(1)
    print(f"[balanced-split] Loaded cached features: {features.shape}")
    return features


# ---------------------------------------------------------------------------
# Sub-divide large clusters
# ---------------------------------------------------------------------------

def subdivide_large_clusters(
    assignments: Dict[str, int],
    max_group_size: int,
    features: np.ndarray,
    all_rels: List[str],
) -> Dict[str, int]:
    """Sub-divide clusters larger than max_group_size via secondary clustering.

    Returns a new mapping: rel -> group_id (finer than original cluster_id).
    """
    from sklearn.cluster import AgglomerativeClustering

    rel_to_idx = {rel: i for i, rel in enumerate(all_rels)}

    # Group rels by original cluster
    clusters: Dict[int, List[str]] = defaultdict(list)
    for rel, cid in assignments.items():
        clusters[cid].append(rel)

    new_assignments: Dict[str, int] = {}
    next_group_id = 0
    n_subdivided = 0
    large_info = []

    for cid in sorted(clusters.keys()):
        members = clusters[cid]

        if len(members) <= max_group_size:
            for rel in members:
                new_assignments[rel] = next_group_id
            next_group_id += 1
        else:
            # Determine number of sub-clusters
            n_sub = max(2, (len(members) + max_group_size - 1) // max_group_size)
            idxs = [rel_to_idx[r] for r in members]
            sub_features = features[idxs]

            clustering = AgglomerativeClustering(
                n_clusters=min(n_sub, len(members)),
                metric="cosine",
                linkage="average",
            )
            sub_labels = clustering.fit_predict(sub_features)

            sub_groups: Dict[int, List[str]] = defaultdict(list)
            for rel, sl in zip(members, sub_labels):
                sub_groups[sl].append(rel)

            sub_sizes = [len(v) for v in sub_groups.values()]
            large_info.append(
                f"  cluster {cid}: {len(members)} -> "
                f"{n_sub} sub-groups (sizes: {sorted(sub_sizes, reverse=True)})"
            )

            n_subdivided += 1
            for sl in sorted(sub_groups.keys()):
                for rel in sub_groups[sl]:
                    new_assignments[rel] = next_group_id
                next_group_id += 1

    print(f"[balanced-split] Subdivided {n_subdivided} large clusters "
          f"(threshold={max_group_size})")
    for info in large_info:
        print(info)
    print(f"[balanced-split] {len(clusters)} original clusters -> "
          f"{next_group_id} groups")

    # Report group size stats
    group_sizes = Counter(new_assignments.values())
    sizes = sorted(group_sizes.values())
    print(f"[balanced-split] Group sizes: min={min(sizes)}, max={max(sizes)}, "
          f"median={sizes[len(sizes)//2]}, mean={sum(sizes)/len(sizes):.1f}")

    return new_assignments


# ---------------------------------------------------------------------------
# Balanced allocation
# ---------------------------------------------------------------------------

def get_tier(rel: str) -> str:
    return rel.split("/")[0]


def allocate_groups_balanced(
    group_assignments: Dict[str, int],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> Dict[str, List[str]]:
    """Allocate groups to train/val/test with tier-balanced constrained greedy.

    Per tier independently:
      1. Sort groups largest-first (with random tiebreak)
      2. Greedily assign each group to the split with the largest
         remaining deficit (target - current), prioritising test > val > train
    """
    rng = random.Random(SEED)

    # Group rels by group_id
    groups: Dict[int, List[str]] = defaultdict(list)
    for rel, gid in group_assignments.items():
        groups[gid].append(rel)

    # Dominant tier per group
    group_tier: Dict[int, str] = {}
    for gid, members in groups.items():
        tiers = Counter(get_tier(r) for r in members)
        group_tier[gid] = tiers.most_common(1)[0][0]

    tier_groups: Dict[str, List[int]] = {"Easy": [], "Medium": [], "Hard": []}
    for gid, tier in group_tier.items():
        tier_groups[tier].append(gid)

    train_rels, val_rels, test_rels = [], [], []

    for tier in ("Easy", "Medium", "Hard"):
        gids = tier_groups[tier]
        tier_total = sum(len(groups[gid]) for gid in gids)

        target_test = int(round(tier_total * test_ratio))
        target_val = int(round(tier_total * val_ratio))
        target_train = tier_total - target_test - target_val

        # Largest-first with random tiebreak
        rng.shuffle(gids)
        gids.sort(key=lambda g: len(groups[g]), reverse=True)

        current = {"train": 0, "val": 0, "test": 0}
        targets = {"train": target_train, "val": target_val, "test": target_test}
        tier_splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        for gid in gids:
            members = groups[gid]
            n = len(members)

            deficits = {s: targets[s] - current[s] for s in ("test", "val", "train")}

            # Pick split with largest positive deficit; fallback to train
            best_split = max(
                ("test", "val", "train"),
                key=lambda s: (deficits[s] > 0, deficits[s])
            )

            tier_splits[best_split].extend(members)
            current[best_split] += n

        train_rels.extend(tier_splits["train"])
        val_rels.extend(tier_splits["val"])
        test_rels.extend(tier_splits["test"])

        print(f"[balanced-split] {tier}: "
              f"train={current['train']} (target {target_train}), "
              f"val={current['val']} (target {target_val}), "
              f"test={current['test']} (target {target_test})")

    rng.shuffle(train_rels)
    rng.shuffle(val_rels)
    rng.shuffle(test_rels)

    return {"train": train_rels, "val": val_rels, "test": test_rels}


# ---------------------------------------------------------------------------
# Diagnostics and saving
# ---------------------------------------------------------------------------

def print_split_diagnostics(splits: Dict[str, List[str]]) -> None:
    total = sum(len(v) for v in splits.values())

    print(f"\n{'='*60}")
    print(f"  BALANCED GROUP SPLIT DIAGNOSTICS")
    print(f"{'='*60}")

    for name in ("train", "val", "test"):
        rels = splits[name]
        n = len(rels)
        tiers = Counter(get_tier(r) for r in rels)
        pct = n / total * 100

        print(f"\n  {name}: {n} samples ({pct:.1f}%)")
        for t in ("Easy", "Medium", "Hard"):
            tc = tiers.get(t, 0)
            tp = tc / n * 100 if n > 0 else 0
            print(f"    {t}: {tc} ({tp:.1f}%)")

    print(f"\n  Total: {total}")

    # Compare with original split
    print(f"\n  Comparison with original split:")
    print(f"  {'':15} {'Original':>10} {'Balanced':>10}")
    print(f"  {'train':15} {'1200':>10} {len(splits['train']):>10}")
    print(f"  {'val':15} {'150':>10} {len(splits['val']):>10}")
    print(f"  {'test':15} {'150':>10} {len(splits['test']):>10}")
    print(f"{'='*60}\n")


def check_no_group_leakage(
    splits: Dict[str, List[str]],
    group_assignments: Dict[str, int],
) -> bool:
    """Verify no group spans multiple splits."""
    split_of_group: Dict[int, str] = {}
    ok = True
    for split_name, rels in splits.items():
        for rel in rels:
            gid = group_assignments[rel]
            if gid in split_of_group and split_of_group[gid] != split_name:
                print(f"[ERROR] Group {gid} appears in both "
                      f"{split_of_group[gid]} and {split_name}!")
                ok = False
            split_of_group[gid] = split_name

    if ok:
        n_groups = len(set(group_assignments.values()))
        groups_per_split = Counter(split_of_group.values())
        print(f"[balanced-split] Leakage check PASSED. "
              f"{n_groups} groups -> {dict(groups_per_split)}")
    return ok


def save_split(
    splits: Dict[str, List[str]],
    group_assignments: Dict[str, int],
    out_dir: Path,
    max_group_size: int,
) -> None:
    """Save split files, group metadata, and summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rels in splits.items():
        path = out_dir / f"{name}.txt"
        with open(path, "w") as f:
            for rel in sorted(rels):
                f.write(rel + "\n")
        print(f"[balanced-split] Saved {path} ({len(rels)} samples)")

    # Group assignments
    meta_path = out_dir / "group_assignments.json"
    with open(meta_path, "w") as f:
        json.dump(group_assignments, f, indent=2, sort_keys=True)
    print(f"[balanced-split] Saved group assignments to {meta_path}")

    # Summary
    summary = {
        "seed": SEED,
        "max_group_size": max_group_size,
        "n_groups": len(set(group_assignments.values())),
        "splits": {},
    }
    for name, rels in splits.items():
        tiers = Counter(get_tier(r) for r in rels)
        summary["splits"][name] = {
            "total": len(rels),
            "Easy": tiers.get("Easy", 0),
            "Medium": tiers.get("Medium", 0),
            "Hard": tiers.get("Hard", 0),
        }

    summary_path = out_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[balanced-split] Saved summary to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build balanced group-based split for leakage prevention")
    parser.add_argument("--extract-features", action="store_true",
                        help="Extract ResNet-50 features and cache to .npy "
                             "(requires torch, torchvision, cv2)")
    parser.add_argument("--max-group-size", type=int, default=30,
                        help="Max images per group before sub-dividing (default: 30)")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print diagnostics without saving")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: balanced_group_split/)")
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    all_rels = scan_all_rels()
    print(f"[balanced-split] Found {len(all_rels)} images")

    # --- Feature extraction (optional, GPU step) ---
    if args.extract_features:
        extract_and_cache_features(all_rels)
        print(f"[balanced-split] Feature extraction complete.")
        if not args.dry_run and args.extract_features:
            # Continue to build split
            pass

    # --- Build split ---
    print(f"\n[balanced-split] Building balanced group split ...")
    print(f"[balanced-split] Target ratios: train={args.train_ratio:.0%} "
          f"val={args.val_ratio:.0%} test={args.test_ratio:.0%}")
    print(f"[balanced-split] Max group size: {args.max_group_size}")

    # Load inputs
    orig_assignments = load_cluster_assignments()
    features = load_cached_features(len(all_rels))

    # Sub-divide
    group_assignments = subdivide_large_clusters(
        orig_assignments, args.max_group_size, features, all_rels)

    # Allocate
    splits = allocate_groups_balanced(
        group_assignments,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # Diagnostics
    print_split_diagnostics(splits)
    check_no_group_leakage(splits, group_assignments)

    # Save
    if not args.dry_run:
        out_dir = Path(args.out_dir) if args.out_dir else BALANCED_SPLIT_DIR
        save_split(splits, group_assignments, out_dir, args.max_group_size)
        print(f"\n[balanced-split] Done! Split saved to {out_dir}")
    else:
        print(f"\n[balanced-split] Dry run complete. Remove --dry-run to save.")


if __name__ == "__main__":
    main()

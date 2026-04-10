"""Generate stratified training subsets for low-data experiments (Experiment E).

Usage::

    python -m shared_eval.subset_splits --fractions 0.2 0.4 0.6 0.8

Reads ``baseline_unet/splits/train.txt``, stratifies by ``(tier, has_spalling)``
to preserve class distribution, and writes subset files like
``baseline_unet/splits/train_20.txt``.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from baseline_unet import config as C
from baseline_unet.dataset import (
    has_spalling_from_mask,
    mask_path,
    read_mask_rgb,
    read_split_file,
)
from baseline_unet.splits import SPLIT_FILES

SEED = 42


def _bucket_files(files: List[str]) -> Dict[Tuple[str, bool], List[str]]:
    """Group files by (difficulty_tier, has_spalling)."""
    buckets: Dict[Tuple[str, bool], List[str]] = defaultdict(list)
    for rel in files:
        tier = rel.split("/", 1)[0]
        m = read_mask_rgb(mask_path(C.DATA_ROOT, rel))
        sp = has_spalling_from_mask(m)
        buckets[(tier, sp)].append(rel)
    return dict(buckets)


def generate_subset(
    train_files: List[str], fraction: float, seed: int = SEED,
) -> List[str]:
    """Stratified subsample: sample ``fraction`` from each bucket."""
    rng = random.Random(seed)
    buckets = _bucket_files(train_files)

    subset: List[str] = []
    for key in sorted(buckets.keys()):
        items = list(buckets[key])
        rng.shuffle(items)
        n = max(1, round(len(items) * fraction))
        subset.extend(items[:n])

    rng.shuffle(subset)
    return subset


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified training subsets",
    )
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8],
        help="Fractions of training data to sample (default: 0.2 0.4 0.6 0.8)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
    )
    args = parser.parse_args()

    train_path = SPLIT_FILES["train"]
    train_files = read_split_file(train_path)
    print(f"[subset] Full training set: {len(train_files)} files")

    for frac in args.fractions:
        subset = generate_subset(train_files, frac, seed=args.seed)
        pct = int(round(frac * 100))
        out_path = C.SPLITS_DIR / f"train_{pct}.txt"
        with open(out_path, "w") as f:
            for line in subset:
                f.write(line + "\n")
        print(f"[subset] train_{pct}.txt: {len(subset)} files "
              f"(target ~{int(round(len(train_files) * frac))})")


if __name__ == "__main__":
    main()

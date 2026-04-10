"""Two-level stratified split: (difficulty x has_spalling) -> 80/10/10.

Run once to produce ``splits/train.txt|val.txt|test.txt``. Idempotent: if all
three files already exist, we skip generation.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import config as C
from .dataset import (
    decode_mask,
    has_spalling_from_mask,
    image_path,
    mask_path,
    read_mask_rgb,
)

SPLIT_FILES = {
    "train": C.SPLITS_DIR / "train.txt",
    "val": C.SPLITS_DIR / "val.txt",
    "test": C.SPLITS_DIR / "test.txt",
}


def _scan_all(root: Path) -> List[Tuple[str, bool]]:
    items: List[Tuple[str, bool]] = []
    for diff in C.DIFFICULTIES:
        img_dir = root / diff / "Images"
        imgs = sorted(img_dir.glob("*.jpg"))
        for p in imgs:
            rel = f"{diff}/{p.name}"
            m = read_mask_rgb(mask_path(root, rel))
            items.append((rel, has_spalling_from_mask(m)))
    return items


def _split_bucket(files: List[str], rng: random.Random) -> Tuple[List[str], List[str], List[str]]:
    n = len(files)
    files = list(files)
    rng.shuffle(files)
    if n == 0:
        return [], [], []
    if n == 1:
        return files[:1], [], []
    if n == 2:
        return files[:1], files[1:2], []
    if n == 3:
        return files[:1], files[1:2], files[2:3]
    if n < 10:
        # at least 1 val + 1 test
        return files[: n - 2], files[n - 2 : n - 1], files[n - 1 :]
    n_val = max(1, round(C.VAL_RATIO * n))
    n_test = max(1, round(C.TEST_RATIO * n))
    n_train = n - n_val - n_test
    return files[:n_train], files[n_train : n_train + n_val], files[n_train + n_val :]


def _save(lines: List[str], path: Path) -> None:
    with open(path, "w") as f:
        for ln in lines:
            f.write(ln + "\n")


def _describe_split(name: str, files: List[str], root: Path) -> None:
    n = len(files)
    n_crack_only = 0
    n_spalling = 0
    px = np.zeros(C.NUM_CLASSES, dtype=np.int64)
    overlap_total = 0
    fg_total = 0
    for rel in files:
        m = read_mask_rgb(mask_path(root, rel))
        label, ov = decode_mask(m)
        has_sp = bool((label == 2).any())
        has_cr = bool((label == 1).any())
        if has_sp:
            n_spalling += 1
        elif has_cr:
            n_crack_only += 1
        for c in range(C.NUM_CLASSES):
            px[c] += int((label == c).sum())
        fg_total += int((label > 0).sum())
        overlap_total += ov
    tot = px.sum()
    frac = px / max(tot, 1)
    ov_ratio = overlap_total / max(fg_total, 1)
    print(f"[split:{name}] n={n}  crack_only={n_crack_only}  spalling={n_spalling}")
    print(f"[split:{name}] pixel frac: bg={frac[0]:.4f} crack={frac[1]:.4f} spalling={frac[2]:.4f}")
    print(f"[split:{name}] mask overlap / foreground = {ov_ratio:.6f}")


def ensure_splits(force: bool = False) -> Dict[str, List[str]]:
    if not force and all(p.exists() for p in SPLIT_FILES.values()):
        return {k: [ln.strip() for ln in open(p).read().splitlines() if ln.strip()] for k, p in SPLIT_FILES.items()}

    rng = random.Random(C.SEED)
    items = _scan_all(C.DATA_ROOT)
    # buckets: (difficulty, has_spalling)
    buckets: Dict[Tuple[str, bool], List[str]] = {}
    for rel, sp in items:
        diff = rel.split("/", 1)[0]
        buckets.setdefault((diff, sp), []).append(rel)

    train: List[str] = []
    val: List[str] = []
    test: List[str] = []

    print("[split] bucket sizes and per-bucket (train, val, test):")
    for key in sorted(buckets.keys()):
        b_train, b_val, b_test = _split_bucket(buckets[key], rng)
        print(f"  bucket {key}: n={len(buckets[key])}  -> ({len(b_train)}, {len(b_val)}, {len(b_test)})")
        train += b_train; val += b_val; test += b_test

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)

    _save(train, SPLIT_FILES["train"])
    _save(val, SPLIT_FILES["val"])
    _save(test, SPLIT_FILES["test"])

    _describe_split("train", train, C.DATA_ROOT)
    _describe_split("val", val, C.DATA_ROOT)
    _describe_split("test", test, C.DATA_ROOT)

    return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="regenerate even if split files exist")
    args = p.parse_args()
    ensure_splits(force=args.force)

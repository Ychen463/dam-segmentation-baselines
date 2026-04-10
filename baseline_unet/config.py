"""Global config for the Step-0 U-Net baseline sanity check."""
from __future__ import annotations

import os
from pathlib import Path

# ----- Paths -----
# NOTE: the dataset folder name is literally "Damage Segmentaion" (sic) in the repo.
CODES_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = CODES_DIR / "Dataset" / "DamSegment" / "Damage Segmentaion"

PKG_DIR = Path(__file__).resolve().parent
SPLITS_DIR = PKG_DIR / "splits"
RUNS_DIR = PKG_DIR / "runs"
RUN_NAME = "unet_r34_ce_dice"
RUN_DIR = RUNS_DIR / RUN_NAME
SAMPLES_DIR = RUN_DIR / "samples"
SANITY_DIR = RUNS_DIR / "sanity"

for p in (SPLITS_DIR, RUN_DIR, SAMPLES_DIR, SANITY_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ----- Data -----
DIFFICULTIES = ["Easy", "Medium", "Hard"]
DIFF_PREFIX = {"Easy": "E", "Medium": "M", "Hard": "H"}
NUM_CLASSES = 3
CLASS_NAMES = ["background", "crack", "spalling"]

# ----- Training -----
IMG_SIZE = 320  # downscaled from 640 for MPS sanity speed
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
# Rough inverse-frequency fallback. Auto weights from compute_class_weights()
# are what actually gets used unless --use-manual-weights is passed.
CE_WEIGHTS = [0.2, 2.0, 3.0]
LOSS_CE_W = 0.5
LOSS_DICE_W = 0.5
SEED = 42

# ----- Device -----
# Set to "mps"; train.py will auto-fallback to CPU if MPS is unavailable.
DEVICE = "mps"

# ----- Split ratios (per bucket) -----
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ----- Misc -----
# Allow MPS op fallback to CPU for unsupported kernels.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

#!/usr/bin/env bash
# Round 5: M2-M5 — Three novel contributions on Mask2Former
# M2: Difficulty-aware loss weighting
# M3: clDice auxiliary loss
# M4: Difficulty + clDice
# M5: Full method (curriculum + difficulty + clDice)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== M2: Difficulty-aware loss weighting ====="
python -m baseline_mask2former.train --ablation M2

echo "===== M3: clDice auxiliary loss ====="
python -m baseline_mask2former.train --ablation M3

echo "===== M4: Difficulty + clDice ====="
python -m baseline_mask2former.train --ablation M4

echo "===== M5: Full method (curriculum + difficulty + clDice) ====="
python -m baseline_mask2former.train --ablation M5

echo "All M2-M5 experiments complete."

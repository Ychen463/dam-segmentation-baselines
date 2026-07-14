#!/usr/bin/env bash
# Round 8: Replace clDice with Skeleton Recall Loss (SRL)
# SRL (Kirchhoff et al., ECCV 2024) is lighter, multi-class friendly,
# and avoids online skeletonization overhead.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== P1v3: Plain + SRL (baseline for SRL) ====="
python -m full_method.train --ablation P1v3

echo "===== D1v3: Difficulty reweight + SRL ====="
python -m full_method.train --ablation D1v3

echo "===== F1v3: Full method (C2 + difficulty reweight + SRL) ====="
python -m full_method.train --ablation F1v3

echo "Round 8 (SRL) complete."

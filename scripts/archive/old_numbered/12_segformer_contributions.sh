#!/usr/bin/env bash
# Round 6: SegFormer M0-M5 equivalent ablation
# P0 = M0 (plain baseline) — already done
# C2 = M1 (curriculum only) — already done
# D0 = M2 (difficulty weighting)
# P1 = M3 (clDice only)
# D1 = M4 (difficulty + clDice)
# F1 = M5 (full method: C2 + difficulty + clDice)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== D0: Difficulty weighting only ====="
python -m full_method.train --ablation D0

echo "===== P1: clDice only ====="
python -m full_method.train --ablation P1

echo "===== D1: Difficulty + clDice ====="
python -m full_method.train --ablation D1

echo "===== F1: Full method (C2 + difficulty + clDice) ====="
python -m full_method.train --ablation F1

echo "All SegFormer contribution ablations complete."

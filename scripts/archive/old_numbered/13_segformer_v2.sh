#!/usr/bin/env bash
# Round 7: SegFormer v2 — difficulty via sampling (not loss reweight)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== D0v2: Difficulty sampling only ====="
python -m full_method.train --ablation D0v2

echo "===== D1v2: Difficulty sampling + clDice ====="
python -m full_method.train --ablation D1v2

echo "===== F1v2: Full method (C2 + diff sampling + clDice) ====="
python -m full_method.train --ablation F1v2

echo "Round 7 complete."

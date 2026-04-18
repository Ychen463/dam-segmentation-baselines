#!/usr/bin/env bash
# Round 10: DSCformerDam + Snake Auxiliary Loss
# H series = G series + direct snake branch supervision (focal BCE + Dice)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== H0: DSCformer + snake aux (no SRL, no curriculum) ====="
python -m full_method.train --ablation H0

echo "===== H1: DSCformer + snake aux + SRL ====="
python -m full_method.train --ablation H1

echo "===== H2: DSCformer full + snake aux + SRL ====="
python -m full_method.train --ablation H2

echo "Round 10 (DSCformer + Snake Aux) complete."

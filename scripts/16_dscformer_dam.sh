#!/usr/bin/env bash
# Round 9: DSCformerDam — SegFormer + Dynamic Snake Convolution crack branch
# DSConv (ICCV 2023) adds crack-specific geometric inductive bias via
# cumulative offset learning that follows tubular crack geometry.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== G0: DSCformer plain (no curriculum, no topo loss) ====="
python -m full_method.train --ablation G0

echo "===== G1: DSCformer + SRL ====="
python -m full_method.train --ablation G1

echo "===== G2: DSCformer full (C2 + difficulty reweight + SRL) ====="
python -m full_method.train --ablation G2

echo "Round 9 (DSCformerDam) complete."

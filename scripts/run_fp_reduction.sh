#!/usr/bin/env bash
# Run FP-Reduction experiments (Crack IoU improvement via precision).
# FP1: Precision-oriented Tversky (alpha=0.6, beta=0.4)
# FP2: Balanced Tversky (alpha=0.5, beta=0.5)
# FP3: Precision Tversky + crack-only boundary Dice
#
# Usage:
#   bash scripts/run_fp_reduction.sh          # run all
#   bash scripts/run_fp_reduction.sh FP1      # run specific
set -euo pipefail

cd "$(dirname "$0")/.."

PRESETS="${@:-FP1 FP2 FP3}"

for preset in $PRESETS; do
    echo ""
    echo "============================================================"
    echo " Running preset: $preset"
    echo "============================================================"

    python -m full_method.train \
        --ablation "$preset" \
        --epochs 100 \
        2>&1 | tee "results/fp_${preset}.log"
done

echo ""
echo "All FP-reduction experiments complete."

#!/usr/bin/env bash
# Run tier-leveraging experiments (T1-T6).
# All build on DKD10 baseline (DSCformerDam + DTKD, mIoU_fg=72.35%).
#
# Usage:
#   bash scripts/run_tier_leverage.sh          # run all
#   bash scripts/run_tier_leverage.sh T1       # run specific preset
#   bash scripts/run_tier_leverage.sh T1 T4    # run multiple
set -euo pipefail

cd "$(dirname "$0")/.."

PRESETS="${@:-T1 T2 T3 T4 T5 T6}"

for preset in $PRESETS; do
    echo ""
    echo "============================================================"
    echo " Running preset: $preset"
    echo "============================================================"
    python -m full_method.train \
        --ablation "$preset" \
        --epochs 100 \
        2>&1 | tee "results/tier_${preset}.log"
done

echo ""
echo "All tier-leverage experiments complete."

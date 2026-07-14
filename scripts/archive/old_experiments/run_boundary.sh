#!/usr/bin/env bash
# Run Boundary-Privileged DTKD experiments (BR1-BR6).
# BR1-BR4, BR6, BR6b: 100 epochs from scratch with DKD10 base config.
# BR5: 30 epochs Phase 2, resume from DKD10 checkpoint.
#
# Usage:
#   bash scripts/run_boundary.sh              # run all
#   bash scripts/run_boundary.sh BR1          # run specific preset
#   bash scripts/run_boundary.sh BR6 BR6b     # run multiple
set -euo pipefail

cd "$(dirname "$0")/.."

PRESETS="${@:-BR1 BR1b BR2 BR3 BR4 BR5 BR6 BR6b}"

for preset in $PRESETS; do
    echo ""
    echo "============================================================"
    echo " Running preset: $preset"
    echo "============================================================"

    if [ "$preset" = "BR5" ]; then
        # Phase 2: resume from DKD10, 30 additional epochs (total 130)
        python -m full_method.train \
            --ablation "$preset" \
            --resume runs/dkd10_no_srl/last.pt \
            --epochs 130 \
            2>&1 | tee "results/boundary_${preset}.log"
    else
        python -m full_method.train \
            --ablation "$preset" \
            --epochs 100 \
            2>&1 | tee "results/boundary_${preset}.log"
    fi
done

echo ""
echo "All boundary-privileged KD experiments complete."

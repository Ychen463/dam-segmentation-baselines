#!/usr/bin/env bash
# Run Width-Aware Boundary Loss + Hard-Tier Fine-Tuning experiments.
#
# WA1: WABL only (DKD2 config + WABL loss)
# WA2: WABL + balanced Tversky (FP2 + WABL)
# HF1: Hard-heavy sampling + precision Tversky (resume from DKD2)
# HF2: Hard-heavy + WABL + balanced Tversky (resume from DKD2)
#
# Usage:
#   bash scripts/run_wabl_htft.sh          # run all
#   bash scripts/run_wabl_htft.sh WA1      # run specific
#   bash scripts/run_wabl_htft.sh HF1      # run HF1 (resumes from DKD2)
set -euo pipefail

cd "$(dirname "$0")/.."

PRESETS="${@:-WA1 WA2 HF1 HF2}"

for preset in $PRESETS; do
    echo ""
    echo "============================================================"
    echo " Running preset: $preset"
    echo "============================================================"

    # HF presets resume from DKD2 checkpoint
    if [[ "$preset" == HF* ]]; then
        echo " [HTFT] Resuming from dual_kd_classaware_DKD2/best.pt"
        python -m full_method.train \
            --ablation "$preset" \
            --epochs 130 \
            --resume full_method/runs/dual_kd_classaware_DKD2/best.pt \
            2>&1 | tee "results/${preset}.log"
    else
        python -m full_method.train \
            --ablation "$preset" \
            --epochs 100 \
            2>&1 | tee "results/${preset}.log"
    fi
done

echo ""
echo "All WABL/HTFT experiments complete."

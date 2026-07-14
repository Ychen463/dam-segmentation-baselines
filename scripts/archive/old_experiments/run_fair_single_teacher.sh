#!/usr/bin/env bash
# Fair single-teacher ablation: all students identical (no SRL), only teacher config varies.
# 5 conditions × 1 seed (42) = 5 runs (~100 epochs each)
# Reference: G0 (no KD), DKD10 (dual class-conditional, already trained)

set -euo pipefail
cd "$(dirname "$0")/.."

SEED=42
EPOCHS=100

CONFIGS=(
    "G0|dscformer_plain_G0"                          # no KD baseline
    "KD_T1_fair|kd_t1only_fair"                      # T1 only (G1)
    "KD_T2_fair|kd_t2only_fair"                      # T2 only (SAM2)
    "KD_DUAL_equal|kd_dual_equal_fair"               # dual equal weight
    "KD_DUAL_classaware|kd_dual_classaware_fair"     # dual class-conditional (= row c)
)

mkdir -p logs

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r ABLATION NAME <<< "$entry"
    RUNDIR="full_method/runs/${NAME}"
    if [[ -f "${RUNDIR}/best.pt" ]]; then
        echo "[${ABLATION}] Already trained, skipping."
        continue
    fi
    echo "=== Training ${ABLATION} (${NAME}) ==="
    python -m full_method.train \
        --ablation "$ABLATION" \
        --name "$NAME" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        2>&1 | tee "logs/${NAME}.log"
done

echo ""
echo "=== All fair single-teacher runs complete ==="
echo "Run evaluation with: python -m full_method.eval --runs kd_t1only_fair kd_t2only_fair kd_dual_equal_fair kd_dual_classaware_fair dscformer_plain_G0"

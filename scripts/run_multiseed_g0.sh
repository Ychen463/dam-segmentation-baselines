#!/usr/bin/env bash
# Multi-seed training for DSConv-only (G0) to disentangle
# variance reduction: DSConv vs DTKD.
#
# Trains G0 with seeds 42, 123, 2024, then evaluates per-image.
# Run on RunPod:
#   cd /workspace/Codes && bash scripts/run_multiseed_g0.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 123 2024)
EVAL_ONLY=false
if [[ "${1:-}" == "--eval-only" ]]; then
    EVAL_ONLY=true
fi

# ============================================================================
# Step 1: Train G0 with 3 seeds
# ============================================================================

if [[ "$EVAL_ONLY" == "false" ]]; then

mkdir -p logs results/multiseed
echo "=== Multi-seed G0 (DSConv-only) training ==="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  G0 Seed: $SEED"
    echo "=========================================="

    RUN_NAME="dscformer_plain_G0_seed${SEED}"
    if [[ -f "full_method/runs/${RUN_NAME}/best.pt" ]]; then
        echo "[seed=$SEED] G0 already trained, skipping."
    else
        echo "[seed=$SEED] Training DSConv-only (G0) ..."
        python -m full_method.train \
            --ablation G0 \
            --name "$RUN_NAME" \
            --seed "$SEED" \
            2>&1 | tee "logs/G0_seed${SEED}.log"
    fi
done

fi  # end if not eval-only

# ============================================================================
# Step 2: Per-image evaluation
# ============================================================================

echo ""
echo "=== Per-image evaluation ==="

for SEED in "${SEEDS[@]}"; do
    MODEL="dscformer_plain_G0_seed${SEED}"
    CKPT="full_method/runs/${MODEL}/best.pt"
    if [[ ! -f "$CKPT" ]]; then
        echo "[WARN] Checkpoint not found: $CKPT — skipping"
        continue
    fi
    echo "[eval] $MODEL ..."
    python -m shared_eval.eval_all \
        --model "$MODEL" \
        --split test \
        --per-tier \
        --per-image \
        --output-dir results/multiseed/ \
        2>&1 || echo "[WARN] eval failed for $MODEL"
done

# ============================================================================
# Step 3: Aggregate (including G0)
# ============================================================================

echo ""
echo "=== Aggregating multi-seed results (with G0) ==="
python scripts/aggregate_multiseed.py \
    --results-dir results/multiseed/ \
    --seeds 42 123 2024 \
    --output results/multiseed/stability_summary.json

echo ""
echo "=== Done ==="
echo "Copy back:"
echo "  results/multiseed/stability_summary.json"
echo "  results/multiseed/dscformer_plain_G0_seed*_test.json"

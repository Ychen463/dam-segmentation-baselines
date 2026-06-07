#!/usr/bin/env bash
# Multi-seed training + evaluation for statistical significance.
#
# Trains 3 seeds x 3 key models:
#   - SegFormer-B2 baseline (P0)
#   - DSCFormer + SRL (G1)
#   - DSCFormer + DTKD (DKD2)
#
# Then runs per-image evaluation and paired bootstrap tests.
#
# Usage (on RunPod):
#   bash scripts/run_multiseed.sh          # full run
#   bash scripts/run_multiseed.sh --eval-only  # skip training, just eval

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 123 2024)
EVAL_ONLY=false
if [[ "${1:-}" == "--eval-only" ]]; then
    EVAL_ONLY=true
fi

# ============================================================================
# Step 1: Train each model with 3 seeds
# ============================================================================

if [[ "$EVAL_ONLY" == "false" ]]; then

mkdir -p logs results/multiseed
echo "=== Multi-seed training ==="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Seed: $SEED"
    echo "=========================================="

    # (a) SegFormer-B2 baseline (P0)
    if [[ -f "full_method/runs/plain_segformer_P0_seed${SEED}/best.pt" ]]; then
        echo "[seed=$SEED] P0 already trained, skipping."
    else
        echo "[seed=$SEED] Training SegFormer-B2 (P0) ..."
        python -m full_method.train \
            --ablation P0 \
            --name "plain_segformer_P0_seed${SEED}" \
            --seed "$SEED" \
            2>&1 | tee "logs/P0_seed${SEED}.log"
    fi

    # (b) DSCFormer + SRL (G1)
    if [[ -f "full_method/runs/dscformer_srl_G1_seed${SEED}/best.pt" ]]; then
        echo "[seed=$SEED] G1 already trained, skipping."
    else
        echo "[seed=$SEED] Training DSCFormer+SRL (G1) ..."
        python -m full_method.train \
            --ablation G1 \
            --name "dscformer_srl_G1_seed${SEED}" \
            --seed "$SEED" \
            2>&1 | tee "logs/G1_seed${SEED}.log"
    fi

    # (c) DSCFormer + DTKD (DKD2)
    if [[ -f "full_method/runs/dual_kd_classaware_DKD2_seed${SEED}/best.pt" ]]; then
        echo "[seed=$SEED] DKD2 already trained, skipping."
    else
        echo "[seed=$SEED] Training DSCFormer+DTKD (DKD2) ..."
        python -m full_method.train \
            --ablation DKD2 \
            --name "dual_kd_classaware_DKD2_seed${SEED}" \
            --seed "$SEED" \
            2>&1 | tee "logs/DKD2_seed${SEED}.log"
    fi

done

fi  # end if not eval-only

# ============================================================================
# Step 2: Register multi-seed models and run per-image evaluation
# ============================================================================

echo ""
echo "=== Per-image evaluation ==="

MODELS_TO_EVAL=""
for SEED in "${SEEDS[@]}"; do
    for PRESET_NAME in "plain_segformer_P0" "dscformer_srl_G1" "dual_kd_classaware_DKD2"; do
        MODEL="${PRESET_NAME}_seed${SEED}"
        # Check if checkpoint exists
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
done

# ============================================================================
# Step 3: Aggregate multi-seed results (mean +/- std)
# ============================================================================

echo ""
echo "=== Aggregating multi-seed results ==="
python scripts/aggregate_multiseed.py \
    --results-dir results/multiseed/ \
    --seeds 42 123 2024 \
    --output results/multiseed/stability_summary.json

# ============================================================================
# Step 4: Run paired bootstrap tests (DSCFormer+DTKD vs SegFormer-B2)
# ============================================================================

# Also evaluate the original seed=42 models with per-tier
echo ""
echo "=== Per-tier eval for original models ==="
for ORIG_MODEL in "dscformer_srl_G1" "dual_kd_classaware_DKD2"; do
    CKPT="full_method/runs/${ORIG_MODEL}/best.pt"
    if [[ -f "$CKPT" ]]; then
        echo "[eval] ${ORIG_MODEL} (per-tier) ..."
        python -m shared_eval.eval_all \
            --model "$ORIG_MODEL" \
            --split test \
            --per-tier \
            --output-dir results/ \
            2>&1 || echo "[WARN] eval failed for $ORIG_MODEL"
    fi
done

echo ""
echo "=== Paired bootstrap significance tests ==="

# Use seed=42 per-image CSVs for the bootstrap test (main reported results)
# Compare DKD2 vs P0 (the key claim: DTKD > SegFormer baseline)
python -m shared_eval.stats_significance \
    --results-dir results/multiseed/ \
    --split test \
    --model-a dual_kd_classaware_DKD2_seed42 \
    --model-b plain_segformer_P0_seed42 \
    --output-dir results/multiseed/

# Compare DKD2 vs G1 (DTKD adds value over DSCFormer+SRL)
python -m shared_eval.stats_significance \
    --results-dir results/multiseed/ \
    --split test \
    --model-a dual_kd_classaware_DKD2_seed42 \
    --model-b dscformer_srl_G1_seed42 \
    --output-dir results/multiseed/

echo ""
echo "=== Done. Check results/multiseed/ ==="

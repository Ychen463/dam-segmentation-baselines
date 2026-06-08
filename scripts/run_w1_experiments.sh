#!/usr/bin/env bash
# Run all W1 experiments with the FINAL model (DKD10, no SRL).
#
# This script replaces old DKD2-based results with DKD10-based ones:
#   1. Per-tier evaluation of DKD10 (eval only)
#   2. Statistical significance: G0 vs DKD10 (per-image eval + Wilcoxon)
#   3. Cross-dataset S2DS evaluation of DKD10 (eval only)
#   4. Multi-seed DKD10 training (seeds 123, 2024; seed 42 = existing)
#   5. DKD14 (equal-weight ablation) training + eval
#
# Usage (on RunPod):
#   bash scripts/run_w1_experiments.sh           # full run
#   bash scripts/run_w1_experiments.sh --eval-only  # skip training, just eval

set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_ONLY=false
if [[ "${1:-}" == "--eval-only" ]]; then
    EVAL_ONLY=true
fi

mkdir -p logs results/w1

echo "============================================"
echo "  W1 Experiments: DKD10 as Final Model"
echo "============================================"

# ============================================================================
# 1. Per-tier evaluation of DKD10
# ============================================================================

echo ""
echo "=== [1/5] Per-tier evaluation of DKD10 ==="

CKPT_DKD10="full_method/runs/dkd10_no_srl/best.pt"
if [[ ! -f "$CKPT_DKD10" ]]; then
    echo "[ERROR] DKD10 checkpoint not found at $CKPT_DKD10"
    echo "  Train DKD10 first: bash scripts/run_dkd_step.sh DKD10"
    exit 1
fi

python -m shared_eval.eval_all \
    --model dkd10_no_srl \
    --split test \
    --per-tier \
    --output-dir results/w1/ \
    2>&1 | tee logs/w1_pertier.log || echo "[WARN] per-tier eval failed"

# ============================================================================
# 2. Statistical significance: G0 vs DKD10 (per-image + Wilcoxon)
# ============================================================================

echo ""
echo "=== [2/5] Per-image eval for statistical tests ==="

# Per-image eval for G0 and DKD10
for MODEL in dscformer_plain_G0 dkd10_no_srl; do
    echo "[eval] $MODEL (per-image) ..."
    python -m shared_eval.eval_all \
        --model "$MODEL" \
        --split test \
        --per-image \
        --output-dir results/w1/ \
        2>&1 || echo "[WARN] per-image eval failed for $MODEL"
done

echo ""
echo "=== Wilcoxon signed-rank tests ==="

# DKD10 vs G0 (DSConv-only baseline)
python -m shared_eval.stats_significance \
    --results-dir results/w1/ \
    --split test \
    --model-a dkd10_no_srl \
    --model-b dscformer_plain_G0 \
    --output-dir results/w1/ \
    2>&1 | tee logs/w1_stats.log || echo "[WARN] stats test failed"

# ============================================================================
# 3. Cross-dataset S2DS evaluation of DKD10
# ============================================================================

echo ""
echo "=== [3/5] Cross-dataset S2DS evaluation ==="

python scripts/eval_cross_dataset.py \
    --model dkd10_no_srl \
    --tta \
    --per-image \
    2>&1 | tee logs/w1_crossdataset.log || echo "[WARN] cross-dataset eval failed"

# ============================================================================
# 4. Multi-seed DKD10 (seeds 42, 123, 2024)
# ============================================================================

echo ""
echo "=== [4/5] Multi-seed DKD10 ==="

SEEDS=(42 123 2024)

if [[ "$EVAL_ONLY" == "false" ]]; then
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="dkd10_no_srl_seed${SEED}"

        # seed 42 = the original DKD10 run; just symlink/copy if needed
        if [[ "$SEED" == "42" ]]; then
            if [[ -f "full_method/runs/dkd10_no_srl/best.pt" ]] && [[ ! -d "full_method/runs/${RUN_NAME}" ]]; then
                echo "[seed=42] Symlinking original DKD10 run ..."
                ln -s "$(pwd)/full_method/runs/dkd10_no_srl" "full_method/runs/${RUN_NAME}"
            fi
            echo "[seed=42] Using original DKD10 run."
            continue
        fi

        if [[ -f "full_method/runs/${RUN_NAME}/best.pt" ]]; then
            echo "[seed=$SEED] DKD10 already trained, skipping."
            continue
        fi

        echo "[seed=$SEED] Training DKD10 ..."
        python -m full_method.train \
            --ablation DKD10 \
            --name "$RUN_NAME" \
            --seed "$SEED" \
            2>&1 | tee "logs/DKD10_seed${SEED}.log"
    done
fi

# Eval all seeds
for SEED in "${SEEDS[@]}"; do
    RUN_NAME="dkd10_no_srl_seed${SEED}"
    CKPT="full_method/runs/${RUN_NAME}/best.pt"
    if [[ ! -f "$CKPT" ]]; then
        echo "[WARN] Checkpoint not found: $CKPT — skipping"
        continue
    fi
    echo "[eval] $RUN_NAME (per-image) ..."
    python -m shared_eval.eval_all \
        --model "$RUN_NAME" \
        --split test \
        --per-image \
        --output-dir results/w1/ \
        2>&1 || echo "[WARN] eval failed for $RUN_NAME"
done

# Aggregate multi-seed results
echo ""
echo "=== Aggregating multi-seed DKD10 results ==="
python scripts/aggregate_multiseed.py \
    --results-dir results/w1/ \
    --seeds 42 123 2024 \
    --models dkd10_no_srl \
    --output results/w1/dkd10_multiseed_summary.json \
    2>&1 || echo "[WARN] aggregation failed"

# ============================================================================
# 5. DKD14 (equal-weight ablation) train + eval
# ============================================================================

echo ""
echo "=== [5/5] DKD14 (equal-weight, no class-conditional) ==="

if [[ "$EVAL_ONLY" == "false" ]]; then
    if [[ -f "full_method/runs/dkd14_equal_weight/best.pt" ]]; then
        echo "[DKD14] Already trained, skipping."
    else
        echo "[DKD14] Training ..."
        python -m full_method.train \
            --ablation DKD14 \
            --name dkd14_equal_weight \
            2>&1 | tee logs/DKD14.log
    fi
fi

CKPT_DKD14="full_method/runs/dkd14_equal_weight/best.pt"
if [[ -f "$CKPT_DKD14" ]]; then
    echo "[DKD14] Evaluating (plain + TTA) ..."
    python -m full_method.eval_tta \
        --run dkd14_equal_weight \
        --ablation DKD14 \
        --no-tta \
        2>&1 || echo "[WARN] DKD14 plain eval failed"

    python -m full_method.eval_tta \
        --run dkd14_equal_weight \
        --ablation DKD14 \
        2>&1 || echo "[WARN] DKD14 TTA eval failed"
else
    echo "[WARN] DKD14 checkpoint not found — skipping eval"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================"
echo "  W1 Experiments Complete"
echo "============================================"
echo ""
echo "Results saved to results/w1/"
echo ""
echo "Training runs needed: $(if $EVAL_ONLY; then echo 'SKIPPED (--eval-only)'; else echo '2 new (DKD10 seed123, seed2024) + 1 new (DKD14)'; fi)"
echo ""
echo "Next steps:"
echo "  1. Check results/w1/ for per-tier, per-image, stats, cross-dataset outputs"
echo "  2. Update paper tables with DKD10 numbers"
echo "  3. git add + commit"

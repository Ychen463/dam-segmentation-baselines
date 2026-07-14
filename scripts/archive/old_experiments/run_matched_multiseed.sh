#!/bin/bash
# Fully-matched multi-seed DTKD mechanism isolation experiment
#
# Purpose: Determine whether heterogeneous teachers produce a NET GAIN
#          over the no-KD baseline under fully matched conditions.
#
# Design: 5 conditions × 3 seeds = 15 training runs
#   1. RERUN_NOKD:    No KD (DSConv-only baseline)
#   2. RERUN_T1ONLY:  T1-only KD (w_T2=0)
#   3. RERUN_DUP_T1:  Duplicated T1 (same ckpt for both teachers)
#   4. RERUN_HETERO:  Heterogeneous T1+T2, equal weights
#   5. RERUN_HETERO_CC: Heterogeneous T1+T2, class-conditional weights (full model)
#
# All conditions share:
#   - Same T1 checkpoint: runs/dscformer_srl_G1/best.pt
#   - Same T2 checkpoint: runs/sam_lora_srl_SAM2/best.pt (where applicable)
#   - Same α=0.5, τ=4.0
#   - Same architecture (DSCformerDam, SegFormer-B2 + DSConv k=9 ch=64)
#   - Same augmentation, LR, epochs, etc.
#   - Only variable: teacher configuration + training seed
#
# Usage on RunPod:
#   cd /workspace/dam-segmentation-baselines
#   bash scripts/run_matched_multiseed.sh
#
# Expected time: ~6h on A40 (15 × ~25min)

set -e

SEEDS=(42 123 2024)
PRESETS=(RERUN_NOKD RERUN_T1ONLY RERUN_DUP_T1 RERUN_HETERO RERUN_HETERO_CC)
LABELS=(
    "No-KD baseline"
    "T1-only KD"
    "Duplicated T1"
    "Hetero equal-wt"
    "Hetero class-cond (full)"
)

TOTAL=$((${#PRESETS[@]} * ${#SEEDS[@]}))
COUNT=0

echo "============================================================"
echo "  Matched Multi-Seed DTKD Mechanism Isolation"
echo "============================================================"
echo "  Conditions: ${#PRESETS[@]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Total runs: $TOTAL"
echo "============================================================"
echo ""

for i in "${!PRESETS[@]}"; do
    p="${PRESETS[$i]}"
    l="${LABELS[$i]}"
    for s in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        RUN_NAME="${p,,}_s${s}"  # lowercase preset + seed suffix
        echo "[$COUNT/$TOTAL] ${l} (seed=$s) → ${RUN_NAME}"
        python -m full_method.train \
            --ablation "$p" \
            --seed "$s" \
            --name "${RUN_NAME}"
        echo "  Done."
        echo ""
    done
done

# Evaluate all runs
echo "============================================================"
echo "  Evaluating all runs..."
echo "============================================================"
python scripts/eval_matched_multiseed.py

echo ""
echo "Done! Results saved to results/matched_multiseed.json"

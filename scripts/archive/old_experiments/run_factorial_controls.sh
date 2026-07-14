#!/usr/bin/env bash
# ===========================================================================
# Factorial ablation + control experiments for DTKD claims.
#
# Experiments:
#   1. 2x2 factorial: SegFormer+DTKD (no DSConv) — missing cell
#   2. Teacher ensemble upper bounds (no training needed)
#   3. Duplicated-teacher control (same T1 used twice)
#   4. Same-architecture different-seed teacher ensemble
#   5. Label smoothing control (soft targets without teachers)
#
# Prerequisites:
#   - Teacher checkpoints exist:
#       full_method/runs/dscformer_srl_G1/best.pt
#       full_method/runs/sam_lora_srl_SAM2/best.pt
#   - For CTRL_SAME_ARCH: need dscformer_srl_G1_seed123/best.pt
#     (train with: python -m full_method.train --ablation G1
#                  --name dscformer_srl_G1_seed123 --seed 123)
#
# Usage (on RunPod):
#   bash scripts/run_factorial_controls.sh              # full run
#   bash scripts/run_factorial_controls.sh --eval-only   # skip training
#   bash scripts/run_factorial_controls.sh --ensemble-only  # only ensemble eval
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_ONLY=false
ENSEMBLE_ONLY=false
if [[ "${1:-}" == "--eval-only" ]]; then
    EVAL_ONLY=true
elif [[ "${1:-}" == "--ensemble-only" ]]; then
    ENSEMBLE_ONLY=true
fi

mkdir -p logs results/factorial

# ===========================================================================
# Part 0: Teacher ensemble upper bounds (no training needed)
# ===========================================================================

echo ""
echo "============================================"
echo "  Part 0: Teacher Ensemble Upper Bounds"
echo "============================================"

python scripts/eval_teacher_ensemble.py \
    --device cuda \
    --split test \
    --output results/factorial/teacher_ensemble.json \
    2>&1 | tee logs/teacher_ensemble.log

# ===========================================================================
# Part 1: Same-arch teacher (need seed-123 G1 if not already trained)
# ===========================================================================

if [[ "$ENSEMBLE_ONLY" == "false" ]]; then

echo ""
echo "============================================"
echo "  Part 1: Ensure seed-123 DSCFormer+SRL teacher exists"
echo "============================================"

if [[ -f "full_method/runs/dscformer_srl_G1_seed123/best.pt" ]]; then
    echo "[skip] dscformer_srl_G1_seed123 already trained."
else
    echo "[train] Training DSCFormer+SRL with seed=123 for same-arch control..."
    python -m full_method.train \
        --ablation G1 \
        --name dscformer_srl_G1_seed123 \
        --seed 123 \
        2>&1 | tee logs/G1_seed123.log
fi

# Same-arch teacher ensemble eval
echo ""
echo "  Same-architecture teacher ensemble (seed42 + seed123)..."
python scripts/eval_teacher_ensemble.py \
    --device cuda \
    --split test \
    --t1-ckpt runs/dscformer_srl_G1/best.pt \
    --t2-ckpt runs/dscformer_srl_G1_seed123/best.pt \
    --t1-type dscformer \
    --t2-type dscformer \
    --output results/factorial/same_arch_ensemble.json \
    2>&1 | tee logs/same_arch_ensemble.log

# ===========================================================================
# Part 2: Training experiments
# ===========================================================================

if [[ "$EVAL_ONLY" == "false" ]]; then

echo ""
echo "============================================"
echo "  Part 2: Training Factorial & Controls"
echo "============================================"

# --- Exp 1: SegFormer + DTKD (no DSConv) — 2x2 missing cell ---
EXP="segformer_dtkd_FACT1"
if [[ -f "full_method/runs/${EXP}/best.pt" ]]; then
    echo "[skip] ${EXP} already trained."
else
    echo "[train] SegFormer + DTKD (no DSConv) ..."
    python -m full_method.train \
        --ablation FACT1 \
        --name "${EXP}" \
        2>&1 | tee "logs/${EXP}.log"
fi

# --- Exp 2: Duplicated Teacher 1 control ---
EXP="dscformer_dup_t1_CTRL"
if [[ -f "full_method/runs/${EXP}/best.pt" ]]; then
    echo "[skip] ${EXP} already trained."
else
    echo "[train] DSConv + duplicated T1 control ..."
    python -m full_method.train \
        --ablation CTRL_DUP_T1 \
        --name "${EXP}" \
        2>&1 | tee "logs/${EXP}.log"
fi

# --- Exp 3: Same-architecture different-seed teachers ---
EXP="dscformer_samearch_CTRL"
if [[ -f "full_method/runs/${EXP}/best.pt" ]]; then
    echo "[skip] ${EXP} already trained."
else
    if [[ -f "full_method/runs/dscformer_srl_G1_seed123/best.pt" ]]; then
        echo "[train] DSConv + same-arch teachers (seed42 + seed123) ..."
        python -m full_method.train \
            --ablation CTRL_SAME_ARCH \
            --name "${EXP}" \
            2>&1 | tee "logs/${EXP}.log"
    else
        echo "[WARN] Skipping CTRL_SAME_ARCH: seed-123 teacher not found."
    fi
fi

# --- Exp 4: Label smoothing control ---
EXP="dscformer_labelsmooth_CTRL"
if [[ -f "full_method/runs/${EXP}/best.pt" ]]; then
    echo "[skip] ${EXP} already trained."
else
    echo "[train] DSConv + label smoothing (epsilon=0.1) control ..."
    python -m full_method.train \
        --ablation CTRL_LABEL_SMOOTH \
        --name "${EXP}" \
        2>&1 | tee "logs/${EXP}.log"
fi

fi  # end if not eval-only

# ===========================================================================
# Part 3: Evaluate all trained models
# ===========================================================================

echo ""
echo "============================================"
echo "  Part 3: Evaluation"
echo "============================================"

MODELS=(
    "segformer_dtkd_FACT1"
    "dscformer_dup_t1_CTRL"
    "dscformer_samearch_CTRL"
    "dscformer_labelsmooth_CTRL"
)

for MODEL in "${MODELS[@]}"; do
    CKPT="full_method/runs/${MODEL}/best.pt"
    if [[ -f "$CKPT" ]]; then
        echo "[eval] ${MODEL} ..."
        python -m shared_eval.eval_all \
            --model "$MODEL" \
            --split test \
            --per-tier \
            --per-image \
            --output-dir results/factorial/ \
            2>&1 || echo "[WARN] eval failed for $MODEL"
    else
        echo "[skip] ${MODEL}: checkpoint not found"
    fi
done

fi  # end if not ensemble-only

# ===========================================================================
# Summary
# ===========================================================================

echo ""
echo "============================================"
echo "  Done. Results in results/factorial/"
echo "============================================"
echo ""
echo "Key questions answered:"
echo "  1. 2x2 factorial: Does DTKD help without DSConv? -> segformer_dtkd_FACT1"
echo "  2. Ensemble upper bound: Does student match/exceed ensemble? -> teacher_ensemble.json"
echo "  3. Duplicated teacher: Is diversity needed? -> dscformer_dup_t1_CTRL"
echo "  4. Same-arch ensemble: Is architectural complementarity key? -> dscformer_samearch_CTRL"
echo "  5. Label smoothing: Is soft-target regularisation sufficient? -> dscformer_labelsmooth_CTRL"
echo ""
echo "Reference: DTKD student mIoU_fg=72.3, DSConv-only=70.4, SegFormer=70.3"

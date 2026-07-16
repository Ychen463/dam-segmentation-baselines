#!/usr/bin/env bash
# Multi-seed training on balanced group split for core method ranking.
#
# Trains 3 seeds each for:
#   - SegFormer-B2 (baseline)
#   - Teacher 1: DSConv+SRL (G1)
#   - DTKD Student (DKD2)
#
# DTKD requires teacher checkpoints. By default uses seed-0 teachers
# (existing runs). For full rigor, each DTKD seed uses the same teachers.
#
# Usage:
#   bash scripts/run_balanced_multiseed.sh              # all 3 models × 3 seeds
#   bash scripts/run_balanced_multiseed.sh segformer     # SegFormer only
#   bash scripts/run_balanced_multiseed.sh teacher1      # Teacher 1 only
#   bash scripts/run_balanced_multiseed.sh dtkd          # DTKD only

set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BS="${CODES_DIR}/baseline_unet/splits/balanced_group_split"
TR="${BS}/train.txt"
VA="${BS}/val.txt"

cd "$CODES_DIR"

# Verify split files
for f in "$TR" "$VA"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing: $f"
        exit 1
    fi
done

SEEDS=(42 123 7)

run_segformer() {
    echo ""
    echo "=== SegFormer-B2 multi-seed (balanced split) ==="
    for s in "${SEEDS[@]}"; do
        NAME="segformer_b2_bgsplit_s${s}"
        echo "--- SegFormer-B2 seed=${s} → ${NAME} ---"
        python -m baseline_segformer.train --preset A --epochs 100 \
            --train-split "$TR" --val-split "$VA" \
            --name "$NAME" --seed "$s"
    done
}

run_teacher1() {
    echo ""
    echo "=== Teacher 1 (DSConv+SRL G1) multi-seed (balanced split) ==="
    for s in "${SEEDS[@]}"; do
        NAME="dscformer_srl_G1_bgsplit_s${s}"
        echo "--- Teacher1 seed=${s} → ${NAME} ---"
        python -m full_method.train --ablation G1 \
            --train-split "$TR" --val-split "$VA" \
            --name "$NAME" --seed "$s"
    done
}

run_dtkd() {
    echo ""
    echo "=== DTKD Student (DKD2) multi-seed (balanced split) ==="

    # Use existing seed-0 teachers for all DTKD seeds
    T1_CKPT="${CODES_DIR}/full_method/runs/dscformer_srl_G1_bgsplit/best.pt"
    T2_CKPT="${CODES_DIR}/full_method/runs/sam_lora_srl_SAM2_bgsplit/best.pt"

    if [ ! -f "$T1_CKPT" ] || [ ! -f "$T2_CKPT" ]; then
        echo "[ERROR] Teacher checkpoints not found:"
        echo "  T1: $T1_CKPT"
        echo "  T2: $T2_CKPT"
        exit 1
    fi

    for s in "${SEEDS[@]}"; do
        NAME="dtkd_DKD2_bgsplit_s${s}"
        echo "--- DTKD seed=${s} → ${NAME} ---"
        python -m full_method.train --ablation DKD2 \
            --train-split "$TR" --val-split "$VA" \
            --name "$NAME" --seed "$s" \
            --kd-teacher-checkpoint "$T1_CKPT" \
            --kd-teacher2-checkpoint "$T2_CKPT"
    done
}

# Dispatch
PHASE="${1:-all}"

case "$PHASE" in
    segformer)  run_segformer ;;
    teacher1)   run_teacher1 ;;
    dtkd)       run_dtkd ;;
    all)
        run_segformer
        run_teacher1
        run_dtkd
        ;;
    *)
        echo "Usage: $0 {all|segformer|teacher1|dtkd}"
        exit 1
        ;;
esac

echo ""
echo "=== Multi-seed balanced split training complete ==="
echo ""
echo "Collect results with:"
echo "  python scripts/collect_multiseed_bgsplit.py"

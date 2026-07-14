#!/usr/bin/env bash
# Retrain all models on balanced group split for leakage-free final evaluation.
#
# All hyperparameters are LOCKED from original-split development.
# This is a ONE-TIME final evaluation — do NOT tune anything here.
#
# Usage:
#   bash scripts/run_balanced_split_retrain.sh          # full pipeline
#   bash scripts/run_balanced_split_retrain.sh phase1    # baselines only
#   bash scripts/run_balanced_split_retrain.sh phase2    # teachers only
#   bash scripts/run_balanced_split_retrain.sh phase3    # student only
#   bash scripts/run_balanced_split_retrain.sh eval      # evaluate only
set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BS="${CODES_DIR}/baseline_unet/splits/balanced_group_split"
TR="${BS}/train.txt"
VA="${BS}/val.txt"
TE="${BS}/test.txt"
SUFFIX="bgsplit"

cd "$CODES_DIR"

# Verify split files
for f in "$TR" "$VA" "$TE"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing: $f"
        echo "  Run: python scripts/build_balanced_group_split.py --extract-features"
        exit 1
    fi
done

echo "========================================================"
echo "  Balanced Group Split Retraining"
echo "========================================================"
echo "  train: $(wc -l < "$TR") samples"
echo "  val:   $(wc -l < "$VA") samples"
echo "  test:  $(wc -l < "$TE") samples"
echo "  suffix: ${SUFFIX}"
echo "========================================================"

PHASE="${1:-all}"

# ── Phase 1: Baselines ─────────────────────────────────────────────────

run_phase1() {
    echo ""
    echo "=== Phase 1: Baselines ==="

    echo "--- 1/4 SegFormer B2 (512, 100ep) ---"
    python -m baseline_segformer.train --preset A --epochs 100 \
        --train-split "$TR" --val-split "$VA" \
        --name "segformer_b2_plain_512_${SUFFIX}"

    echo "--- 2/4 Mask2Former (512, 100ep) ---"
    python -m baseline_mask2former.train --ablation M0 --epochs 100 \
        --train-split "$TR" --val-split "$VA" \
        --name "mask2former_plain_M0_${SUFFIX}"

    echo "--- 3/4 U-Net R34 (320, 30ep) ---"
    python -m baseline_unet.train --epochs 30 \
        --train-split "$TR" --val-split "$VA" \
        --name "unet_r34_ce_dice_${SUFFIX}"

    echo "--- 4/4 DeepLabV3+ R50 (512, 50ep) ---"
    python -m baseline_deeplab.train --preset A --epochs 50 \
        --train-split "$TR" --val-split "$VA" \
        --name "deeplabv3p_r50_512_${SUFFIX}"

    echo "=== Phase 1 complete ==="
}

# ── Phase 2: Teachers ──────────────────────────────────────────────────

run_phase2() {
    echo ""
    echo "=== Phase 2: Teachers ==="

    echo "--- T1: DSCFormer-SRL G1 (100ep) ---"
    python -m full_method.train --ablation G1 \
        --train-split "$TR" --val-split "$VA" \
        --name "dscformer_srl_G1_${SUFFIX}"

    echo "--- T2: SAM2-LoRA (50ep) ---"
    python -m full_method.train --ablation SAM2 \
        --train-split "$TR" --val-split "$VA" \
        --name "sam_lora_srl_SAM2_${SUFFIX}"

    echo "=== Phase 2 complete ==="
}

# ── Phase 3: Student (depends on Phase 2) ──────────────────────────────

run_phase3() {
    echo ""
    echo "=== Phase 3: Student ==="

    T1_CKPT="${CODES_DIR}/full_method/runs/dscformer_srl_G1_${SUFFIX}/best.pt"
    T2_CKPT="${CODES_DIR}/full_method/runs/sam_lora_srl_SAM2_${SUFFIX}/best.pt"

    if [ ! -f "$T1_CKPT" ] || [ ! -f "$T2_CKPT" ]; then
        echo "[ERROR] Teacher checkpoints not found. Run phase2 first."
        echo "  T1: $T1_CKPT"
        echo "  T2: $T2_CKPT"
        exit 1
    fi

    echo "--- DTKD DKD2 (100ep) ---"
    python -m full_method.train --ablation DKD2 \
        --train-split "$TR" --val-split "$VA" \
        --name "dual_kd_classaware_DKD2_${SUFFIX}" \
        --kd-teacher-checkpoint "$T1_CKPT" \
        --kd-teacher2-checkpoint "$T2_CKPT"

    echo "=== Phase 3 complete ==="
}

# ── Phase 4: DSConv-only baseline (G0, no SRL) ────────────────────────

run_phase_g0() {
    echo ""
    echo "=== DSConv-only baseline (G0) ==="

    echo "--- DSCFormer plain G0 (100ep) ---"
    python -m full_method.train --ablation G0 \
        --train-split "$TR" --val-split "$VA" \
        --name "dscformer_plain_G0_${SUFFIX}"
}

# ── Evaluate all models on balanced test set ───────────────────────────

run_eval() {
    echo ""
    echo "=== Final Evaluation on Balanced Group Split Test Set ==="

    python scripts/eval_balanced_split.py --test-split "$TE"

    echo "=== Evaluation complete ==="
}

# ── Dispatch ───────────────────────────────────────────────────────────

case "$PHASE" in
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    phase3) run_phase3 ;;
    g0)     run_phase_g0 ;;
    eval)   run_eval ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        run_phase_g0
        run_eval
        ;;
    *)
        echo "Usage: $0 {all|phase1|phase2|phase3|g0|eval}"
        exit 1
        ;;
esac

echo ""
echo "========================================================"
echo "  Done: $PHASE"
echo "========================================================"

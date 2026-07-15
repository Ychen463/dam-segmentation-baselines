#!/usr/bin/env bash
# Run Curriculum KD (CKD) experiments: T1-init + selective T2 rescue.
#
# Usage:
#   bash scripts/run_ckd_experiments.sh              # run all phases
#   bash scripts/run_ckd_experiments.sh diag          # diagnostics only
#   bash scripts/run_ckd_experiments.sh b0             # T1 baseline only
#   bash scripts/run_ckd_experiments.sh c0 c3          # specific experiments
#
# Prerequisites:
#   - T1 checkpoint: full_method/runs/dscformer_srl_G1/best.pt
#   - T2 checkpoint: full_method/runs/sam_lora_srl_SAM2/best.pt
#   - Balanced group split: baseline_unet/splits/balanced_group_split/{train,val,test}.txt
#
# Experiment matrix:
#   B0: T1 val metrics (no training, eval only)
#   C0: Continued training, GT-only (no KD)
#   C1: T1-only anchoring KD
#   C2: Dual-equal KD (original DTKD + T1-init)
#   C3: Selective T2 rescue (MAIN experiment)
#   C4: T2 rescue with lower KD weight
#
# Run order: diag -> b0 -> c0 -> c3 -> c2 -> c4 -> (c1 if C3>C0)

set -euo pipefail

SPLIT_DIR="baseline_unet/splits/balanced_group_split"
T1_CKPT="full_method/runs/dscformer_srl_G1/best.pt"
T2_CKPT="full_method/runs/sam_lora_srl_SAM2/best.pt"

# Verify prerequisites
for f in "$T1_CKPT" "$T2_CKPT" "$SPLIT_DIR/train.txt" "$SPLIT_DIR/val.txt"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Required file not found: $f"
        exit 1
    fi
done

# Common args for all CKD experiments (use balanced group split)
COMMON_ARGS="--train-split $SPLIT_DIR/train.txt --val-split $SPLIT_DIR/val.txt"

run_diag() {
    echo ""
    echo "=============================================="
    echo "  Step 0: Diagnostic Scripts"
    echo "=============================================="

    echo "--- Running teacher complementarity diagnosis ---"
    python scripts/teacher_complementarity_diagnosis.py

    echo ""
    echo "--- Running confidence routing diagnosis ---"
    python scripts/confidence_routing_diagnosis.py

    echo ""
    echo "[diag] Check results in $SPLIT_DIR/routing_diagnosis.json"
    echo "[diag] Review routing feasibility before proceeding."
}

run_b0() {
    echo ""
    echo "=============================================="
    echo "  B0: T1 Baseline (1-epoch eval)"
    echo "=============================================="
    # Run 1 epoch to get T1 val metrics (T1-init + 1 epoch at lr=1e-5)
    # Val metrics at epoch 1 serve as the T1 baseline reference
    python -m full_method.train \
        --ablation CKD_B0 \
        --epochs 1 \
        $COMMON_ARGS
}

run_experiment() {
    local PRESET="$1"
    local DESC="$2"

    echo ""
    echo "=============================================="
    echo "  $PRESET: $DESC"
    echo "=============================================="
    python -m full_method.train \
        --ablation "$PRESET" \
        $COMMON_ARGS

    # Report val metrics from the run
    local RUN_NAME
    RUN_NAME=$(python -c "
from full_method.config import ABLATION_PRESETS
print(ABLATION_PRESETS['$PRESET']['name'])
")
    local RUN_DIR="full_method/runs/$RUN_NAME"
    if [ -f "$RUN_DIR/best.pt" ]; then
        echo "[eval] $PRESET best.pt at $RUN_DIR/best.pt"
        echo "[eval] Val metrics from training log above."
    else
        echo "[warn] No best.pt for $PRESET at $RUN_DIR"
    fi
}

# Determine which phases to run
if [ $# -gt 0 ]; then
    PHASES=("$@")
else
    PHASES=(diag b0 c0 c3 c2 c4)
fi

for PHASE in "${PHASES[@]}"; do
    case "$PHASE" in
        diag)
            run_diag
            ;;
        b0)
            run_b0
            ;;
        c0)
            run_experiment CKD_C0 "Continued training, GT-only (no KD)"
            ;;
        c1)
            run_experiment CKD_C1 "T1-only anchoring KD"
            ;;
        c2)
            run_experiment CKD_C2 "Dual-equal KD (DTKD + T1-init)"
            ;;
        c3)
            run_experiment CKD_C3 "Selective T2 rescue (MAIN)"
            ;;
        c4)
            run_experiment CKD_C4 "T2 rescue, lower KD weight"
            ;;
        *)
            echo "[ERROR] Unknown phase: $PHASE"
            echo "  Valid phases: diag, b0, c0, c1, c2, c3, c4"
            exit 1
            ;;
    esac
done

echo ""
echo "=== CKD experiments complete ==="
echo ""
echo "Recommended analysis order:"
echo "  1. Compare C3 vs C0 (T2 rescue value)"
echo "  2. Compare C3 vs B0 (improvement over T1)"
echo "  3. If C3 > C0 and C3 > B0: run C1 for T1 anchoring comparison"
echo "  4. If C3 - C0 >= 0.3 mIoU_fg: proceed to 3-seed validation"

#!/usr/bin/env bash
# Run Group-Robust Training experiments (G0-G4).
#
# Usage:
#   bash scripts/run_group_robust.sh              # run all Phase 1 (g0, g0r, g2)
#   bash scripts/run_group_robust.sh g0            # specific experiment
#   bash scripts/run_group_robust.sh g0 g0r g2     # multiple experiments
#   bash scripts/run_group_robust.sh g1c g4 g3     # Phase 2 experiments
#
# Prerequisites:
#   - T1 checkpoint: full_method/runs/dscformer_srl_G1_bgsplit/best.pt
#   - Group assignments: baseline_unet/splits/balanced_group_split/group_assignments.json
#   - Balanced group split: baseline_unet/splits/balanced_group_split/{train,val}.txt
#
# Experiment matrix:
#   G0:  Shuffle baseline (= C0 + group metrics)
#   G0R: Equal-weight with replacement (isolate replacement effect)
#   G2:  Inverse-sqrt group weighting (primary candidate)
#   G1C: Capped group-uniform (Phase 2)
#   G4:  JTT two-stage (Phase 2)
#   G3:  GroupDRO (Phase 2)
#
# Run order: G0 → G0R → G2 → G1C → G4 → G3
# No test evaluation — val only.

set -euo pipefail

SPLIT_DIR="baseline_unet/splits/balanced_group_split"
T1_CKPT="full_method/runs/dscformer_srl_G1_bgsplit/best.pt"
GA_FILE="$SPLIT_DIR/group_assignments.json"

# Verify prerequisites
for f in "$T1_CKPT" "$GA_FILE" "$SPLIT_DIR/train.txt" "$SPLIT_DIR/val.txt"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Required file not found: $f"
        exit 1
    fi
done

COMMON_ARGS="--train-split $SPLIT_DIR/train.txt --val-split $SPLIT_DIR/val.txt"

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

    local RUN_NAME
    RUN_NAME=$(python -c "
from full_method.config import ABLATION_PRESETS
print(ABLATION_PRESETS['$PRESET']['name'])
")
    local RUN_DIR="full_method/runs/$RUN_NAME"
    if [ -f "$RUN_DIR/val_report.txt" ]; then
        echo ""
        echo "[result] $PRESET val report:"
        cat "$RUN_DIR/val_report.txt"
    fi
    if [ -f "$RUN_DIR/group_metrics.csv" ]; then
        echo ""
        echo "[result] $PRESET group metrics (last row):"
        tail -1 "$RUN_DIR/group_metrics.csv"
    fi
}

# Determine which phases to run
if [ $# -gt 0 ]; then
    PHASES=("$@")
else
    # Default: Phase 1 only
    PHASES=(g0 g0r g2)
fi

for PHASE in "${PHASES[@]}"; do
    case "$PHASE" in
        g0)
            run_experiment GR_G0 "Shuffle baseline (C0 + group metrics)"
            ;;
        g0r)
            run_experiment GR_G0R "Equal-weight with replacement"
            ;;
        g2)
            run_experiment GR_G2 "Inverse-sqrt group weighting"
            ;;
        g1c)
            run_experiment GR_G1C "Capped group-uniform"
            ;;
        g4)
            run_experiment GR_G4 "JTT two-stage"
            ;;
        g3)
            run_experiment GR_G3 "GroupDRO"
            ;;
        *)
            echo "[ERROR] Unknown phase: $PHASE"
            echo "  Valid phases: g0, g0r, g2, g1c, g4, g3"
            exit 1
            ;;
    esac
done

echo ""
echo "=== Group-Robust experiments complete ==="
echo ""
echo "Key comparisons:"
echo "  G0R vs G0:  effect of with-replacement sampling"
echo "  G2 vs G0R:  net effect of inverse-sqrt group weighting"
echo "  G1C vs G0R: net effect of capped uniform weighting"
echo "  G3 vs G0:   DRO value"
echo "  G4 vs G0:   JTT value"
echo ""
echo "Success criteria (pre-defined, val-based):"
echo "  mean mIoU_fg >= G0 - 0.3"
echo "  CVaR20 mIoU_fg >= G0 + 1.5 OR p10 mIoU_fg >= G0 + 2.0"
echo "  crack IoU/recall drop < 0.5 vs G0"

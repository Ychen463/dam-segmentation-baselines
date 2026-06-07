#!/usr/bin/env bash
# Train & evaluate DKD7/DKD8/DKD9 optimization experiments.
#
# DKD7: DKD2 + curriculum learning
# DKD8: DKD2 + G0 as Teacher 1 (higher ConnR teacher)
# DKD9: DKD2 + moderate augmentation
#
# Prerequisites (must already exist on RunPod):
#   full_method/runs/dscformer_srl_G1/best.pt      (Teacher 1 for DKD7/DKD9)
#   full_method/runs/dscformer_plain_G0/best.pt     (Teacher 1 for DKD8)
#   full_method/runs/sam_lora_srl_SAM2/best.pt      (Teacher 2 for all)
#
# Usage:
#   bash scripts/run_dkd789.sh            # train + eval all 3
#   bash scripts/run_dkd789.sh --eval-only # skip training, just eval

set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_ONLY=false
if [[ "${1:-}" == "--eval-only" ]]; then
    EVAL_ONLY=true
fi

PRESETS=(DKD7 DKD8 DKD9)
NAMES=(dkd7_curriculum dkd8_teacher_g0 dkd9_modaug)

# ============================================================================
# Step 0: Verify teacher checkpoints exist
# ============================================================================

echo "=== Checking teacher checkpoints ==="
for ckpt in \
    full_method/runs/dscformer_srl_G1/best.pt \
    full_method/runs/dscformer_plain_G0/best.pt \
    full_method/runs/sam_lora_srl_SAM2/best.pt; do
    if [[ -f "$ckpt" ]]; then
        echo "  [OK] $ckpt"
    else
        echo "  [MISSING] $ckpt"
        if [[ "$EVAL_ONLY" == "false" ]]; then
            echo "  ERROR: Cannot train without teacher checkpoint. Aborting."
            exit 1
        fi
    fi
done

# ============================================================================
# Step 1: Train each model (~6-8h each, ~20h total)
# ============================================================================

if [[ "$EVAL_ONLY" == "false" ]]; then

mkdir -p logs

for i in "${!PRESETS[@]}"; do
    PRESET="${PRESETS[$i]}"
    NAME="${NAMES[$i]}"
    CKPT="full_method/runs/${NAME}/best.pt"

    if [[ -f "$CKPT" ]]; then
        echo ""
        echo "[${PRESET}] Already trained (${CKPT} exists), skipping."
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  Training ${PRESET} (${NAME})"
    echo "=========================================="
    python -m full_method.train \
        --ablation "$PRESET" \
        --name "$NAME" \
        2>&1 | tee "logs/${PRESET}.log"
done

fi  # end if not eval-only

# ============================================================================
# Step 2: Evaluate each model
# ============================================================================

echo ""
echo "=== Evaluation ==="

for i in "${!PRESETS[@]}"; do
    PRESET="${PRESETS[$i]}"
    NAME="${NAMES[$i]}"
    CKPT="full_method/runs/${NAME}/best.pt"

    if [[ ! -f "$CKPT" ]]; then
        echo "[WARN] ${NAME}: checkpoint not found, skipping eval."
        continue
    fi

    echo ""
    echo "[eval] ${NAME} ..."

    # Standard eval
    python -m full_method.eval_tta \
        --ablation "$PRESET" \
        --name "$NAME" \
        --no-tta \
        2>&1 || echo "[WARN] eval failed for $NAME"

    # TTA eval
    python -m full_method.eval_tta \
        --ablation "$PRESET" \
        --name "$NAME" \
        2>&1 || echo "[WARN] TTA eval failed for $NAME"
done

# ============================================================================
# Step 3: Print comparison table
# ============================================================================

echo ""
echo "=== Results comparison ==="
python3 -c "
import json, os

models = {
    'DKD2 (baseline)': 'results/dual_kd_classaware_DKD2_test.json',
    'DKD7 (+curriculum)': 'results/dkd7_curriculum_test.json',
    'DKD8 (G0 teacher)': 'results/dkd8_teacher_g0_test.json',
    'DKD9 (+mod aug)': 'results/dkd9_modaug_test.json',
}

print(f\"{'Model':<25} {'mIoU_fg':>8} {'IoU_cr':>8} {'IoU_sp':>8} {'BF1_fg':>8} {'clDice':>8} {'ConnR_fg':>9}\")
print('-' * 80)

for name, path in models.items():
    if not os.path.exists(path):
        print(f'{name:<25} (not found)')
        continue
    d = json.load(open(path))['overall']
    print(f\"{name:<25} {d['mIoU_fg']*100:>7.1f}% {d['IoU_crack']*100:>7.1f}% {d['IoU_spalling']*100:>7.1f}% {d['BF1_fg_mean']*100:>7.1f}% {d.get('clDice_fg_mean',0)*100:>7.1f}% {d.get('ConnR_fg_mean',0)*100:>8.1f}%\")
"

echo ""
echo "=== Done. Check results/ for JSON files ==="

#!/usr/bin/env bash
# Retrain key models on group-based split to assess spatial leakage.
# Run on RunPod after: python scripts/eval_group_split.py --build-split
set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GS="${CODES_DIR}/baseline_unet/splits/group_split"
TR="${GS}/train.txt"
VA="${GS}/val.txt"

cd "$CODES_DIR"

# Verify split files exist
if [ ! -f "$TR" ] || [ ! -f "$VA" ]; then
    echo "[ERROR] Group split files not found. Run:"
    echo "  python scripts/eval_group_split.py --build-split"
    exit 1
fi

echo "=== Group-split retraining ==="
echo "  train: $(wc -l < "$TR") samples"
echo "  val:   $(wc -l < "$VA") samples"

# ── Phase 1: Independent baselines ──────────────────────────────────────

echo ""
echo "=== Phase 1: Baselines ==="

echo "--- 1/6 U-Net R34 (320, 30ep) ---"
python -m baseline_unet.train --epochs 30 \
    --train-split "$TR" --val-split "$VA" \
    --name unet_r34_ce_dice_gsplit

echo "--- 2/6 DeepLabV3+ R50 (512, 50ep) ---"
python -m baseline_deeplab.train --preset A --epochs 50 \
    --train-split "$TR" --val-split "$VA" \
    --name deeplabv3p_r50_512_gsplit

echo "--- 3/6 SegFormer B2 plain (512, 100ep) ---"
python -m baseline_segformer.train --preset A --epochs 100 \
    --train-split "$TR" --val-split "$VA" \
    --name segformer_b2_plain_512_gsplit

echo "--- 4/6 Mask2Former (512, 100ep) ---"
python -m baseline_mask2former.train --ablation M0 --epochs 100 \
    --train-split "$TR" --val-split "$VA" \
    --name mask2former_plain_M0_gsplit

# ── Phase 2: Teachers for DTKD ──────────────────────────────────────────

echo ""
echo "=== Phase 2: Teachers ==="

echo "--- 5/6 DSCFormer-SRL G1 (teacher 1, 100ep) ---"
python -m full_method.train --ablation G1 \
    --train-split "$TR" --val-split "$VA" \
    --name dscformer_srl_G1_gsplit

echo "--- 6/6 SAM2-LoRA (teacher 2, 50ep) ---"
python -m full_method.train --ablation SAM2 \
    --train-split "$TR" --val-split "$VA" \
    --name sam_lora_srl_SAM2_gsplit

# ── Phase 3: Student (depends on Phase 2 teachers) ─────────────────────

echo ""
echo "=== Phase 3: Student ==="

echo "--- 7/7 DTKD DKD2 (100ep) ---"
python -m full_method.train --ablation DKD2 \
    --train-split "$TR" --val-split "$VA" \
    --name dual_kd_classaware_DKD2_gsplit \
    --kd-teacher-checkpoint runs/dscformer_srl_G1_gsplit/best.pt \
    --kd-teacher2-checkpoint runs/sam_lora_srl_SAM2_gsplit/best.pt

# ── Phase 4: Evaluate ──────────────────────────────────────────────────

echo ""
echo "=== Phase 4: Evaluate ==="
python scripts/eval_group_split.py --evaluate --retrained --compare

echo ""
echo "=== All group-split retraining complete ==="
echo "Results: ${GS}/group_split_results_retrained.csv"

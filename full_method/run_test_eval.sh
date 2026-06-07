#!/bin/bash
# Evaluate key runs on test set (plain + TTA) for paper tables.
# Run from the Codes/ directory on RunPod.
#
# Usage:
#   cd /workspace/Codes
#   bash full_method/run_test_eval.sh

set -e

echo "============================================"
echo "Test set evaluation for paper tables"
echo "============================================"

# --- 1. DKD5e (best model) ---
echo ""
echo ">>> DKD5e: plain test eval"
python -m full_method.eval_tta --run dkd5e_crk07_spl02 --ablation DKD5e --no-tta

echo ""
echo ">>> DKD5e: TTA eval"
python -m full_method.eval_tta --run dkd5e_crk07_spl02 --ablation DKD5e

# --- 2. G1 teacher (DSCformer + SRL) ---
echo ""
echo ">>> G1: plain test eval"
python -m full_method.eval_tta --run dscformer_srl_G1 --no-tta

echo ""
echo ">>> G1: TTA eval"
python -m full_method.eval_tta --run dscformer_srl_G1

# --- 3. G0 (DSCformer, no SRL) ---
if [ -d "full_method/runs/dscformer_G0" ]; then
    echo ""
    echo ">>> G0: plain test eval"
    python -m full_method.eval_tta --run dscformer_G0 --model-type dscformer --no-tta
fi

# --- 4. P0 baseline (SegFormer-B2) ---
if [ -d "full_method/runs/segformer_b2_full_512" ]; then
    echo ""
    echo ">>> P0: plain test eval"
    python -m full_method.eval_tta --run segformer_b2_full_512 --model-type segformer --no-tta
fi

# --- 5. SAM-LoRA ---
if [ -d "full_method/runs/sam_lora_srl_SAM2" ]; then
    echo ""
    echo ">>> SAM2: plain test eval"
    python -m full_method.eval_tta --run sam_lora_srl_SAM2 --no-tta
fi

echo ""
echo "============================================"
echo "All evaluations complete!"
echo "Check full_method/runs/*/test_report_*.txt"
echo "============================================"

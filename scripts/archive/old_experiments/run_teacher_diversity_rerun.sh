#!/bin/bash
# Matched-checkpoint rerun: T1-only vs duplicated-T1 vs heterogeneous DTKD
# All three use the SAME T1 checkpoint, same seed, same hyperparameters.
# The ONLY variable is teacher identity.
#
# Usage on RunPod:
#   cd /workspace/dam-segmentation-baselines
#   bash scripts/run_teacher_diversity_rerun.sh
#
# Expected time: ~3 x 25min = ~75 min on A40

set -e

echo "============================================"
echo "  Teacher Diversity Matched-Checkpoint Rerun"
echo "============================================"

# Verify teacher checkpoints exist
T1_CKPT="full_method/runs/dscformer_srl_G1/best.pt"
T2_CKPT="full_method/runs/sam_lora_srl_SAM2/best.pt"

if [ ! -f "$T1_CKPT" ]; then
    echo "ERROR: T1 checkpoint not found: $T1_CKPT"
    exit 1
fi
if [ ! -f "$T2_CKPT" ]; then
    echo "ERROR: T2 checkpoint not found: $T2_CKPT"
    exit 1
fi

echo "T1 checkpoint: $T1_CKPT ($(md5sum $T1_CKPT | cut -d' ' -f1))"
echo "T2 checkpoint: $T2_CKPT ($(md5sum $T2_CKPT | cut -d' ' -f1))"
echo ""

# 1. T1-only KD
echo "[1/3] Training RERUN_T1ONLY (T1-only KD)..."
python -m full_method.train --ablation RERUN_T1ONLY --seed 42
echo "  Done."

# 2. Duplicated T1
echo "[2/3] Training RERUN_DUP_T1 (duplicated T1)..."
python -m full_method.train --ablation RERUN_DUP_T1 --seed 42
echo "  Done."

# 3. Heterogeneous T1+T2
echo "[3/3] Training RERUN_HETERO (heterogeneous T1+T2)..."
python -m full_method.train --ablation RERUN_HETERO --seed 42
echo "  Done."

echo ""
echo "============================================"
echo "  All 3 training runs complete."
echo "  Now evaluating..."
echo "============================================"

# Evaluate all three
python scripts/eval_teacher_diversity_rerun.py

echo ""
echo "Done! Results saved to results/teacher_diversity_rerun.json"

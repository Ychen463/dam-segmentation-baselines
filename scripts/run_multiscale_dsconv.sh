#!/bin/bash
# Multi-Scale DSConv experiments
#
# Runs:
#   1. MS0: Multi-Scale DSConv (5/9/15), no SRL — compare with DSConv-only baseline
#   2. MS1: Multi-Scale DSConv (5/9/15) + SRL — full version
#
# Baselines already trained:
#   - dscformer_plain_G0 (single-scale DSConv k=9, no SRL)
#   - dscformer_srl_G1_v2 (single-scale DSConv k=9, + SRL)
#
# Usage on RunPod:
#   cd /workspace/dam-segmentation-baselines
#   bash scripts/run_multiscale_dsconv.sh
#
# Expected time: ~50 min on A40 (2 x 25min)

set -e

echo "============================================"
echo "  Multi-Scale DSConv Experiments"
echo "============================================"

# 1. MS0: Multi-Scale, no SRL
echo "[1/2] Training MS0 (Multi-Scale DSConv, no SRL)..."
python -m full_method.train --ablation MS0 --seed 42
echo "  Done."

# 2. MS1: Multi-Scale + SRL
echo "[2/2] Training MS1 (Multi-Scale DSConv + SRL)..."
python -m full_method.train --ablation MS1 --seed 42
echo "  Done."

# Evaluate
echo ""
echo "Evaluating..."
python scripts/eval_multiscale_dsconv.py

echo ""
echo "Done! Results saved to results/multiscale_dsconv.json"

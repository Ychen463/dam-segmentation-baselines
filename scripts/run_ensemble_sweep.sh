#!/bin/bash
# Ensemble sweep: test all multi-model combinations
#
# Tests diverse architecture ensembles:
#   G0 (DSCformer), G1 (DSCformer+SRL), P0 (SegFormer),
#   SAM2 (SAM-LoRA), DKD2 (Dual-KD student), G0 multi-seed
#
# Three passes:
#   1. No TTA, no PP (baseline ensemble performance)
#   2. With TTA (ensemble + test-time augmentation)
#   3. With PP (ensemble + TAPP post-processing)
#
# Usage on RunPod:
#   cd /workspace/dam-segmentation-baselines
#   bash scripts/run_ensemble_sweep.sh
#
# Expected time: ~15-20 min on A40 (inference only, no training)

set -e

echo "============================================"
echo "  Ensemble Sweep Evaluation"
echo "============================================"

# Pass 1: Plain ensemble (no TTA, no PP)
echo ""
echo "[1/3] Plain ensemble..."
python scripts/run_ensemble_sweep.py
echo "  Done."

# Pass 2: Ensemble + TTA
echo ""
echo "[2/3] Ensemble + TTA..."
python scripts/run_ensemble_sweep.py --tta
echo "  Done."

# Pass 3: Ensemble + TAPP post-processing (no TTA, faster)
echo ""
echo "[3/3] Ensemble + TAPP post-processing..."
python scripts/run_ensemble_sweep.py --with-pp
echo "  Done."

echo ""
echo "============================================"
echo "  All done! Results:"
echo "    results/ensemble_sweep.json"
echo "    results/ensemble_sweep_tta.json"
echo "============================================"

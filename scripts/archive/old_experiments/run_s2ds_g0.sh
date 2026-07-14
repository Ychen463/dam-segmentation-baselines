#!/usr/bin/env bash
# Cross-dataset eval: DSCFormer (G0, row b) on S2DS.
# Decomposes the S2DS transfer gain: SegFormer→DSCFormer→DSCFormer+DTKD
# Run on RunPod:
#   cd /workspace/Codes && bash scripts/run_s2ds_g0.sh
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

# 1. Confirm G0 checkpoint exists
CKPT="full_method/runs/dscformer_plain_G0/best.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: missing G0 checkpoint: $CKPT"
    echo "You may need to retrain: python -m full_method.train --ablation G0"
    exit 1
fi
echo "G0 checkpoint OK: $CKPT"

# 2. Confirm S2DS is prepared
if [ ! -f "Dataset/S2DS/test_files.txt" ]; then
    echo "ERROR: S2DS not prepared. Run: python scripts/prepare_s2ds.py --raw-dir /path/to/s2ds/"
    exit 1
fi
echo "S2DS dataset OK."

# 3. Evaluate G0 on S2DS (no TTA, with per-image CSV for stats)
echo ""
echo "=== Evaluating DSCFormer (G0) on S2DS ==="
python scripts/eval_cross_dataset.py --model dscformer_plain_G0 --per-image

echo ""
echo "=== Done ==="
echo "Results saved to: results/cross_dataset/s2ds_dscformer_plain_G0.json"
echo "Per-image CSV:    results/cross_dataset/s2ds_dscformer_plain_G0_per_image.csv"
echo ""
echo "Copy back to local:"
echo "  results/cross_dataset/s2ds_dscformer_plain_G0.json"
echo "  results/cross_dataset/s2ds_dscformer_plain_G0_per_image.csv"

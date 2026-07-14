#!/usr/bin/env bash
# Retrain DSCFormer-only (G0, row b) on feature-cluster split.
# This fills the missing comparator in Table 11 so we can verify
# that DTKD's +1.9 mIoU gain over DSConv replicates independently.
#
# Run on RunPod:
#   cd /workspace/Codes && bash scripts/run_gsplit_g0.sh
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

echo "=== Retrain G0 (DSCFormer-only) on feature-cluster split ==="
echo "  train: $(wc -l < "$TR") samples"
echo "  val:   $(wc -l < "$VA") samples"

# Train G0 on cluster split
python -m full_method.train --ablation G0 \
    --train-split "$TR" --val-split "$VA" \
    --name dscformer_plain_G0_gsplit

# Evaluate on cluster test set
echo ""
echo "=== Evaluate G0 on cluster test set ==="
python scripts/eval_group_split.py --evaluate --retrained \
    --models dscformer_plain_G0

echo ""
echo "=== Results ==="
echo "Copy back: full_method/runs/dscformer_plain_G0_gsplit/test_report.txt"

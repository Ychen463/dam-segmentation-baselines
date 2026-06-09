#!/bin/bash
# Run ACCW (Adaptive Class-Conditional Weighting) experiments on RunPod.
#
# ACCW1: Full adaptive + conf-KD (main result)
# ACCW2: Adaptive only, no conf-KD (ablation)
# ACCW3: Adaptive from equal init (ablation)
#
# Usage:
#   bash scripts/run_accw.sh           # run all 3
#   bash scripts/run_accw.sh ACCW1     # run specific preset

set -e
cd "$(dirname "$0")/.."

PRESETS="${1:-ACCW1 ACCW2 ACCW3}"

for preset in $PRESETS; do
    echo "=========================================="
    echo "  Running $preset"
    echo "=========================================="
    python -m full_method.train --preset "$preset"
    echo ""
    echo "[$preset] done."
    echo ""
done

echo "All ACCW experiments complete."

# Evaluate all on test set
echo ""
echo "=========================================="
echo "  Evaluation summary"
echo "=========================================="
for preset in $PRESETS; do
    name=$(python -c "
from full_method.config import ABLATION_PRESETS
print(ABLATION_PRESETS['$preset']['name'])
")
    echo "--- $preset ($name) ---"
    best="full_method/runs/$name/best.pt"
    if [ -f "$best" ]; then
        python -c "
import torch
ckpt = torch.load('$best', map_location='cpu', weights_only=False)
print(f'  mIoU_fg: {ckpt.get(\"mIoU_fg\", \"?\"):.4f}')
if 'accw_weights' in ckpt:
    aw = ckpt['accw_weights']
    print(f'  learned w2: bg={aw[\"w2_bg\"]:.4f} crack={aw[\"w2_crack\"]:.4f} spalling={aw[\"w2_spalling\"]:.4f}')
"
    else
        echo "  best.pt not found"
    fi
done

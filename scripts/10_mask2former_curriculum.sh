#!/usr/bin/env bash
# M0/M1: Mask2Former Swin-Small — plain baseline + P2 soft curriculum transfer
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== M0: Mask2Former plain (100 epochs) ==="
python -m baseline_mask2former.train --ablation M0

echo ""
echo "=== M1: Mask2Former + P2 soft curriculum (100 epochs) ==="
python -m baseline_mask2former.train --ablation M1

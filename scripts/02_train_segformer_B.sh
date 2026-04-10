#!/usr/bin/env bash
# 训练 SegFormer-B2 static curriculum (preset B, 100 ep)
# 产出: baseline_segformer/runs/segformer_b2_staticcurr_512/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline_segformer.train --preset B

#!/usr/bin/env bash
# 训练 Mask2Former Swin-Small (40 ep, lr=1e-4, img=512, bs=4)
# 产出: baseline_mask2former/runs/mask2former_swin_small_512/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline_mask2former.train

#!/usr/bin/env bash
# 训练 DeepLabV3+ ResNet34 (preset B, 30 ep, lr=1e-3, img=320, bs=8)
# 产出: baseline_deeplab/runs/deeplabv3p_r34_320/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline_deeplab.train --preset B

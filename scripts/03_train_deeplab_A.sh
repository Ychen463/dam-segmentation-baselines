#!/usr/bin/env bash
# 训练 DeepLabV3+ ResNet50 (preset A, 50 ep, lr=5e-4, img=512, bs=4)
# 产出: baseline_deeplab/runs/deeplabv3p_r50_512/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline_deeplab.train --preset A

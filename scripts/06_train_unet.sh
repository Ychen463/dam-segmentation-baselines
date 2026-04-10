#!/usr/bin/env bash
# 训练 U-Net ResNet34 (30 ep, lr=1e-3, img=320, bs=8)
# 产出: baseline_unet/runs/unet_r34_ce_dice/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline_unet.train

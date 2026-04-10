#!/usr/bin/env bash
# 验证数据集完整性: channel order, shape, class bounds, overlap stats, previews
# 产出: baseline_unet/runs/sanity/peek_*.png
# 在第一次训练前跑一次即可
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline_unet.dataset --peek

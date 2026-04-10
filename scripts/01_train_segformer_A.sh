#!/usr/bin/env bash
# 训练 SegFormer-B2 plain (preset A, 100 ep, lr=6e-5, img=512, bs=4)
# 产出: baseline_segformer/runs/segformer_b2_plain_512/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline_segformer.train --preset A

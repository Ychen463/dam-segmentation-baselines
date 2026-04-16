#!/usr/bin/env bash
# 训练 Full Method: SegFormer-B2 + Boundary Head + Dynamic Curriculum
# 产出: full_method/runs/segformer_b2_full_512/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."
python -m full_method.train

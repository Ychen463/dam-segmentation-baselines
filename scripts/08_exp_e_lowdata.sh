#!/usr/bin/env bash
# 实验 E: 低数据实验 (SegFormer-B2 plain, 20/40/60/80%)
# 前置: 01_train_segformer_A.sh 已完成
# 产出: baseline_segformer/runs/segformer_b2_plain_512/best_{20,40,60,80}.pt
set -euo pipefail
cd "$(dirname "$0")/.."

# 备份 100% best.pt
cp baseline_segformer/runs/segformer_b2_plain_512/best.pt \
   baseline_segformer/runs/segformer_b2_plain_512/best_100pct.pt

# 生成子集 split 文件
python -m shared_eval.subset_splits --fractions 0.2 0.4 0.6 0.8

for frac in 20 40 60 80; do
    echo "===== $(date '+%H:%M:%S') | Training ${frac}% ====="
    python -m baseline_segformer.train --preset A \
        --train-split baseline_unet/splits/train_${frac}.txt
    cp baseline_segformer/runs/segformer_b2_plain_512/best.pt \
       baseline_segformer/runs/segformer_b2_plain_512/best_${frac}.pt
done

# 恢复 100% best.pt
cp baseline_segformer/runs/segformer_b2_plain_512/best_100pct.pt \
   baseline_segformer/runs/segformer_b2_plain_512/best.pt

echo "Done. Checkpoints: best_{20,40,60,80}.pt + best.pt (100%)"

#!/usr/bin/env bash
# 1. Retrain Mask2Former  2. Eval  3. Regenerate Fig 3 (without Mask2Former)
set -euo pipefail

# Clean stale metrics so training doesn't error
rm -f baseline_mask2former/runs/mask2former_plain_M0/metrics.csv
rm -f baseline_mask2former/runs/mask2former_swin_small_512/metrics.csv

echo "=== 1. Retrain Mask2Former M0 ==="
python -m baseline_mask2former.train --ablation M0 --epochs 100

echo "=== 2. Eval Mask2Former ==="
python -m shared_eval.eval_all --model mask2former_swin_small_512 --split test --per-tier
python -m shared_eval.eval_all --model mask2former_swin_small_512 --split test --postprocess
python -m shared_eval.eval_all --model mask2former_swin_small_512 --split test --tta
python -m shared_eval.eval_all --model mask2former_swin_small_512 --split test --multiscale

echo "=== 3. Regenerate Fig 3 (without Mask2Former) ==="
python scripts/fig3_qualitative.py \
    --models deeplabv3p_r50_512 segformer_b2_plain_512 dataopt_baseline_N0 \
    --output figures/fig3_qualitative.pdf

echo "=== Done ==="

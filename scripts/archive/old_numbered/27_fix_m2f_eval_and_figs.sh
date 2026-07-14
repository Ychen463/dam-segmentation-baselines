#!/usr/bin/env bash
# Fix Mask2Former eval (registry path now corrected) + regenerate Fig 3 with all models
set -euo pipefail

echo "=== 1. Re-eval Mask2Former (fixed checkpoint path) ==="
python -m shared_eval.eval_all --model mask2former_swin_small_512 --split test --per-tier
python -m shared_eval.eval_all --model mask2former_swin_small_512 --split test --postprocess
python -m shared_eval.eval_all --model mask2former_swin_small_512 --split test --tta
python -m shared_eval.eval_all --model mask2former_swin_small_512 --split test --multiscale

echo "=== 2. Regenerate Fig 3 (with Mask2Former restored) ==="
python scripts/fig3_qualitative.py \
    --models deeplabv3p_r50_512 segformer_b2_plain_512 mask2former_swin_small_512 dataopt_baseline_N0 \
    --output figures/fig3_qualitative.pdf

echo "=== Done ==="

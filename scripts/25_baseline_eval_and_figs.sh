#!/usr/bin/env bash
# Post-retrain: eval baselines (per-tier) + generate paper figures
set -euo pipefail

echo "=== 1. Baseline eval with per-tier ==="
for m in unet_r34_320 deeplabv3p_r50_512 deeplabv3p_r34_320 \
         segformer_b2_plain_512 segformer_b2_staticcurr_512 \
         mask2former_swin_small_512; do
    echo "--- ${m} ---"
    python -m shared_eval.eval_all --model "${m}" --split test --per-tier
    python -m shared_eval.eval_all --model "${m}" --split test --postprocess
    python -m shared_eval.eval_all --model "${m}" --split test --tta
    python -m shared_eval.eval_all --model "${m}" --split test --multiscale
done

echo "=== 2. Fig 1: Dataset samples ==="
python scripts/fig1_dataset_samples.py --output figures/fig1_dataset.pdf

echo "=== 3. Fig 3: Qualitative comparison ==="
python scripts/fig3_qualitative.py \
    --models deeplabv3p_r50_512 segformer_b2_plain_512 mask2former_swin_small_512 dataopt_baseline_N0 \
    --output figures/fig3_qualitative.pdf

echo "=== Done! Check figures/ and results/ ==="

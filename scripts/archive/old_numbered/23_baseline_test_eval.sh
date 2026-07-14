#!/usr/bin/env bash
# Baseline + DSCformer test evaluation — fill in missing test results
# All checkpoints already exist, pure inference only
set -euo pipefail

echo "=== Phase 1: External baselines ==="
for m in unet_r34_320 deeplabv3p_r50_512 deeplabv3p_r34_320 \
         segformer_b2_plain_512 segformer_b2_staticcurr_512 \
         mask2former_swin_small_512; do
    echo "--- ${m} ---"
    python -m shared_eval.eval_all --model "${m}" --split test --per-tier
    python -m shared_eval.eval_all --model "${m}" --split test --postprocess
    python -m shared_eval.eval_all --model "${m}" --split test --tta
    python -m shared_eval.eval_all --model "${m}" --split test --multiscale
done

echo "=== Phase 2: DSCformer variants ==="
for m in dscformer_plain_G0 dscformer_srl_G1 dscformer_full_G2 \
         dscformer_aux_H0 dscformer_aux_srl_H1; do
    echo "--- ${m} ---"
    python -m shared_eval.eval_all --model "${m}" --split test --per-tier
    python -m shared_eval.eval_all --model "${m}" --split test --postprocess
    python -m shared_eval.eval_all --model "${m}" --split test --tta
    python -m shared_eval.eval_all --model "${m}" --split test --multiscale
done

echo "=== Phase 3: N0 baseline raw (missing _test.json) ==="
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --per-tier

echo "=== All baseline test eval complete ==="
echo "Results saved to results/ directory"

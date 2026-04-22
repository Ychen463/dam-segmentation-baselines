#!/usr/bin/env bash
# Retrain ALL baselines + full eval (checkpoints were lost)
set -euo pipefail

echo "=== 1/5 U-Net R34 (320, ~15min) ==="
python -m baseline_unet.train --epochs 30

echo "=== 2/5 DeepLabV3+ R50 (512, ~20min) ==="
python -m baseline_deeplab.train --preset A --epochs 50

echo "=== 3/5 DeepLabV3+ R34 (320, ~15min) ==="
python -m baseline_deeplab.train --preset B --epochs 30

echo "=== 4/5 SegFormer B2 plain (512, ~30min) ==="
python -m baseline_segformer.train --preset A --epochs 100

echo "=== 5/5 SegFormer B2 static curriculum (512, ~30min) ==="
python -m baseline_segformer.train --preset B --epochs 100

# Mask2Former is expensive (~40min each × 6 presets), train plain only
echo "=== 6/5 Mask2Former plain M0 (512, ~40min) ==="
python -m baseline_mask2former.train --ablation M0 --epochs 100

echo "=== Full test eval for all baselines ==="
for m in unet_r34_320 \
         deeplabv3p_r50_512 deeplabv3p_r34_320 \
         segformer_b2_plain_512 segformer_b2_staticcurr_512 \
         mask2former_swin_small_512; do
    echo "--- eval ${m} ---"
    python -m shared_eval.eval_all --model "${m}" --split test --per-tier
    python -m shared_eval.eval_all --model "${m}" --split test --postprocess
    python -m shared_eval.eval_all --model "${m}" --split test --tta
    python -m shared_eval.eval_all --model "${m}" --split test --multiscale
done

echo "=== All baselines retrained + evaluated ==="

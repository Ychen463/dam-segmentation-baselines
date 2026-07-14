#!/bin/bash
# Snake Channel / Kernel-size sweep experiments
#
# Sweep: snake_channels={32,64} x snake_kernel_size={7,9,13}
# Baseline G0 (ch=64, k=9) already trained — 5 new experiments.
#
# Usage on RunPod:
#   cd /workspace/dam-segmentation-baselines
#   bash scripts/run_snake_sweep.sh
#
# Expected time: ~2h on A40 (5 x ~25min)

set -e

echo "============================================"
echo "  Snake Channel / Kernel-size Sweep"
echo "============================================"
echo "Baseline: G0 (ch=64, k=9) — already trained"
echo ""

PRESETS=(SC0 SC1 SC2 SC3 SC4)
LABELS=(
    "SC0: ch=32, k=9"
    "SC1: ch=64, k=7"
    "SC2: ch=64, k=13"
    "SC3: ch=32, k=7"
    "SC4: ch=32, k=13"
)

for i in "${!PRESETS[@]}"; do
    p="${PRESETS[$i]}"
    l="${LABELS[$i]}"
    echo "[$(($i+1))/${#PRESETS[@]}] Training ${l}..."
    python -m full_method.train --ablation "$p" --seed 42
    echo "  Done."
    echo ""
done

# Evaluate
echo "Evaluating all configurations..."
python scripts/eval_snake_sweep.py

echo ""
echo "Done! Results saved to results/snake_sweep.json"

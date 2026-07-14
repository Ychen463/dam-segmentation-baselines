#!/usr/bin/env bash
# DCNv2 vs DSConv control experiment (3 seeds each).
# Purpose: Test whether DSConv's constrained snake geometry outperforms
# general deformable convolutions (DCNv2).
# Run on RunPod after git pull.
set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODES_DIR"

SEEDS=(42 123 2024)

echo "=== DCNv2 vs DSConv Control Experiment (3 seeds) ==="

# ── DSConv (G0 baseline, 3 seeds) ─────────────────────────────────────────
echo ""
echo "=== DSConv branch (G0) ==="
for seed in "${SEEDS[@]}"; do
    echo "--- G0 seed=$seed ---"
    python -m full_method.train --ablation G0 \
        --name "dsconv_G0_s${seed}" \
        --seed "$seed"
done

# ── DCNv2 control (3 seeds) ───────────────────────────────────────────────
echo ""
echo "=== DCNv2 branch (DCNV2_CTRL) ==="
for seed in "${SEEDS[@]}"; do
    echo "--- DCNV2_CTRL seed=$seed ---"
    python -m full_method.train --ablation DCNV2_CTRL \
        --name "dcnv2_ctrl_s${seed}" \
        --seed "$seed"
done

# ── Standard Conv control (3 seeds, optional) ─────────────────────────────
echo ""
echo "=== Standard Conv branch (CONV_CTRL) ==="
for seed in "${SEEDS[@]}"; do
    echo "--- CONV_CTRL seed=$seed ---"
    python -m full_method.train --ablation CONV_CTRL \
        --name "stdconv_ctrl_s${seed}" \
        --seed "$seed"
done

echo ""
echo "=== All branch control experiments complete ==="
echo "Compare results across runs/dsconv_G0_s*/metrics.json,"
echo "  runs/dcnv2_ctrl_s*/metrics.json, runs/stdconv_ctrl_s*/metrics.json"

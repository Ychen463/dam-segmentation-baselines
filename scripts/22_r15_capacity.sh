#!/usr/bin/env bash
# Round 15: Resolution + Backbone + Multi-Scale — Capacity-First Improvements
# R15a (640 res), R15b (B5 backbone), R15c (multiscale eval), R15d (OHEM),
# R15e (lr=3e-5), R15f (lr=1e-4)
set -euo pipefail

# --- Phase 1: Multi-scale eval on existing N0 (no training needed) ---
echo "=== Phase 1: Multi-scale eval on N0 ==="
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --multiscale
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --multiscale --tta
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --multiscale --postprocess
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --multiscale --tta --postprocess

# --- Phase 2: LR sweep ---
echo "=== Phase 2: LR sweep ==="
for preset in R15e R15f; do
    echo "=== Train ${preset} ==="
    python -m full_method.train --ablation "${preset}" --epochs 100
done

# --- Phase 3: High resolution ---
echo "=== Phase 3: High resolution (R15a) ==="
python -m full_method.train --ablation R15a --epochs 100

# --- Phase 4: Larger backbone ---
echo "=== Phase 4: Larger backbone (R15b) ==="
python -m full_method.train --ablation R15b --epochs 100

# --- Phase 5: OHEM ---
echo "=== Phase 5: OHEM (R15d) ==="
python -m full_method.train --ablation R15d --epochs 100

# --- Full eval for all trained models ---
declare -A MODEL_NAMES=(
    [R15e]="r15_lr3e5_R15e"
    [R15f]="r15_lr1e4_R15f"
    [R15a]="r15_hires_640"
    [R15b]="r15_b5_512"
    [R15d]="r15_ohem_R15d"
)

for preset in R15e R15f R15a R15b R15d; do
    model="${MODEL_NAMES[$preset]}"
    echo "=== Eval ${preset} (${model}) ==="
    python -m shared_eval.eval_all --model "${model}" --split test --per-tier
    python -m shared_eval.eval_all --model "${model}" --split test --postprocess
    python -m shared_eval.eval_all --model "${model}" --split test --tta --postprocess
    python -m shared_eval.eval_all --model "${model}" --split test --multiscale
    python -m shared_eval.eval_all --model "${model}" --split test --multiscale --tta
done

echo "=== Round 15 complete ==="

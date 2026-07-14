#!/usr/bin/env bash
# Round 14: EMA + Lovász + CRF — three orthogonal quick gains
# E1 (EMA only), E2 (Lovász only), E3 (EMA + Lovász), CRF (post-processing)
set -euo pipefail

# --- 1. CRF eval on existing N0 (no training needed) ---
echo "=== CRF eval on N0 baseline ==="
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --crf
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --crf --postprocess
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --tta --crf
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --tta --crf --postprocess

# --- 2. Training + Evaluation ---
declare -A MODEL_NAMES=(
    [E1]="ema_only_E1"
    [E2]="lovasz_only_E2"
    [E3]="ema_lovasz_E3"
)

for preset in E1 E2 E3; do
    model="${MODEL_NAMES[$preset]}"
    echo "=== Train ${preset} ==="
    python -m full_method.train --ablation "${preset}" --epochs 100

    echo "=== Eval ${preset} ==="
    python -m shared_eval.eval_all --model "${model}" --split test --per-tier
    python -m shared_eval.eval_all --model "${model}" --split test --postprocess
    python -m shared_eval.eval_all --model "${model}" --split test --tta
    python -m shared_eval.eval_all --model "${model}" --split test --tta --postprocess

    # CRF combinations
    python -m shared_eval.eval_all --model "${model}" --split test --crf
    python -m shared_eval.eval_all --model "${model}" --split test --crf --postprocess
    python -m shared_eval.eval_all --model "${model}" --split test --tta --crf
    python -m shared_eval.eval_all --model "${model}" --split test --tta --crf --postprocess
done

echo "=== Round 14 complete ==="

#!/usr/bin/env bash
# Round 13: Decoupled Ablation — separate aug vs loss improvements
# N1m (moderate aug), N2a (early SRL), N3a (loss opts), N4a (combined)
set -euo pipefail

# --- Backfill: PP+TTA eval for N0/G1 baseline ---
echo "=== Backfill: N0 PP+TTA eval ==="
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --postprocess
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --tta
python -m shared_eval.eval_all --model dataopt_baseline_N0 --split test --tta --postprocess

# --- Training + Evaluation ---
for preset in N1m N2a N3a N4a; do
    echo "=== Train ${preset} ==="
    python -m full_method.train --ablation "${preset}" --epochs 100

    echo "=== Eval ${preset} ==="
    python -m shared_eval.eval_all --model "dataopt_*_${preset}" --split test --per-tier
    python -m shared_eval.eval_all --model "dataopt_*_${preset}" --split test --postprocess
    python -m shared_eval.eval_all --model "dataopt_*_${preset}" --split test --tta
    python -m shared_eval.eval_all --model "dataopt_*_${preset}" --split test --tta --postprocess
done

# --- MN4: MAC + best data opts (run only if N4a > N0) ---
echo "=== MN4: full MAC + moderate aug + loss opts ==="
python -m full_method.train --ablation MN4 --epochs 100
python -m shared_eval.eval_all --model mac_dataopt_MN4 --split test --per-tier --postprocess
python -m shared_eval.eval_all --model mac_dataopt_MN4 --split test --tta --postprocess

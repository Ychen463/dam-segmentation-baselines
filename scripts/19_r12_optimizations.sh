#!/usr/bin/env bash
# Round 12: Data-Driven Optimizations (N0-N3 ablation + post-processing eval)
set -euo pipefail

# --- Training: N0 → N3 cumulative ablation ---
echo "=== N0: G1 reproduction (sanity check) ==="
python -m full_method.train --ablation N0 --epochs 100

echo "=== N1: + strong augmentation ==="
python -m full_method.train --ablation N1 --epochs 100

echo "=== N2: N1 + SRL epoch 30 ==="
python -m full_method.train --ablation N2 --epochs 100

echo "=== N3: N2 + SRL 0.10 + Tversky ==="
python -m full_method.train --ablation N3 --epochs 100

# --- Evaluation ---
for preset in N0 N1 N2 N3; do
    echo "=== Eval ${preset} ==="
    python -m shared_eval.eval_all --model "dscformer_*_${preset}" --split test --per-tier
    python -m shared_eval.eval_all --model "dscformer_*_${preset}" --split test --postprocess
    python -m shared_eval.eval_all --model "dscformer_*_${preset}" --split test --tta --postprocess
done

# --- MN3: MAC + N3 combined (final best) ---
echo "=== MN3: full MAC + data optimizations ==="
python -m full_method.train --ablation MN3 --epochs 100
python -m shared_eval.eval_all --model mac_dataopt_MN3 --split test --per-tier --postprocess
python -m shared_eval.eval_all --model mac_dataopt_MN3 --split test --tta --postprocess

#!/usr/bin/env bash
# Pipeline 1: A2 → A3a → A3b → S2 (with class-aware diagnostic)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== A2: dynamic difficulty refinement ==="
python -m full_method.train --ablation A2

echo "=== A3a: sampling bonus only (no loss schedule) ==="
python -m full_method.train --ablation A3a

echo "=== A3b: loss schedule only (no sampling bonus) ==="
python -m full_method.train --ablation A3b

echo "=== S2: soft curriculum + crack clDice ==="
python -m full_method.train --ablation S2

echo "========================================"
echo "=== All done. Compare test reports: ==="
echo "========================================"
for f in \
    full_method/runs/ablation_A2_dynamic_refinement/test_report.txt \
    full_method/runs/ablation_A3a_sampling_bonus_only/test_report.txt \
    full_method/runs/ablation_A3b_loss_schedule_only/test_report.txt \
    full_method/runs/stabilized_S2_softcurr_cldice/test_report.txt \
; do
    if [ -f "$f" ]; then
        echo ""; echo "--- $f ---"
        cat "$f"
    fi
done

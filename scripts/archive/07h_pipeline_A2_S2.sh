#!/usr/bin/env bash
# Pipeline 2: A2 → S2 (minimal main line)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== A2: dynamic difficulty refinement ==="
python -m full_method.train --ablation A2

echo "=== S2: soft curriculum + crack clDice ==="
python -m full_method.train --ablation S2

echo "========================================"
echo "=== All done. Compare test reports: ==="
echo "========================================"
for f in \
    full_method/runs/ablation_A2_dynamic_refinement/test_report.txt \
    full_method/runs/stabilized_S2_softcurr_cldice/test_report.txt \
; do
    if [ -f "$f" ]; then
        echo ""; echo "--- $f ---"
        cat "$f"
    fi
done

#!/usr/bin/env bash
# Main experiment pipeline: A0 → A1 → A2 → S2
# Re-runs all with clDice_crack eval metric in test reports.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "========================================"
echo "=== A0: plain SegFormer (no curriculum) ==="
echo "========================================"
python -m baseline_segformer.train --preset A

echo "========================================"
echo "=== A1: static curriculum ==="
echo "========================================"
python -m baseline_segformer.train --preset B

echo "========================================"
echo "=== A2: dynamic difficulty refinement ==="
echo "========================================"
python -m full_method.train --ablation A2

echo "========================================"
echo "=== S2: soft curriculum + crack clDice ==="
echo "========================================"
python -m full_method.train --ablation S2

echo "========================================"
echo "=== All done. Compare test reports: ==="
echo "========================================"
for f in \
    baseline_segformer/runs/segformer_b2_plain_512/test_report.txt \
    baseline_segformer/runs/segformer_b2_staticcurr_512/test_report.txt \
    full_method/runs/ablation_A2_dynamic_refinement/test_report.txt \
    full_method/runs/stabilized_S2_softcurr_cldice/test_report.txt \
; do
    if [ -f "$f" ]; then
        echo ""; echo "--- $f ---"
        cat "$f"
    fi
done

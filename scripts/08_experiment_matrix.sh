#!/usr/bin/env bash
# 2x2 Experiment Matrix: Soft Curriculum x Crack-only clDice
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=100  # each run

echo "========================================"
echo "=== 2x2 Experiment Matrix            ==="
echo "=== P0 -> P1 -> P2 -> P3             ==="
echo "=== ${EPOCHS} epochs each, 4 runs total ==="
echo "========================================"

echo ""
echo ">>> [1/4] P0: Plain SegFormer (canonical baseline) | ${EPOCHS} epochs"
echo ">>> Started at: $(date '+%Y-%m-%d %H:%M:%S')"
python -m full_method.train --ablation P0
echo ">>> P0 finished at: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo ">>> [2/4] P1: Plain + crack-only clDice | ${EPOCHS} epochs"
echo ">>> Started at: $(date '+%Y-%m-%d %H:%M:%S')"
python -m full_method.train --ablation P1
echo ">>> P1 finished at: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo ">>> [3/4] P2: Soft curriculum only (=S1) | ${EPOCHS} epochs"
echo ">>> Started at: $(date '+%Y-%m-%d %H:%M:%S')"
python -m full_method.train --ablation S1
echo ">>> P2 finished at: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo ">>> [4/4] P3: Soft curriculum + crack-only clDice (=S2) | ${EPOCHS} epochs"
echo ">>> Started at: $(date '+%Y-%m-%d %H:%M:%S')"
python -m full_method.train --ablation S2
echo ">>> P3 finished at: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "========================================"
echo "=== All 4 runs done.                 ==="
echo "=== Finished at: $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "========================================"
echo ""
echo "=== Compare test reports: ==="
for f in \
    full_method/runs/plain_segformer_P0/test_report.txt \
    full_method/runs/plain_cldice_P1/test_report.txt \
    full_method/runs/stabilized_S1_softcurr/test_report.txt \
    full_method/runs/stabilized_S2_softcurr_cldice/test_report.txt \
; do
    if [ -f "$f" ]; then
        echo ""; echo "--- $f ---"
        cat "$f"
    fi
done

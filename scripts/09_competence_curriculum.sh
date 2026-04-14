#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=100

echo "========================================="
echo "=== Round 3: Competence Curriculum     ==="
echo "=== P0 -> C1 -> C2 (${EPOCHS} ep each) ==="
echo "========================================="

echo ""
echo ">>> [1/3] P0: Plain SegFormer | ${EPOCHS} epochs"
echo ">>> Started at: $(date '+%Y-%m-%d %H:%M:%S')"
python -m full_method.train --ablation P0
echo ">>> P0 finished at: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo ">>> [2/3] C1: Competence hard unlock | ${EPOCHS} epochs"
echo ">>> Started at: $(date '+%Y-%m-%d %H:%M:%S')"
python -m full_method.train --ablation C1
echo ">>> C1 finished at: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo ">>> [3/3] C2: Competence soft mixing | ${EPOCHS} epochs"
echo ">>> Started at: $(date '+%Y-%m-%d %H:%M:%S')"
python -m full_method.train --ablation C2
echo ">>> C2 finished at: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "========================================="
echo "=== All 3 runs done.                  ==="
echo "=== Finished at: $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "========================================="
echo ""
echo "=== Compare test reports: ==="
for f in \
    full_method/runs/plain_segformer_P0/test_report.txt \
    full_method/runs/competence_hard_C1/test_report.txt \
    full_method/runs/competence_soft_C2/test_report.txt \
; do
    if [ -f "$f" ]; then
        echo ""; echo "--- $f ---"
        cat "$f"
    fi
done

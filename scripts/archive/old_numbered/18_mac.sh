#!/usr/bin/env bash
# Round 11: Morphology-Aware Curriculum (MAC) ablation
# M0: morph difficulty only
# M1: morph + adaptive pacing
# M2: morph + class-conditional loss
# M3: full MAC (morph + pacing + class loss)
set -euo pipefail

echo "=== M0: morph difficulty only ==="
python -m full_method.train --ablation M0

echo "=== M1: morph + adaptive pacing ==="
python -m full_method.train --ablation M1

echo "=== M2: morph + class-conditional loss ==="
python -m full_method.train --ablation M2

echo "=== M3: full MAC ==="
python -m full_method.train --ablation M3

#!/usr/bin/env bash
# Run Topology-Aware KD experiments (Direction B)
# TOPOKD1: full (topo-KD + conf-KD)
# TOPOKD2: topo-KD only (no conf-KD) — isolate topo-KD effect
# TOPOKD3: topo-KD with higher weight (0.20)
set -euo pipefail

cd "$(dirname "$0")/.."

for PRESET in TOPOKD2 TOPOKD3 TOPOKD1; do
    echo "========== Running $PRESET =========="
    python -m full_method.train --ablation "$PRESET" --epochs 100
    echo "========== $PRESET done =========="
    echo ""
done

echo "All TOPOKD experiments complete."

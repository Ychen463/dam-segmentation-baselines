#!/usr/bin/env bash
# Set A ablation: module-level cumulative (A2-A5)
# A0/A1 use baseline_segformer --preset A/B
set -euo pipefail
cd "$(dirname "$0")/.."
for preset in A2 A3 A4 A5; do
    echo "=== Ablation $preset ==="
    python -m full_method.train --ablation "$preset"
done

# Set S probes: soft curriculum fixes (S1-S5)
for preset in S1 S2 S3 S4 S5; do
    echo "=== Probe $preset ==="
    python -m full_method.train --ablation "$preset"
done

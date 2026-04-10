#!/usr/bin/env bash
# Set B ablation: difficulty score mechanism (B0-B5)
# All modules ON, only diff_* weights vary
set -euo pipefail
cd "$(dirname "$0")/.."
for preset in B0 B1 B2 B3 B4 B5; do
    echo "=== Ablation $preset ==="
    python -m full_method.train --ablation "$preset"
done

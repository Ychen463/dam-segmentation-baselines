#!/usr/bin/env bash
# Set A supplementary: class-aware sub-ablation (A3a, A3b, A3)
# Answers: does gain come from sampling bonus, loss scheduling, or both?
set -euo pipefail
cd "$(dirname "$0")/.."
for preset in A3a A3b A3; do
    echo "=== Ablation $preset ==="
    python -m full_method.train --ablation "$preset"
done

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== S1: soft curriculum only ==="
python -m full_method.train --ablation S1

echo "=== S2: soft curriculum + crack clDice ==="
python -m full_method.train --ablation S2

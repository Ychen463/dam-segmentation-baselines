#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== A3a: sampling bonus only (no loss schedule) ==="
python -m full_method.train --ablation A3a

echo "=== A3b: loss schedule only (no sampling bonus) ==="
python -m full_method.train --ablation A3b

#!/usr/bin/env bash
# 实验 F: 效率评测 (Params / FLOPs / Latency / FPS / Peak GPU Memory)
# 产出: results/efficiency.json
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
python -m shared_eval.efficiency --all-models --output-dir results/

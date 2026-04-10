#!/usr/bin/env bash
# 实验 A/B/C: 统一评估所有模型 (需先完成 01-06 训练)
# 产出: results/<model_name>_test.json
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
python -m shared_eval.eval_all --all-models --split test --per-tier --output-dir results/

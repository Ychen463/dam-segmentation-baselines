#!/usr/bin/env bash
# 实验 A/B/C: 统一评估所有模型 (需先完成 01-07 训练)
# 产出: results/<model_name>_test.json          (聚合指标)
#        results/<model_name>_test_per_image.csv (per-image, 用于统计检验)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
python -m shared_eval.eval_all --all-models --split test --per-tier --per-image --output-dir results/

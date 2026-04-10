#!/usr/bin/env bash
# 统计显著性检验: full method vs all baselines
# 前置: 08_eval_all.sh 已完成 (需要 per-image CSV)
# 产出: results/stats_significance_test.json
# 输出: paired Wilcoxon signed-rank test, bootstrap 95% CI, per tier
set -euo pipefail
cd "$(dirname "$0")/.."
python -m shared_eval.stats_significance --results-dir results/ --split test --output-dir results/

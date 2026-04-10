#!/usr/bin/env bash
# 生成/验证 train/val/test split (seed=42, stratified by difficulty x has_spalling)
# 产出: baseline_unet/splits/{train,val,test}.txt
# 幂等: 文件已存在时跳过，加 --force 强制重新生成
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "
from baseline_unet.splits import ensure_splits
import sys
force = '--force' in sys.argv
splits = ensure_splits(force=force)
for k, v in splits.items():
    print(f'  {k}: {len(v)} samples')
print('Splits OK.')
" "$@"

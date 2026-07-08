#!/bin/bash
# Train VMamba-Tiny baseline for crack/spalling segmentation
#
# Prerequisites:
#   pip install mamba-ssm causal-conv1d   # CUDA required for fast selective scan
#   # Falls back to pure PyTorch if not installed (very slow)
#
# Optional: download VMamba-Tiny pretrained weights:
#   wget -O vmamba_tiny_imagenet.pth \
#     https://github.com/MzeroMiko/VMamba/releases/download/v2cls/vssm_tiny_0230s_ckpt_epoch_264.pth
#   Then add: --pretrained-ckpt vmamba_tiny_imagenet.pth

set -e
cd "$(dirname "$0")/.."

SEEDS=(42 123 2024)
PRESET="${1:-T}"  # T=Tiny (default), S=Small

echo "=== VMamba baseline (preset=${PRESET}) ==="

for s in "${SEEDS[@]}"; do
  NAME="vmamba_${PRESET}_s${s}"
  if [ -f "baseline_vmamba/runs/${NAME}/best.pt" ]; then
    echo "[skip] ${NAME} already trained"
    continue
  fi
  echo "=== Training ${NAME} seed=${s} ==="
  python -m baseline_vmamba.train --preset "${PRESET}" --seed "$s" --name "${NAME}"
done

echo "=== Done. Check baseline_vmamba/runs/ for results ==="

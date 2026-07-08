#!/bin/bash
# Train DSConv vs DCNv2 vs StdConv (parameter-matched) + topology eval
set -e
cd "$(dirname "$0")/../full_method"

SEEDS=(42 123 2024)
RUNS=""

# --- Train all conditions ---
for s in "${SEEDS[@]}"; do
  # DSConv G0 (skip if checkpoint exists)
  if [ ! -f "runs/dsconv_G0_s${s}/best.pt" ]; then
    echo "=== Training G0 seed=$s ==="
    python train.py --preset G0 --seed "$s" --name "dsconv_G0_s${s}"
  fi
  RUNS="${RUNS}dsconv_G0_s${s},"

  # DCNv2 control
  if [ ! -f "runs/dcnv2_ctrl_s${s}/best.pt" ]; then
    echo "=== Training DCNV2_CTRL seed=$s ==="
    python train.py --preset DCNV2_CTRL --seed "$s" --name "dcnv2_ctrl_s${s}"
  fi
  RUNS="${RUNS}dcnv2_ctrl_s${s},"

  # StdConv control
  if [ ! -f "runs/stdconv_ctrl_s${s}/best.pt" ]; then
    echo "=== Training CONV_CTRL seed=$s ==="
    python train.py --preset CONV_CTRL --seed "$s" --name "stdconv_ctrl_s${s}"
  fi
  RUNS="${RUNS}stdconv_ctrl_s${s},"
done

# Remove trailing comma
RUNS="${RUNS%,}"

# --- Topology eval ---
cd ..
echo "=== Running topology eval ==="
python scripts/eval_topology.py --run-dirs "$RUNS"

echo "=== Done. Results in results/topology/ ==="

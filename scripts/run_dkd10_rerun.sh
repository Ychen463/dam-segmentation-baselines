#!/usr/bin/env bash
# Re-run DKD10 to recover val mIoU_fg (original metrics.csv was lost).
# Run on RunPod:
#   cd /workspace/Codes && bash scripts/run_dkd10_rerun.sh
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Re-running DKD10 (default DTKD, alpha=0.5, tau=4) ==="

# 1. Confirm teacher checkpoints exist
for ckpt in full_method/runs/dscformer_srl_G1/best.pt full_method/runs/sam_lora_srl_SAM2/best.pt; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: missing teacher checkpoint: $ckpt"
        exit 1
    fi
done
echo "Teacher checkpoints OK."

# 2. Train DKD10 (100 epochs)
python -m full_method.train --ablation DKD10 --name dkd10_no_srl_rerun

# 3. Test evaluation (plain, no TTA)
python -m full_method.eval_tta --run dkd10_no_srl_rerun --no-tta

# 4. Print the key values
echo ""
echo "=========================================="
echo "=== RESULTS ==="
echo "=========================================="
grep "best val mIoU_fg" full_method/runs/dkd10_no_srl_rerun/test_report.txt
grep "  mIoU_fg:" full_method/runs/dkd10_no_srl_rerun/test_report.txt
echo ""
echo "Copy these files back to local:"
echo "  full_method/runs/dkd10_no_srl_rerun/test_report.txt"
echo "  full_method/runs/dkd10_no_srl_rerun/metrics.csv"

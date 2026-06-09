#!/usr/bin/env bash
# Run DGACL Phase 2 experiments (resume from DKD10 checkpoint).
#
# Usage:
#   bash scripts/run_dgacl.sh          # run all 4 ablations
#   bash scripts/run_dgacl.sh DGACL1   # run specific variant
#
# Prerequisites:
#   - DKD10 best.pt at full_method/runs/dkd10_no_srl/best.pt
#   - Teacher checkpoints at full_method/runs/dscformer_srl_G1/best.pt
#                            and full_method/runs/sam_lora_srl_SAM2/best.pt

set -euo pipefail

DKD10_CKPT="full_method/runs/dkd10_no_srl/best.pt"

if [ ! -f "$DKD10_CKPT" ]; then
    echo "[ERROR] DKD10 checkpoint not found: $DKD10_CKPT"
    echo "  Run DKD10 first or adjust the path."
    exit 1
fi

# Step 0: Sanity check (teacher disagreement analysis)
if [ ! -f "results/dgacl/disagreement_analysis.json" ]; then
    echo "=== Running teacher disagreement sanity check ==="
    python scripts/teacher_disagreement_analysis.py
    echo ""
fi

# Determine which variants to run
if [ $# -gt 0 ]; then
    VARIANTS=("$@")
else
    VARIANTS=(DGACL1 DGACL2 DGACL3 DGACL4)
fi

for V in "${VARIANTS[@]}"; do
    echo ""
    echo "=============================================="
    echo "  DGACL Phase 2: $V"
    echo "=============================================="
    python -m full_method.train \
        --ablation "$V" \
        --resume "$DKD10_CKPT" \
        --epochs 30

    echo ""
    echo "--- Evaluating $V on test set ---"
    # Eval uses the best.pt from the run
    RUN_DIR="full_method/runs/$(python -c "
from full_method.config import ABLATION_PRESETS
print(ABLATION_PRESETS['$V']['name'])
")"
    if [ -f "$RUN_DIR/best.pt" ]; then
        python -m full_method.train \
            --ablation "$V" \
            --resume "$RUN_DIR/best.pt" \
            --dry-run
        echo "[eval] $V best.pt found at $RUN_DIR/best.pt"
    else
        echo "[warn] No best.pt for $V at $RUN_DIR"
    fi
done

echo ""
echo "=== All DGACL experiments complete ==="

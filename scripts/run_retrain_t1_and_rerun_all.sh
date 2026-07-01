#!/bin/bash
# Full pipeline: retrain T1, then rerun all DTKD experiments with matched checkpoint.
#
# Step 1: Retrain T1 (DSCformer+SRL, 100 epochs, seed 42)
#         - Saves to runs/dscformer_srl_G1_v2/ (preserves old checkpoint)
# Step 2: Copy new T1 best.pt to runs/dscformer_srl_G1/best.pt
# Step 3: Rerun matched-checkpoint controls (T1-only, dup T1, hetero equal-wt)
# Step 4: Rerun DKD10 (hetero class-conditional weights) — the final model
# Step 5: Evaluate all and save results
#
# Usage on RunPod:
#   cd /workspace/dam-segmentation-baselines
#   bash scripts/run_retrain_t1_and_rerun_all.sh
#
# Expected time: ~25min (T1) + 4×25min (students) = ~125 min on A40

set -e

echo "============================================================"
echo "  Full Pipeline: Retrain T1 + Rerun All DTKD Experiments"
echo "============================================================"

# ---- Step 1: Retrain T1 ----
echo ""
echo "[Step 1/5] Training Teacher 1 (DSCformer+SRL, 100 epochs)..."
echo "  Output: full_method/runs/dscformer_srl_G1_v2/"

python -m full_method.train --ablation G1 --seed 42 --name dscformer_srl_G1_v2

echo "  T1 training complete."

# Verify new T1
NEW_T1="full_method/runs/dscformer_srl_G1_v2/best.pt"
if [ ! -f "$NEW_T1" ]; then
    echo "ERROR: New T1 checkpoint not found: $NEW_T1"
    exit 1
fi
echo "  New T1 checkpoint: $NEW_T1"

# ---- Step 2: Backup old T1, install new T1 ----
echo ""
echo "[Step 2/5] Installing new T1 checkpoint..."
OLD_T1="full_method/runs/dscformer_srl_G1/best.pt"
if [ -f "$OLD_T1" ]; then
    cp "$OLD_T1" "full_method/runs/dscformer_srl_G1/best_old.pt"
    echo "  Old T1 backed up to best_old.pt"
fi
cp "$NEW_T1" "$OLD_T1"
echo "  New T1 installed at: $OLD_T1"
echo "  MD5: $(md5sum $OLD_T1 | cut -d' ' -f1)"

# Verify T2 exists
T2_CKPT="full_method/runs/sam_lora_srl_SAM2/best.pt"
if [ ! -f "$T2_CKPT" ]; then
    echo "ERROR: T2 checkpoint not found: $T2_CKPT"
    exit 1
fi
echo "  T2 checkpoint: $T2_CKPT (MD5: $(md5sum $T2_CKPT | cut -d' ' -f1))"

# ---- Step 3: Rerun matched-checkpoint controls ----
echo ""
echo "[Step 3/5] Running matched-checkpoint controls..."

echo "  [3a] T1-only KD..."
python -m full_method.train --ablation RERUN_T1ONLY --seed 42 --name rerun2_t1only
echo "  Done."

echo "  [3b] Duplicated T1..."
python -m full_method.train --ablation RERUN_DUP_T1 --seed 42 --name rerun2_dup_t1
echo "  Done."

echo "  [3c] Heterogeneous equal-weight..."
python -m full_method.train --ablation RERUN_HETERO --seed 42 --name rerun2_hetero
echo "  Done."

# ---- Step 4: Rerun DKD10 (class-conditional weights) ----
echo ""
echo "[Step 4/5] Running DKD10 (heterogeneous, class-conditional weights)..."
python -m full_method.train --ablation DKD10 --seed 42 --name rerun2_dkd10
echo "  Done."

# ---- Step 5: Evaluate all ----
echo ""
echo "[Step 5/5] Evaluating all checkpoints..."
python scripts/eval_retrain_t1_rerun.py

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Results: results/retrain_t1_rerun.json"
echo "============================================================"

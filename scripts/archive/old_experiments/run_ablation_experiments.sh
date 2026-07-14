#!/usr/bin/env bash
# New ablation experiments for reviewer concerns (Issues 2, 6, 7).
# Run on RunPod after git pull.
set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODES_DIR"

echo "=== Reviewer Ablation Experiments ==="

# ── Issue 2: Single-teacher KD ablation ──────────────────────────────────
# Proves dual-teacher necessity by testing each teacher alone.

echo ""
echo "=== Issue 2: Single-Teacher Ablations ==="

echo "--- KD_T1: Teacher 1 only (DSCFormer-SRL G1) ---"
python -m full_method.train --ablation KD_T1 \
    --name kd_t1only_KD_T1

echo "--- KD_T2: Teacher 2 only (SAM2-LoRA-SRL) ---"
python -m full_method.train --ablation KD_T2 \
    --name kd_t2only_KD_T2

# ── Issue 6: Parameter-matched standard Conv control ─────────────────────
# Tests if DSConv's geometric bias (vs just extra capacity) drives gains.

echo ""
echo "=== Issue 6: Standard Conv Control ==="

echo "--- CONV_CTRL: Standard Conv branch (~1.3M params, no DSConv) ---"
python -m full_method.train --ablation CONV_CTRL \
    --name segformer_stdconv_CONV_CTRL

# ── Issue 7: DSConv-only (G0) cross-dataset eval on S2DS ────────────────
# Isolates DSConv contribution to cross-dataset generalization.
# Only runs if G0 checkpoint exists.

echo ""
echo "=== Issue 7: G0 Cross-Dataset Eval ==="

G0_CKPT="runs/dscformer_plain_G0/best.pt"
if [ -f "$G0_CKPT" ]; then
    echo "--- G0 on S2DS (eval only) ---"
    python -m full_method.evaluate \
        --checkpoint "$G0_CKPT" \
        --test-dir data/S2DS/test \
        --name dscformer_G0_s2ds_eval
else
    echo "[SKIP] G0 checkpoint not found at $G0_CKPT"
    echo "  Train G0 first: python -m full_method.train --ablation G0 --name dscformer_plain_G0"
fi

echo ""
echo "=== All reviewer ablation experiments complete ==="

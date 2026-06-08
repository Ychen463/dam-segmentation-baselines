#!/usr/bin/env bash
# Step-by-step DKD optimization: train ONE experiment, evaluate, show comparison.
#
# Usage:
#   bash scripts/run_dkd_step.sh DKD10    # Train + eval DKD10
#   bash scripts/run_dkd_step.sh DKD11    # Train + eval DKD11
#   bash scripts/run_dkd_step.sh DKD8     # etc.
#
# After each run, inspect the comparison table and decide the next step:
#
#   Step 1: bash scripts/run_dkd_step.sh DKD10
#           → Compare DKD10 (no SRL) vs DKD2 (with SRL)
#           → If DKD10 wins: SRL is harmful, proceed WITHOUT SRL
#           → If DKD2 wins:  SRL helps, try tuning SRL weight/timing
#
#   Step 2 (if DKD10 wins):
#           bash scripts/run_dkd_step.sh DKD8   (better teacher)
#           bash scripts/run_dkd_step.sh DKD9   (moderate aug)
#           bash scripts/run_dkd_step.sh DKD7   (+ curriculum)
#
#   Step 2 (if DKD2 wins):
#           bash scripts/run_dkd_step.sh DKD11  (weaker SRL w=0.02)
#           bash scripts/run_dkd_step.sh DKD12  (later SRL ep=80)
#
# Available presets: DKD7 DKD8 DKD9 DKD10 DKD11 DKD12

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_dkd_step.sh <PRESET>"
    echo "  Available: DKD7 DKD8 DKD9 DKD10 DKD11 DKD12"
    exit 1
fi

PRESET="$1"

# Map preset to run name
declare -A PRESET_NAMES=(
    [DKD7]=dkd7_curriculum
    [DKD8]=dkd8_teacher_g0
    [DKD9]=dkd9_modaug
    [DKD10]=dkd10_no_srl
    [DKD11]=dkd11_weak_srl
    [DKD12]=dkd12_late_srl
    [DKD13]=dkd13_g0_no_srl
    [DKD14]=dkd14_equal_weight
)

NAME="${PRESET_NAMES[$PRESET]:-}"
if [[ -z "$NAME" ]]; then
    echo "ERROR: Unknown preset '$PRESET'"
    echo "  Available: DKD7 DKD8 DKD9 DKD10 DKD11 DKD12 DKD13 DKD14"
    exit 1
fi

CKPT="full_method/runs/${NAME}/best.pt"

# ============================================================================
# Step 1: Train
# ============================================================================

if [[ -f "$CKPT" ]]; then
    echo "[${PRESET}] Already trained (${CKPT} exists), skipping to eval."
else
    echo ""
    echo "=========================================="
    echo "  Training ${PRESET} (${NAME})"
    echo "=========================================="

    mkdir -p logs
    python -m full_method.train \
        --ablation "$PRESET" \
        --name "$NAME" \
        2>&1 | tee "logs/${PRESET}.log"
fi

# ============================================================================
# Step 2: Evaluate (no TTA + TTA)
# ============================================================================

if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] Training failed — no checkpoint at ${CKPT}"
    exit 1
fi

echo ""
echo "=== Evaluating ${NAME} ==="

python -m full_method.eval_tta \
    --run "$NAME" \
    --ablation "$PRESET" \
    --no-tta \
    2>&1 || echo "[WARN] eval failed"

python -m full_method.eval_tta \
    --run "$NAME" \
    --ablation "$PRESET" \
    2>&1 || echo "[WARN] TTA eval failed"

# ============================================================================
# Step 3: Comparison table
# ============================================================================

echo ""
echo "=========================================="
echo "  Results: ${PRESET} vs baselines"
echo "=========================================="

python3 -c "
import json, os

models = [
    ('G0 (DSConv, no SRL)',  'results/dscformer_plain_G0_test.json'),
    ('G1 (DSConv+SRL)',      'results/dscformer_srl_G1_test.json'),
    ('DKD2 (current best)',  'results/dual_kd_classaware_DKD2_test.json'),
    ('${NAME}',              'results/${NAME}_test.json'),
]

print()
print(f\"{'Model':<25} {'mIoU_fg':>8} {'IoU_cr':>8} {'IoU_sp':>8} {'BF1_fg':>8} {'clDice':>8} {'ConnR_fg':>9} {'ConnR_sp':>9}\")
print('=' * 95)

best = {}
for name, path in models:
    if not os.path.exists(path):
        print(f'{name:<25} (not found)')
        continue
    d = json.load(open(path))['overall']
    row = {
        'mIoU_fg': d['mIoU_fg']*100,
        'IoU_cr': d['IoU_crack']*100,
        'IoU_sp': d['IoU_spalling']*100,
        'BF1_fg': d['BF1_fg_mean']*100,
        'clDice': d.get('clDice_fg_mean',0)*100,
        'ConnR_fg': d.get('ConnR_fg_mean',0)*100,
        'ConnR_sp': d.get('ConnR_spalling',0)*100,
    }
    print(f\"{name:<25} {row['mIoU_fg']:>7.1f}% {row['IoU_cr']:>7.1f}% {row['IoU_sp']:>7.1f}% {row['BF1_fg']:>7.1f}% {row['clDice']:>7.1f}% {row['ConnR_fg']:>8.1f}% {row['ConnR_sp']:>8.1f}%\")
    best[name] = row

# Delta vs DKD2
new_name = '${NAME}'
if new_name in best and 'DKD2 (current best)' in best:
    print()
    print(f'Delta ({new_name} - DKD2):')
    b = best['DKD2 (current best)']
    n = best[new_name]
    parts = []
    for k in ['mIoU_fg', 'IoU_cr', 'IoU_sp', 'BF1_fg', 'clDice', 'ConnR_fg', 'ConnR_sp']:
        d = n[k] - b[k]
        flag = '✓' if d > 0.3 else ('✗' if d < -0.3 else '~')
        parts.append(f'  {k}: {d:+.1f} {flag}')
    print('\n'.join(parts))

print()
print('Decision guide:')
if '${PRESET}' == 'DKD10':
    print('  If DKD10 > DKD2 on mIoU AND ConnR: SRL is harmful → next try DKD8')
    print('  If DKD10 < DKD2 on mIoU:           SRL helps → next try DKD11 or DKD12')
    print('  If DKD10 ≈ DKD2 on mIoU but ConnR↑: SRL is a wash → next try DKD8 (no SRL base)')
elif '${PRESET}' in ('DKD11', 'DKD12'):
    print('  Compare with DKD2: did tuning SRL reduce ConnR damage while keeping BF1?')
    print('  Pick the best SRL setting, then try DKD7/DKD8/DKD9 on top of it.')
elif '${PRESET}' == 'DKD8':
    print('  If ConnR_fg improved: G0 is a better Teacher 1.')
    print('  Next: try DKD7 (curriculum) or DKD9 (aug) on this base.')
else:
    print('  Compare all metrics. Pick the best config as the new baseline.')
"

echo ""
echo "=== Done. Decide next step based on the table above. ==="

"""Quick check of CKD experiment val results."""
import csv
import os

NAMES = [
    "ckd_b0_t1_baseline",
    "ckd_c0_gt_only",
    "ckd_c3_t2_rescue",
    "ckd_c2_dual_equal",
    "ckd_c4_t2_rescue_low",
    "ckd_c1_t1_anchor",
]

for name in NAMES:
    print(f"=== {name} ===")
    p = f"full_method/runs/{name}/metrics.csv"
    if not os.path.exists(p):
        print("  (no metrics.csv)")
        continue
    with open(p) as f:
        rows = [r for r in csv.DictReader(f) if r.get("mIoU_fg")]
    if not rows:
        print("  (no val rows)")
        continue
    best = max(rows, key=lambda r: float(r["mIoU_fg"]))
    last = rows[-1]
    print(f'  best: ep={best["epoch"]} mIoU_fg={best["mIoU_fg"]} IoU_cr={best["IoU_crack"]} IoU_sp={best["IoU_spalling"]} BF1_cr={best["BF1_crack"]}')
    print(f'  last: ep={last["epoch"]} mIoU_fg={last["mIoU_fg"]} IoU_cr={last["IoU_crack"]} IoU_sp={last["IoU_spalling"]} BF1_cr={last["BF1_crack"]}')

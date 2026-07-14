"""Multi-model logit ensemble evaluation.

Loads multiple trained models, averages their logits at test time, and
evaluates the ensemble to determine the ceiling of model combination.

Usage:
    python scripts/eval_ensemble.py
    python scripts/eval_ensemble.py --runs dscformer_srl_G1 fp1_tversky_precision fp2_tversky_balanced
    python scripts/eval_ensemble.py --tta
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from full_method import config as C
from full_method.eval_tta import load_model, tta_predict, plain_predict
from full_method.train import build_val_loader

try:
    from shared_eval.metrics_full import SegMetricsFull as MetricsClass
except ImportError:
    from baseline_deeplab.metrics import SegMetricsBF1 as MetricsClass

METRIC_KEYS = [
    "IoU_crack", "IoU_spalling", "mIoU_fg", "mIoU_all",
    "BF1_crack", "BF1_spalling", "BF1_fg_mean",
    "clDice_crack", "ConnR_crack",
    "clDice_spalling", "ConnR_spalling",
]

# Default models to ensemble
DEFAULT_RUNS = [
    "dscformer_srl_G1",           # Teacher T1
    "dual_kd_classaware_DKD2",    # Dual-KD student
    "fp1_tversky_precision",      # Tversky precision
    "fp2_tversky_balanced",       # Tversky balanced
]


def collect_logits(model, loader, device, use_amp, use_tta, tta_scales):
    """Run inference, return list of (logits, masks) batches."""
    all_logits = []
    all_masks = []
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True).float()
        masks_gt = batch["mask"].to(device, non_blocking=True).long()
        target_size = masks_gt.shape[-2:]

        if use_tta:
            logits = tta_predict(model, imgs, tta_scales, use_flip=True,
                                 target_size=target_size, device=device,
                                 use_amp=use_amp)
        else:
            logits = plain_predict(model, imgs, target_size, device,
                                   use_amp=use_amp)
        all_logits.append(logits.cpu())
        all_masks.append(masks_gt.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_masks, dim=0)


def evaluate(logits, masks):
    """Compute metrics from logits and GT masks."""
    metrics = MetricsClass(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    metrics.reset()
    # Process in chunks to avoid memory issues
    bs = 8
    for i in range(0, len(logits), bs):
        metrics.update(logits[i:i+bs], masks[i:i+bs])
    return metrics.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                        help="Run directory names to ensemble")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-scales", type=float, nargs="+",
                        default=[0.75, 1.0, 1.25])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    # Verify all runs exist
    run_dirs = {}
    for name in args.runs:
        rd = C.RUNS_DIR / name
        if not rd.exists():
            print(f"[WARN] {name} not found, skipping")
            continue
        run_dirs[name] = rd

    if len(run_dirs) < 2:
        print("[ERROR] Need at least 2 valid runs for ensemble")
        sys.exit(1)

    # Build test loader
    cfg = C.RunCfg()
    cfg.batch_size = 1
    with open(C.SPLIT_FILES["test"]) as f:
        test_files = [line.strip() for line in f if line.strip()]
    test_loader = build_val_loader(test_files, cfg, device)

    # Collect logits from each model
    all_model_logits = {}
    masks = None

    for name, rd in run_dirs.items():
        print(f"\n[ensemble] Loading {name}...")
        model = load_model(rd, device)
        print(f"[ensemble] Running inference for {name}...")
        t0 = time.time()
        logits, gt_masks = collect_logits(model, test_loader, device, use_amp,
                                          args.tta, args.tta_scales)
        print(f"[ensemble] {name} done in {time.time()-t0:.1f}s")
        all_model_logits[name] = logits
        if masks is None:
            masks = gt_masks
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    names = list(all_model_logits.keys())
    print(f"\n[ensemble] {len(names)} models: {names}")

    # Evaluate individual models first
    results = {}
    for name in names:
        m = evaluate(all_model_logits[name], masks)
        results[name] = m

    # Evaluate all pairwise and higher-order ensembles
    for k in range(2, len(names) + 1):
        for combo in itertools.combinations(names, k):
            ens_name = " + ".join(combo)
            avg_logits = sum(all_model_logits[n] for n in combo) / k
            m = evaluate(avg_logits, masks)
            results[ens_name] = m

    # Print results
    print(f"\n{'Model/Ensemble':<55s} {'CrIoU':>6s} {'SpIoU':>6s} {'mIoU':>6s} "
          f"{'BF1cr':>6s} {'BF1sp':>6s} {'clDcr':>6s} {'ConR':>6s}")
    print("-" * 105)

    # Find best for highlighting
    best_miou = max(r.get("mIoU_fg", 0) for r in results.values())

    for name, m in results.items():
        ci = m.get("IoU_crack", 0) * 100
        si = m.get("IoU_spalling", 0) * 100
        mf = m.get("mIoU_fg", 0) * 100
        bf1c = m.get("BF1_crack", 0) * 100
        bf1s = m.get("BF1_spalling", 0) * 100
        cld = m.get("clDice_crack", 0) * 100
        cnr = m.get("ConnR_crack", 0) * 100
        marker = " ***" if abs(mf - best_miou * 100) < 0.01 else ""
        print(f"{name:<55s} {ci:>6.2f} {si:>6.2f} {mf:>6.2f} "
              f"{bf1c:>6.2f} {bf1s:>6.2f} {cld:>6.2f} {cnr:>6.2f}{marker}")

    # Save
    out_path = Path("results/dgacl/ensemble_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for name, m in results.items():
        save_data[name] = {k: round(v, 6) for k, v in m.items() if isinstance(v, float)}
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[ensemble] Saved to {out_path}")


if __name__ == "__main__":
    main()

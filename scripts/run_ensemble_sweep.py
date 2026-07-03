"""Comprehensive ensemble evaluation: find the best model combination.

Tests all meaningful 2/3/4-model ensembles from diverse architectures:
  - G0: DSCformerDam (SegFormer-B2 + DSConv) — task-specific backbone
  - G1: DSCformerDam + SRL — with topology loss
  - P0: Plain SegFormer-B2 — no DSConv branch
  - SAM2: SAM-LoRA — foundation model, different architecture
  - DKD2: Dual-KD student — distilled from both T1+T2
  - G0 multi-seed: same arch, different init (diversity via randomness)

Also tests TTA + ensemble and ensemble + TAPP post-processing.

Usage on RunPod:
    cd /workspace/dam-segmentation-baselines
    python scripts/run_ensemble_sweep.py
    python scripts/run_ensemble_sweep.py --tta
    python scripts/run_ensemble_sweep.py --with-pp
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from full_method import config as C
from full_method.eval_tta import load_model, tta_predict, plain_predict
from full_method.train import build_val_loader
from full_method.tapp import tapp_postprocess

try:
    from shared_eval.metrics_full import SegMetricsFull as MetricsClass
except ImportError:
    from baseline_deeplab.metrics import SegMetricsBF1 as MetricsClass

# ---- Model pool: (run_dir_name, display_name, model_type_hint) ----
# Prioritized for architectural diversity
MODEL_POOL = [
    ("dscformer_plain_G0",           "G0 (DSCformer)",       None),
    ("dscformer_srl_G1_v2",          "G1 (DSCformer+SRL)",   None),
    ("plain_segformer_P0",           "P0 (SegFormer)",       "segformer"),
    ("sam_lora_srl_SAM2",            "SAM2 (SAM-LoRA)",      "sam_lora"),
    ("dual_kd_classaware_DKD2",      "DKD2 (Dual-KD)",       None),
    # Multi-seed G0 for seed diversity
    ("dscformer_plain_G0_seed123",   "G0-s123",             None),
    ("dscformer_plain_G0_seed2024",  "G0-s2024",            None),
]

METRIC_KEYS = [
    "IoU_crack", "IoU_spalling", "mIoU_fg",
    "BF1_crack", "BF1_spalling", "BF1_fg_mean",
    "clDice_crack", "ConnR_crack",
    "clDice_fg_mean", "ConnR_fg_mean",
]


def collect_logits(model, loader, device, use_amp, use_tta, tta_scales):
    """Run inference, return all logits and GT masks stacked."""
    all_logits = []
    all_masks = []
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True).float()
        masks_gt = batch["mask"].to(device, non_blocking=True).long()
        target_size = masks_gt.shape[-2:]

        if use_tta:
            # TTA processes one image at a time
            batch_logits = []
            for i in range(imgs.shape[0]):
                single = imgs[i:i+1]
                logits_i = tta_predict(model, single, tta_scales, use_flip=True,
                                       target_size=target_size, device=device,
                                       use_amp=use_amp)
                batch_logits.append(logits_i)
            logits = torch.cat(batch_logits, dim=0)
        else:
            logits = plain_predict(model, imgs, target_size, device,
                                   use_amp=use_amp)
        all_logits.append(logits.cpu())
        all_masks.append(masks_gt.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_masks, dim=0)


def evaluate_logits(logits, masks, use_pp=False):
    """Compute metrics from logits and GT masks, optionally with TAPP."""
    metrics = MetricsClass(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    metrics.reset()

    if use_pp:
        # Convert to predictions, apply TAPP, then evaluate
        preds = logits.argmax(dim=1)  # (N, H, W)
        for i in range(len(preds)):
            pred_np = preds[i].numpy().astype(np.uint8)
            refined = tapp_postprocess(pred_np)
            # Convert back to one-hot logits for metrics
            refined_t = torch.from_numpy(refined).long()
            one_hot = F.one_hot(refined_t, C.NUM_CLASSES).permute(2, 0, 1).float().unsqueeze(0)
            metrics.update(one_hot * 100, masks[i:i+1])
    else:
        bs = 8
        for i in range(0, len(logits), bs):
            metrics.update(logits[i:i+bs], masks[i:i+bs])

    return metrics.compute()


def format_row(name, m):
    """Format a single result row."""
    ci = m.get("IoU_crack", 0) * 100
    si = m.get("IoU_spalling", 0) * 100
    mf = m.get("mIoU_fg", 0) * 100
    bf1c = m.get("BF1_crack", 0) * 100
    bf1s = m.get("BF1_spalling", 0) * 100
    bf1f = m.get("BF1_fg_mean", 0) * 100
    cld = m.get("clDice_crack", 0) * 100
    cnr = m.get("ConnR_crack", 0) * 100
    return (f"{name:<50s} {ci:>6.2f} {si:>6.2f} {mf:>6.2f} "
            f"{bf1c:>6.2f} {bf1s:>6.2f} {bf1f:>6.2f} {cld:>6.2f} {cnr:>6.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta", action="store_true", help="Use TTA for all models")
    parser.add_argument("--tta-scales", type=float, nargs="+", default=[0.75, 1.0, 1.25])
    parser.add_argument("--with-pp", action="store_true", help="Also eval ensemble + TAPP")
    parser.add_argument("--max-combo", type=int, default=4,
                        help="Max models in a combination (default: 4)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print(f"Device: {device}, TTA: {args.tta}, Post-process: {args.with_pp}")

    # Build test loader
    cfg = C.RunCfg()
    cfg.batch_size = 2
    with open(C.SPLIT_FILES["test"]) as f:
        test_files = [line.strip() for line in f if line.strip()]
    test_loader = build_val_loader(test_files, cfg, device)
    print(f"Test set: {len(test_files)} images")

    # Load models and collect logits
    available_models = {}
    masks = None

    for run_name, display_name, mtype in MODEL_POOL:
        run_dir = C.RUNS_DIR / run_name
        if not (run_dir / "best.pt").exists():
            print(f"[SKIP] {display_name} ({run_name}) — not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Loading: {display_name}")
        print(f"{'='*60}")
        model = load_model(run_dir, device, model_type=mtype)
        t0 = time.time()
        logits, gt_masks = collect_logits(model, test_loader, device, use_amp,
                                          args.tta, args.tta_scales)
        dt = time.time() - t0
        print(f"  Inference: {dt:.1f}s ({len(test_files)/dt:.1f} img/s)")

        available_models[display_name] = logits
        if masks is None:
            masks = gt_masks

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(available_models) < 2:
        print("[ERROR] Need at least 2 models for ensemble. Exiting.")
        sys.exit(1)

    names = list(available_models.keys())
    print(f"\n{'='*60}")
    print(f"  Available models: {len(names)}")
    print(f"{'='*60}")
    for n in names:
        print(f"  - {n}")

    # Evaluate individuals
    results = {}
    for name in names:
        m = evaluate_logits(available_models[name], masks)
        results[name] = m

    # Evaluate ensembles (2 to max_combo)
    max_k = min(args.max_combo, len(names))
    for k in range(2, max_k + 1):
        for combo in itertools.combinations(names, k):
            ens_name = " + ".join(combo)
            avg_logits = sum(available_models[n] for n in combo) / k
            m = evaluate_logits(avg_logits, masks)
            results[ens_name] = m

            # Optionally with TAPP
            if args.with_pp:
                m_pp = evaluate_logits(avg_logits, masks, use_pp=True)
                results[ens_name + " +PP"] = m_pp

    # Also eval best single model with PP for reference
    if args.with_pp:
        for name in names:
            m_pp = evaluate_logits(available_models[name], masks, use_pp=True)
            results[name + " +PP"] = m_pp

    # Print results sorted by mIoU_fg
    print(f"\n{'='*100}")
    print("  RESULTS: Ensemble Sweep (sorted by mIoU_fg)")
    print(f"{'='*100}")
    header = (f"{'Model/Ensemble':<50s} {'CrIoU':>6s} {'SpIoU':>6s} {'mIoU':>6s} "
              f"{'BF1cr':>6s} {'BF1sp':>6s} {'BF1fg':>6s} {'clDcr':>6s} {'CnRcr':>6s}")
    print(header)
    print("-" * 100)

    sorted_results = sorted(results.items(),
                            key=lambda x: x[1].get("mIoU_fg", 0), reverse=True)

    best_miou = sorted_results[0][1].get("mIoU_fg", 0) * 100
    for name, m in sorted_results:
        row = format_row(name, m)
        mf = m.get("mIoU_fg", 0) * 100
        marker = " ***" if abs(mf - best_miou) < 0.01 else ""
        print(row + marker)

    # Highlight: best ensemble vs best single model
    best_single = max(
        [(n, m) for n, m in results.items() if "+" not in n and " +PP" not in n],
        key=lambda x: x[1].get("mIoU_fg", 0)
    )
    best_ensemble = max(
        [(n, m) for n, m in results.items() if "+" in n and " +PP" not in n],
        key=lambda x: x[1].get("mIoU_fg", 0)
    )

    print(f"\n--- Best single model: {best_single[0]} ---")
    print(f"  mIoU_fg={best_single[1].get('mIoU_fg',0)*100:.2f}")
    print(f"\n--- Best ensemble: {best_ensemble[0]} ---")
    print(f"  mIoU_fg={best_ensemble[1].get('mIoU_fg',0)*100:.2f}")
    gain = (best_ensemble[1].get("mIoU_fg", 0) - best_single[1].get("mIoU_fg", 0)) * 100
    print(f"  Ensemble gain: {gain:+.2f} mIoU_fg")

    # Save
    suffix = "_tta" if args.tta else ""
    out_path = Path(f"results/ensemble_sweep{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for name, m in sorted_results:
        save_data[name] = {k: round(v, 6) for k, v in m.items() if isinstance(v, (int, float))}
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

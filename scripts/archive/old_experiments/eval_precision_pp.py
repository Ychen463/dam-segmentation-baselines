"""Evaluate precision-focused post-processing on existing checkpoints.

Tests multiple FP-reduction strategies on the test set WITHOUT retraining:
  1. Baseline (no PP)
  2. MSF only (existing TAPP: remove small/round blobs)
  3. CBE only (crack boundary erosion: thin the predictions)
  4. CBE + MSF (erosion then blob removal)
  5. Confidence threshold sweep (P(crack) > 0.5/0.6/0.7 instead of argmax)

Reports Crack IoU, Precision, Recall, BF1, and clDice for each variant.

Usage:
    python scripts/eval_precision_pp.py
    python scripts/eval_precision_pp.py --run dkd10_no_srl
    python scripts/eval_precision_pp.py --run dscformer_srl_G1 --tta
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from full_method import config as C
from full_method.eval_tta import load_model, tta_predict
from full_method.train import build_val_loader
from full_method.tapp import tapp_postprocess

try:
    from shared_eval.metrics_full import SegMetricsFull as MetricsClass
except ImportError:
    from baseline_deeplab.metrics import SegMetricsBF1 as MetricsClass


REPORT_KEYS = [
    "IoU_crack", "IoU_spalling", "mIoU_fg",
    "BF1_crack", "BF1_spalling", "BF1_fg_mean",
    "clDice_crack", "ConnR_crack",
]


def run_baseline(model, loader, device, use_amp, use_tta, tta_scales):
    """Run inference, return raw logits list and GT masks."""
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
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                out = model(imgs)
                logits = F.interpolate(out["seg_logits"].float(),
                                       size=target_size, mode="bilinear",
                                       align_corners=False)
        all_logits.append(logits.cpu())
        all_masks.append(masks_gt.cpu())
    return all_logits, all_masks


def eval_from_preds(pred_tensor, masks_gt):
    """Evaluate pre-computed predictions against GT."""
    metrics = MetricsClass(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    metrics.reset()
    one_hot = F.one_hot(pred_tensor.long(), C.NUM_CLASSES).permute(0, 3, 1, 2).float()
    metrics.update(one_hot * 100, masks_gt)
    return metrics.compute()


def eval_from_logits(all_logits, all_masks):
    """Evaluate with argmax (baseline)."""
    metrics = MetricsClass(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    metrics.reset()
    for logits, masks in zip(all_logits, all_masks):
        metrics.update(logits, masks)
    return metrics.compute()


def apply_threshold(logits, threshold, crack_class=1):
    """Apply confidence threshold: only predict crack if P(crack) > threshold."""
    probs = F.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    # Override: if argmax says crack but confidence < threshold, use 2nd-best
    crack_prob = probs[:, crack_class]
    low_conf = (pred == crack_class) & (crack_prob < threshold)
    if low_conf.any():
        # Set crack logit to -inf and re-argmax
        masked_logits = logits.clone()
        masked_logits[:, crack_class][low_conf] = -1e6
        alt_pred = masked_logits.argmax(dim=1)
        pred[low_conf] = alt_pred[low_conf]
    return pred


def apply_pp(pred_np, use_cbe=False, use_msf=False, cbe_radius=1, msf_min_area=30):
    """Apply post-processing to numpy prediction."""
    result = pred_np.copy()
    for b in range(result.shape[0]):
        result[b] = tapp_postprocess(
            result[b],
            use_cbe=use_cbe, cbe_erosion_radius=cbe_radius,
            use_msf=use_msf, msf_min_area=msf_min_area,
            use_sgf=False,
        )
    return result


def compute_crack_pr(pred, gt, crack_class=1):
    """Quick crack precision/recall from pred and gt tensors."""
    gt_c = (gt == crack_class)
    pred_c = (pred == crack_class)
    tp = (gt_c & pred_c).sum().item()
    fp = (~gt_c & pred_c).sum().item()
    fn = (gt_c & ~pred_c).sum().item()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    return p, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="dscformer_srl_G1",
                        help="Run directory name")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-scales", type=float, nargs="+",
                        default=[0.75, 1.0, 1.25])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = C.RUNS_DIR / args.run
    if not run_dir.exists():
        print(f"[ERROR] {run_dir} not found")
        sys.exit(1)

    print(f"[eval-pp] Loading model from {run_dir}")
    model = load_model(run_dir, device)

    cfg = C.RunCfg()
    cfg.batch_size = 4
    test_files = []
    with open(C.SPLIT_FILES["test"]) as f:
        test_files = [line.strip() for line in f if line.strip()]
    test_loader = build_val_loader(test_files, cfg, device)

    use_amp = (device == "cuda")

    # Step 1: Collect all logits
    print("[eval-pp] Running inference ...")
    t0 = time.time()
    all_logits, all_masks = run_baseline(model, test_loader, device, use_amp,
                                          args.tta, args.tta_scales)
    print(f"[eval-pp] Inference done in {time.time()-t0:.1f}s")

    # Concat for easy manipulation
    logits_cat = torch.cat(all_logits, dim=0)
    masks_cat = torch.cat(all_masks, dim=0)
    base_pred = logits_cat.argmax(dim=1)

    # Define variants
    variants = []

    # 1. Baseline
    variants.append(("Baseline", base_pred))

    # 2. MSF only
    pp_msf = apply_pp(base_pred.numpy(), use_msf=True)
    variants.append(("MSF (min_area=30)", torch.from_numpy(pp_msf)))

    # 3. MSF aggressive
    pp_msf_agg = apply_pp(base_pred.numpy(), use_msf=True, msf_min_area=50)
    variants.append(("MSF (min_area=50)", torch.from_numpy(pp_msf_agg)))

    # 4. CBE only (1px erosion)
    pp_cbe1 = apply_pp(base_pred.numpy(), use_cbe=True, cbe_radius=1)
    variants.append(("CBE (r=1)", torch.from_numpy(pp_cbe1)))

    # 5. CBE (2px)
    pp_cbe2 = apply_pp(base_pred.numpy(), use_cbe=True, cbe_radius=2)
    variants.append(("CBE (r=2)", torch.from_numpy(pp_cbe2)))

    # 6. CBE + MSF
    pp_combo = apply_pp(base_pred.numpy(), use_cbe=True, cbe_radius=1, use_msf=True)
    variants.append(("CBE(1)+MSF(30)", torch.from_numpy(pp_combo)))

    # 7. CBE + MSF aggressive
    pp_combo2 = apply_pp(base_pred.numpy(), use_cbe=True, cbe_radius=1,
                          use_msf=True, msf_min_area=50)
    variants.append(("CBE(1)+MSF(50)", torch.from_numpy(pp_combo2)))

    # 8-10. Confidence thresholds
    for thresh in [0.55, 0.60, 0.65]:
        pred_t = apply_threshold(logits_cat, thresh)
        variants.append((f"Threshold>{thresh:.2f}", pred_t))

    # 11-12. Threshold + PP
    for thresh in [0.55, 0.60]:
        pred_t = apply_threshold(logits_cat, thresh)
        pp_t = apply_pp(pred_t.numpy(), use_cbe=True, cbe_radius=1, use_msf=True)
        variants.append((f"Thr>{thresh:.2f}+CBE+MSF", torch.from_numpy(pp_t)))

    # Evaluate all variants
    results = {}
    print(f"\n{'Variant':<22s} {'CrackIoU':>9s} {'Prec':>7s} {'Recall':>7s} "
          f"{'mIoU_fg':>8s} {'BF1_cr':>7s} {'BF1_sp':>7s} {'clD_cr':>7s} {'ConR':>7s}")
    print("-" * 98)

    for name, pred in variants:
        # Precision/recall
        p, r = compute_crack_pr(pred, masks_cat)

        # Full metrics
        m = eval_from_preds(pred, masks_cat)
        results[name] = m

        ci = m.get("IoU_crack", 0) * 100
        si = m.get("IoU_spalling", 0) * 100
        mf = m.get("mIoU_fg", 0) * 100
        bf1c = m.get("BF1_crack", 0) * 100
        bf1s = m.get("BF1_spalling", 0) * 100
        cld = m.get("clDice_crack", 0) * 100
        cnr = m.get("ConnR_crack", 0) * 100

        # Delta vs baseline
        if name == "Baseline":
            base_ci = ci
            delta = ""
        else:
            delta = f" ({ci - base_ci:+.2f})"

        print(f"{name:<22s} {ci:>8.2f}{delta:>7s} {p*100:>6.2f} {r*100:>6.2f} "
              f"{mf:>8.2f} {bf1c:>7.2f} {bf1s:>7.2f} {cld:>7.2f} {cnr:>7.2f}")

    # Save
    import json
    out_path = Path("results/dgacl/precision_pp_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for name, m in results.items():
        save_data[name] = {k: round(v, 6) for k, v in m.items() if isinstance(v, float)}
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[eval-pp] Saved to {out_path}")


if __name__ == "__main__":
    main()

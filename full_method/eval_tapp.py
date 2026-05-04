"""Evaluate with TAPP (Topology-Aware Post-Processing Pipeline).

Runs model inference (optionally with TTA), then applies TAPP post-processing
(MSF + SGF) and reports metrics. Also runs ablations: no post-proc, MSF only,
SGF only, MSF+SGF.

Usage:
    # TAPP only (no TTA), on G1
    python eval_tapp.py --run dscformer_srl_G1

    # TTA + TAPP combined
    python eval_tapp.py --run dscformer_srl_G1 --tta

    # Ablation: MSF only
    python eval_tapp.py --run dscformer_srl_G1 --no-sgf

    # Ablation: SGF only
    python eval_tapp.py --run dscformer_srl_G1 --no-msf
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
from full_method.model import DSCformerDam, SegFormerWithBoundary
from full_method.train import build_val_loader
from full_method.tapp import tapp_postprocess
from full_method.eval_tta import load_model, tta_predict


METRIC_KEYS = (
    "IoU_background", "IoU_crack", "IoU_spalling",
    "Dice_background", "Dice_crack", "Dice_spalling",
    "mIoU_fg", "mIoU_all", "pixel_acc",
    "BF1_crack", "BF1_spalling", "BF1_fg_mean",
    "clDice_crack", "clDice_spalling", "clDice_fg_mean",
    "ConnR_crack", "ConnR_spalling", "ConnR_fg_mean",
)


def run_eval(model, test_loader, device, use_amp, use_tta, tta_scales,
             use_msf, use_sgf, msf_kwargs, sgf_kwargs):
    """Run inference + optional TTA + optional TAPP, return metrics dict."""
    try:
        from shared_eval.metrics_full import SegMetricsFull
        metrics = SegMetricsFull(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    except ImportError:
        from full_method.train import SegMetricsBF1
        metrics = SegMetricsBF1(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)

    metrics.reset()
    n_filtered = 0
    n_bridges = 0

    for i, batch in enumerate(test_loader):
        imgs = batch["image"].to(device, non_blocking=True).float()
        masks_gt = batch["mask"].to(device, non_blocking=True).long()
        target_size = masks_gt.shape[-2:]

        # Inference
        if use_tta:
            avg_logits = tta_predict(model, imgs, tta_scales, use_flip=True,
                                     target_size=target_size, device=device,
                                     use_amp=use_amp)
        else:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                out = model(imgs)
                avg_logits = F.interpolate(out["seg_logits"].float(),
                                           size=target_size, mode="bilinear",
                                           align_corners=False)

        # Argmax to get predicted mask
        pred = avg_logits.argmax(dim=1)  # (B, H, W)

        # Apply TAPP per image
        if use_msf or use_sgf:
            pred_np = pred.cpu().numpy()
            for b in range(pred_np.shape[0]):
                before = pred_np[b].copy()
                pred_np[b] = tapp_postprocess(
                    pred_np[b],
                    use_msf=use_msf, use_sgf=use_sgf,
                    **msf_kwargs, **sgf_kwargs,
                )
                # Count changes
                removed = ((before == 1) & (pred_np[b] == 0)).sum()
                added = ((before == 0) & (pred_np[b] == 1)).sum()
                n_filtered += removed
                n_bridges += added

            # Convert back to logits-like format for metrics
            # Create one-hot and use as "logits"
            pred_tensor = torch.from_numpy(pred_np).to(device).long()
            one_hot = F.one_hot(pred_tensor, C.NUM_CLASSES).permute(0, 3, 1, 2).float()
            metrics.update(one_hot * 100, masks_gt)  # scale up so argmax works
        else:
            metrics.update(avg_logits, masks_gt)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_loader)}]")

    m = metrics.compute()
    return m, n_filtered, n_bridges


def print_results(label: str, m: dict, n_filtered: int = 0, n_bridges: int = 0):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for k in METRIC_KEYS:
        val = m.get(k)
        if val is not None:
            print(f"  {k}: {val:.6f}")
    if n_filtered > 0:
        print(f"  [MSF] pixels filtered: {n_filtered:,}")
    if n_bridges > 0:
        print(f"  [SGF] pixels added: {n_bridges:,}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="TAPP evaluation")
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--tta", action="store_true", help="enable TTA")
    parser.add_argument("--tta-scales", type=float, nargs="+",
                        default=[0.75, 1.0, 1.25])
    parser.add_argument("--no-msf", action="store_true", help="disable MSF")
    parser.add_argument("--no-sgf", action="store_true", help="disable SGF")
    # MSF parameters
    parser.add_argument("--msf-min-area", type=int, default=30)
    parser.add_argument("--msf-min-ecc", type=float, default=0.85)
    parser.add_argument("--msf-max-sol", type=float, default=0.85)
    # SGF parameters
    parser.add_argument("--sgf-max-gap", type=int, default=25)
    parser.add_argument("--sgf-max-angle", type=float, default=60.0)
    parser.add_argument("--sgf-dilate", type=int, default=2)
    # Full ablation mode
    parser.add_argument("--ablation", action="store_true",
                        help="run all 4 variants: none, MSF, SGF, MSF+SGF")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = C.RUNS_DIR / args.run
    if not run_dir.exists():
        print(f"[ERROR] {run_dir} not found")
        sys.exit(1)

    model = load_model(run_dir, device)

    cfg = C.RunCfg()
    cfg.batch_size = 1
    test_files = []
    with open(C.SPLIT_FILES["test"]) as f:
        test_files = [line.strip() for line in f if line.strip()]
    test_loader = build_val_loader(test_files, cfg, device)

    use_amp = (device == "cuda")
    tta_tag = "+TTA" if args.tta else ""

    msf_kwargs = {
        "msf_min_area": args.msf_min_area,
        "msf_min_eccentricity": args.msf_min_ecc,
        "msf_max_solidity": args.msf_max_sol,
    }
    sgf_kwargs = {
        "sgf_max_gap": args.sgf_max_gap,
        "sgf_max_angle_deg": args.sgf_max_angle,
        "sgf_dilate_radius": args.sgf_dilate,
    }

    if args.ablation:
        # Run all 4 variants
        variants = [
            ("Baseline" + tta_tag, False, False),
            ("MSF only" + tta_tag, True, False),
            ("SGF only" + tta_tag, False, True),
            ("MSF+SGF (TAPP)" + tta_tag, True, True),
        ]
        all_results = {}
        for label, use_msf, use_sgf in variants:
            print(f"\n>>> Running: {label}")
            t0 = time.time()
            m, nf, nb = run_eval(model, test_loader, device, use_amp,
                                 args.tta, args.tta_scales,
                                 use_msf, use_sgf, msf_kwargs, sgf_kwargs)
            elapsed = time.time() - t0
            print_results(f"{label}  ({elapsed:.1f}s)", m, nf, nb)
            all_results[label] = m

        # Save comparison table
        report_path = run_dir / "tapp_ablation.txt"
        with open(report_path, "w") as f:
            f.write(f"TAPP Ablation: {args.run}\n")
            f.write(f"TTA: {args.tta}  scales: {args.tta_scales}\n")
            f.write(f"MSF: min_area={args.msf_min_area} min_ecc={args.msf_min_ecc} max_sol={args.msf_max_sol}\n")
            f.write(f"SGF: max_gap={args.sgf_max_gap} max_angle={args.sgf_max_angle} dilate={args.sgf_dilate}\n\n")

            header = f"{'Variant':<25}"
            for k in METRIC_KEYS:
                header += f"  {k:>12}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for label, m in all_results.items():
                row = f"{label:<25}"
                for k in METRIC_KEYS:
                    v = m.get(k)
                    row += f"  {v:>12.6f}" if v is not None else f"  {'N/A':>12}"
                f.write(row + "\n")
        print(f"\n[TAPP] wrote {report_path}")

    else:
        # Single run
        use_msf = not args.no_msf
        use_sgf = not args.no_sgf
        modules = []
        if use_msf:
            modules.append("MSF")
        if use_sgf:
            modules.append("SGF")
        label = "+".join(modules) if modules else "No post-proc"
        label = f"{args.run} {tta_tag} {label}"

        print(f"[TAPP] Running: {label}")
        t0 = time.time()
        m, nf, nb = run_eval(model, test_loader, device, use_amp,
                             args.tta, args.tta_scales,
                             use_msf, use_sgf, msf_kwargs, sgf_kwargs)
        elapsed = time.time() - t0
        print_results(f"{label}  ({elapsed:.1f}s)", m, nf, nb)

        # Save report
        suffix = "tapp"
        if args.tta:
            suffix = "tta_tapp"
        report_path = run_dir / f"test_report_{suffix}.txt"
        with open(report_path, "w") as f:
            f.write(f"run: {args.run}\n")
            f.write(f"tta: {args.tta}\n")
            f.write(f"msf: {use_msf} (min_area={args.msf_min_area} min_ecc={args.msf_min_ecc} max_sol={args.msf_max_sol})\n")
            f.write(f"sgf: {use_sgf} (max_gap={args.sgf_max_gap} max_angle={args.sgf_max_angle} dilate={args.sgf_dilate})\n")
            f.write("test set metrics:\n")
            for k in METRIC_KEYS:
                f.write(f"  {k}: {m.get(k)}\n")
        print(f"[TAPP] wrote {report_path}")


if __name__ == "__main__":
    main()

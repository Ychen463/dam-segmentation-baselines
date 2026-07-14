"""Teacher boundary analysis: compare T1 vs T2 accuracy/confidence at boundary vs body pixels.

Loads both frozen teachers, partitions pixels into boundary vs body regions
(per-class), and reports:
  - T1 vs T2 disagreement (KL) at boundary vs body
  - Per-teacher entropy at boundary vs body
  - Per-teacher accuracy at boundary vs body
  - Class-wise breakdown (crack boundary vs spalling boundary)
  - Tier-wise breakdown (Easy/Medium/Hard)

This informs whether SATE (Spatially-Adaptive Teacher Ensemble) at boundaries
is justified, or if simpler approaches suffice.

Usage:
    python scripts/teacher_boundary_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False

from baseline_unet.dataset import build_transforms, read_split_file
from full_method import config as C
from full_method.config import RunCfg, apply_preset
from full_method.dataset import FullMethodDataset, build_records, dict_collate
from full_method.model import DSCformerDam
from full_method.sam_model import TopoLoRASAM


def extract_boundary_mask(targets: torch.Tensor, width: int = 3,
                          classes: tuple = (1, 2)) -> torch.Tensor:
    """Binary boundary mask (union of all specified classes).

    Returns (B, H, W) bool tensor: True = boundary pixel.
    """
    B, H, W = targets.shape
    boundary = targets.new_zeros((B, H, W), dtype=torch.bool)
    for cls in classes:
        cls_mask = (targets == cls).float().unsqueeze(1)  # (B,1,H,W)
        if cls_mask.sum() < 1:
            continue
        ks = 2 * width + 1
        dil = F.max_pool2d(cls_mask, ks, stride=1, padding=width)
        ero = -F.max_pool2d(-cls_mask, ks, stride=1, padding=width)
        strip = (dil - ero) > 0.5  # (B,1,H,W)
        boundary = boundary | strip[:, 0]
    return boundary


def extract_class_boundary_mask(targets: torch.Tensor, cls: int,
                                width: int = 3) -> torch.Tensor:
    """Boundary mask for a single class. Returns (B, H, W) bool."""
    cls_mask = (targets == cls).float().unsqueeze(1)
    if cls_mask.sum() < 1:
        return targets.new_zeros(targets.shape, dtype=torch.bool)
    ks = 2 * width + 1
    dil = F.max_pool2d(cls_mask, ks, stride=1, padding=width)
    ero = -F.max_pool2d(-cls_mask, ks, stride=1, padding=width)
    return ((dil - ero) > 0.5)[:, 0]


def compute_region_stats(t1_logits, t2_logits, targets, region_mask, temperature=4.0):
    """Compute stats for a spatial region defined by region_mask (B,H,W) bool.

    Returns dict with:
      kl_t1t2_mean, entropy_t1_mean, entropy_t2_mean,
      acc_t1, acc_t2, n_pixels
    """
    n_pixels = region_mask.sum().item()
    if n_pixels == 0:
        return {
            "kl_t1t2_mean": 0.0, "entropy_t1_mean": 0.0, "entropy_t2_mean": 0.0,
            "acc_t1": 0.0, "acc_t2": 0.0, "n_pixels": 0
        }

    # Softmax probs
    t1_prob = F.softmax(t1_logits / temperature, dim=1)  # (B,C,H,W)
    t2_prob = F.softmax(t2_logits / temperature, dim=1)

    # KL(T1 || T2) per pixel
    kl = (t1_prob * (t1_prob.clamp_min(1e-8).log()
                     - t2_prob.clamp_min(1e-8).log())).sum(1)  # (B,H,W)
    kl_mean = kl[region_mask].mean().item()

    # Entropy per teacher (use temperature=1 for interpretable entropy)
    t1_prob_t1 = F.softmax(t1_logits, dim=1)
    t2_prob_t1 = F.softmax(t2_logits, dim=1)
    ent_t1 = -(t1_prob_t1 * t1_prob_t1.clamp_min(1e-8).log()).sum(1)  # (B,H,W)
    ent_t2 = -(t2_prob_t1 * t2_prob_t1.clamp_min(1e-8).log()).sum(1)

    ent_t1_mean = ent_t1[region_mask].mean().item()
    ent_t2_mean = ent_t2[region_mask].mean().item()

    # Accuracy (argmax vs GT)
    t1_pred = t1_logits.argmax(dim=1)  # (B,H,W)
    t2_pred = t2_logits.argmax(dim=1)
    acc_t1 = (t1_pred[region_mask] == targets[region_mask]).float().mean().item()
    acc_t2 = (t2_pred[region_mask] == targets[region_mask]).float().mean().item()

    return {
        "kl_t1t2_mean": round(kl_mean, 6),
        "entropy_t1_mean": round(ent_t1_mean, 6),
        "entropy_t2_mean": round(ent_t2_mean, 6),
        "acc_t1": round(acc_t1, 4),
        "acc_t2": round(acc_t2, 4),
        "n_pixels": n_pixels,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[boundary-analysis] device={device}")

    # Config
    cfg = RunCfg()
    apply_preset(cfg, "DKD10")

    # Load training records
    train_files = read_split_file(C.SPLIT_FILES["train"])
    records = build_records(train_files, C.DATA_ROOT)
    print(f"[boundary-analysis] {len(records)} training samples")

    tier_map = {r["id"]: r["tier"] for r in records}

    # Dataset + loader
    tfm = build_transforms(cfg.img_size, train=False)
    ds = FullMethodDataset(C.DATA_ROOT, records, transform=tfm)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2,
                        collate_fn=dict_collate, pin_memory=True)

    # Load Teacher 1 (DSCformerDam G1)
    t1_path = (C.PKG_DIR / cfg.kd_teacher_checkpoint).resolve()
    print(f"[boundary-analysis] loading T1 from {t1_path}")
    t1_state = torch.load(t1_path, map_location=device, weights_only=False)
    t1_model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)
    t1_model.load_state_dict(t1_state["model"])
    t1_model.eval()

    # Load Teacher 2 (SAM2 LoRA)
    t2_path = (C.PKG_DIR / cfg.kd_teacher2_checkpoint).resolve()
    print(f"[boundary-analysis] loading T2 from {t2_path}")
    t2_state = torch.load(t2_path, map_location=device, weights_only=False)
    t2_model = TopoLoRASAM(
        sam_checkpoint=cfg.sam_checkpoint,
        num_classes=C.NUM_CLASSES,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        fpn_dim=cfg.sam_fpn_dim,
        sam_img_size=cfg.sam_img_size,
    ).to(device)
    t2_model.load_state_dict(t2_state["model"])
    t2_model.eval()

    temperature = cfg.kd_temperature
    boundary_width = 3

    # Accumulators: key -> list of per-batch region stats
    # We accumulate raw pixel counts and weighted sums to compute global means
    accum_keys = [
        "overall_boundary", "overall_body",
        "crack_boundary", "spalling_boundary",
    ]
    tier_names = {0: "Easy", 1: "Medium", 2: "Hard"}
    for tn in tier_names.values():
        accum_keys.append(f"{tn}_boundary")
        accum_keys.append(f"{tn}_body")

    # For weighted accumulation: sum of (metric * n_pixels) and sum of n_pixels
    accum = {k: {"kl_sum": 0.0, "ent_t1_sum": 0.0, "ent_t2_sum": 0.0,
                 "correct_t1": 0, "correct_t2": 0, "n_pixels": 0}
             for k in accum_keys}

    def accumulate(key, t1_logits, t2_logits, targets, mask, temperature):
        """Accumulate stats for a region."""
        n = mask.sum().item()
        if n == 0:
            return

        t1_prob = F.softmax(t1_logits / temperature, dim=1)
        t2_prob = F.softmax(t2_logits / temperature, dim=1)
        kl = (t1_prob * (t1_prob.clamp_min(1e-8).log()
                         - t2_prob.clamp_min(1e-8).log())).sum(1)

        t1_prob_raw = F.softmax(t1_logits, dim=1)
        t2_prob_raw = F.softmax(t2_logits, dim=1)
        ent_t1 = -(t1_prob_raw * t1_prob_raw.clamp_min(1e-8).log()).sum(1)
        ent_t2 = -(t2_prob_raw * t2_prob_raw.clamp_min(1e-8).log()).sum(1)

        t1_pred = t1_logits.argmax(dim=1)
        t2_pred = t2_logits.argmax(dim=1)

        a = accum[key]
        a["kl_sum"] += kl[mask].sum().item()
        a["ent_t1_sum"] += ent_t1[mask].sum().item()
        a["ent_t2_sum"] += ent_t2[mask].sum().item()
        a["correct_t1"] += (t1_pred[mask] == targets[mask]).sum().item()
        a["correct_t2"] += (t2_pred[mask] == targets[mask]).sum().item()
        a["n_pixels"] += n

    print("[boundary-analysis] running inference ...")
    n_batches = len(loader)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if (bi + 1) % 50 == 0:
                print(f"  batch {bi+1}/{n_batches}")

            imgs = batch["image"].to(device).float()
            masks = batch["mask"].to(device).long()
            sids = batch["sample_id"]
            H, W = masks.shape[-2:]

            t1_out = t1_model(imgs)
            t1_logits = F.interpolate(t1_out["seg_logits"].float(), (H, W),
                                      mode="bilinear", align_corners=False)
            t2_out = t2_model(imgs)
            t2_logits = F.interpolate(t2_out["seg_logits"].float(), (H, W),
                                      mode="bilinear", align_corners=False)

            # Overall boundary vs body
            bdry = extract_boundary_mask(masks, width=boundary_width, classes=(1, 2))
            body = ~bdry

            accumulate("overall_boundary", t1_logits, t2_logits, masks, bdry, temperature)
            accumulate("overall_body", t1_logits, t2_logits, masks, body, temperature)

            # Per-class boundary
            crack_bdry = extract_class_boundary_mask(masks, cls=1, width=boundary_width)
            spalling_bdry = extract_class_boundary_mask(masks, cls=2, width=boundary_width)
            accumulate("crack_boundary", t1_logits, t2_logits, masks, crack_bdry, temperature)
            accumulate("spalling_boundary", t1_logits, t2_logits, masks, spalling_bdry, temperature)

            # Per-tier: need per-sample masks
            for i, sid in enumerate(sids):
                tier = tier_map.get(sid, -1)
                if tier not in tier_names:
                    continue
                tn = tier_names[tier]
                # Single-sample slices
                t1_i = t1_logits[i:i+1]
                t2_i = t2_logits[i:i+1]
                m_i = masks[i:i+1]

                bdry_i = extract_boundary_mask(m_i, width=boundary_width, classes=(1, 2))
                body_i = ~bdry_i

                accumulate(f"{tn}_boundary", t1_i, t2_i, m_i, bdry_i, temperature)
                accumulate(f"{tn}_body", t1_i, t2_i, m_i, body_i, temperature)

    # Finalize: compute means from accumulated sums
    def finalize(key):
        a = accum[key]
        n = a["n_pixels"]
        if n == 0:
            return {"kl_t1t2_mean": 0.0, "entropy_t1_mean": 0.0,
                    "entropy_t2_mean": 0.0, "acc_t1": 0.0, "acc_t2": 0.0,
                    "n_pixels": 0}
        return {
            "kl_t1t2_mean": round(a["kl_sum"] / n, 6),
            "entropy_t1_mean": round(a["ent_t1_sum"] / n, 6),
            "entropy_t2_mean": round(a["ent_t2_sum"] / n, 6),
            "acc_t1": round(a["correct_t1"] / n, 4),
            "acc_t2": round(a["correct_t2"] / n, 4),
            "n_pixels": n,
        }

    results = {
        "overall": {
            "boundary": finalize("overall_boundary"),
            "body": finalize("overall_body"),
        },
        "by_class": {
            "crack_boundary": finalize("crack_boundary"),
            "spalling_boundary": finalize("spalling_boundary"),
        },
        "by_tier": {},
    }
    for tn in tier_names.values():
        results["by_tier"][tn] = {
            "boundary": finalize(f"{tn}_boundary"),
            "body": finalize(f"{tn}_body"),
        }

    # Print summary
    print("\n" + "=" * 70)
    print("TEACHER BOUNDARY ANALYSIS")
    print("=" * 70)

    def print_region(label, stats):
        n = stats["n_pixels"]
        if n == 0:
            print(f"  {label:25s}: no pixels")
            return
        print(f"  {label:25s}: n={n:>10,d}  "
              f"KL={stats['kl_t1t2_mean']:.4f}  "
              f"ent_T1={stats['entropy_t1_mean']:.4f}  "
              f"ent_T2={stats['entropy_t2_mean']:.4f}  "
              f"acc_T1={stats['acc_t1']:.4f}  "
              f"acc_T2={stats['acc_t2']:.4f}")

    print("\n--- Overall ---")
    print_region("Boundary", results["overall"]["boundary"])
    print_region("Body", results["overall"]["body"])

    bdry_s = results["overall"]["boundary"]
    body_s = results["overall"]["body"]
    if bdry_s["n_pixels"] > 0 and body_s["n_pixels"] > 0:
        print(f"\n  Boundary/Body ratios:")
        print(f"    KL:     {bdry_s['kl_t1t2_mean'] / max(body_s['kl_t1t2_mean'], 1e-8):.2f}x")
        print(f"    Ent T1: {bdry_s['entropy_t1_mean'] / max(body_s['entropy_t1_mean'], 1e-8):.2f}x")
        print(f"    Ent T2: {bdry_s['entropy_t2_mean'] / max(body_s['entropy_t2_mean'], 1e-8):.2f}x")
        print(f"    Acc T1: {bdry_s['acc_t1']:.4f} vs {body_s['acc_t1']:.4f} (diff={bdry_s['acc_t1']-body_s['acc_t1']:+.4f})")
        print(f"    Acc T2: {bdry_s['acc_t2']:.4f} vs {body_s['acc_t2']:.4f} (diff={bdry_s['acc_t2']-body_s['acc_t2']:+.4f})")

    print("\n--- By Class ---")
    print_region("Crack boundary", results["by_class"]["crack_boundary"])
    print_region("Spalling boundary", results["by_class"]["spalling_boundary"])

    c_s = results["by_class"]["crack_boundary"]
    sp_s = results["by_class"]["spalling_boundary"]
    if c_s["n_pixels"] > 0 and sp_s["n_pixels"] > 0:
        print(f"\n  Crack vs Spalling boundary:")
        print(f"    T1 better at crack bdry?  acc_T1={c_s['acc_t1']:.4f} vs acc_T2={c_s['acc_t2']:.4f}")
        print(f"    T1 better at spall bdry?  acc_T1={sp_s['acc_t1']:.4f} vs acc_T2={sp_s['acc_t2']:.4f}")

    print("\n--- By Tier ---")
    for tn in ["Easy", "Medium", "Hard"]:
        if tn in results["by_tier"]:
            print(f"\n  [{tn}]")
            print_region("  Boundary", results["by_tier"][tn]["boundary"])
            print_region("  Body", results["by_tier"][tn]["body"])

    # Decision guidance
    print("\n" + "=" * 70)
    print("SATE DECISION GUIDANCE")
    print("=" * 70)
    if bdry_s["n_pixels"] > 0 and body_s["n_pixels"] > 0:
        kl_ratio = bdry_s["kl_t1t2_mean"] / max(body_s["kl_t1t2_mean"], 1e-8)
        acc_gap_t1 = body_s["acc_t1"] - bdry_s["acc_t1"]
        acc_gap_t2 = body_s["acc_t2"] - bdry_s["acc_t2"]
        t1_better_at_bdry = bdry_s["acc_t1"] > bdry_s["acc_t2"]

        print(f"  Teacher disagreement at boundary: {kl_ratio:.2f}x body")
        print(f"  T1 accuracy drop at boundary: {acc_gap_t1:+.4f}")
        print(f"  T2 accuracy drop at boundary: {acc_gap_t2:+.4f}")
        print(f"  Better teacher at boundary: {'T1' if t1_better_at_bdry else 'T2'} "
              f"(acc={max(bdry_s['acc_t1'], bdry_s['acc_t2']):.4f})")

        if kl_ratio > 1.5 and abs(bdry_s["acc_t1"] - bdry_s["acc_t2"]) > 0.02:
            print("\n  >> SATE RECOMMENDED: teachers disagree more at boundaries,")
            print("     and one teacher is clearly better → adaptive weighting can help.")
        elif kl_ratio > 1.5:
            print("\n  >> BOUNDARY LABEL SMOOTHING may help: high disagreement but")
            print("     neither teacher is clearly better → reduce confidence at boundary.")
        else:
            print("\n  >> STUDENT-SIDE ONLY may suffice: teachers are similarly")
            print("     confident/accurate at boundary vs body.")

    # Save
    out_dir = Path("results/dgacl")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "teacher_boundary_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[boundary-analysis] saved to {out_path}")


if __name__ == "__main__":
    main()

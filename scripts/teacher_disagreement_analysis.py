"""Sanity check: teacher disagreement distribution across training samples.

Loads both frozen teachers (T1=G1 DSCformer, T2=SAM2 LoRA), runs inference on
all training samples, computes per-sample KL(T1 || T2) pixel-mean, groups by
tier, and reports statistics.

Go/no-go: if disagreement varies meaningfully by tier → proceed with DGACL.
If flat → redesign difficulty signal.

Usage:
    python scripts/teacher_disagreement_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[analysis] device={device}")

    # Config (use DKD10 settings for model params)
    cfg = RunCfg()
    apply_preset(cfg, "DKD10")

    # Load training records
    train_files = read_split_file(C.SPLIT_FILES["train"])
    records = build_records(train_files, C.DATA_ROOT)
    print(f"[analysis] {len(records)} training samples")

    # Build tier lookup
    tier_map = {r["id"]: r["tier"] for r in records}

    # Dataset + loader
    tfm = build_transforms(cfg.img_size, mode="val")
    ds = FullMethodDataset(records, C.DATA_ROOT, transform=tfm)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2,
                        collate_fn=dict_collate, pin_memory=True)

    # Load Teacher 1 (DSCformerDam G1)
    t1_path = (C.PKG_DIR / cfg.kd_teacher_checkpoint).resolve()
    print(f"[analysis] loading T1 from {t1_path}")
    t1_state = torch.load(t1_path, map_location=device, weights_only=False)
    t1_model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)
    t1_model.load_state_dict(t1_state["model"])
    t1_model.eval()

    # Load Teacher 2 (SAM2 LoRA)
    t2_path = (C.PKG_DIR / cfg.kd_teacher2_checkpoint).resolve()
    print(f"[analysis] loading T2 from {t2_path}")
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

    # Optionally load DKD10 student for student-teacher gap
    student_model = None
    dkd10_best = C.PKG_DIR / "runs" / "dkd10_no_srl" / "best.pt"
    if dkd10_best.exists():
        print(f"[analysis] loading DKD10 student from {dkd10_best}")
        s_state = torch.load(dkd10_best, map_location=device, weights_only=False)
        student_model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)
        student_model.load_state_dict(s_state["model"])
        student_model.eval()

    temperature = cfg.kd_temperature
    results = {}  # sample_id -> {disagreement, gap, tier}

    print("[analysis] running inference ...")
    with torch.no_grad():
        for batch in loader:
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

            # KL(T1 || T2) per pixel
            t1_prob = F.softmax(t1_logits / temperature, dim=1)
            t2_prob = F.softmax(t2_logits / temperature, dim=1)
            kl = (t1_prob * (t1_prob.clamp_min(1e-8).log()
                             - t2_prob.clamp_min(1e-8).log())).sum(1)  # (B,H,W)
            disagree = kl.mean(dim=(1, 2)).cpu().numpy()

            # Student-teacher gap (if student available)
            gap = np.zeros(len(sids))
            if student_model is not None:
                s_out = student_model(imgs)
                s_logits = F.interpolate(s_out["seg_logits"].float(), (H, W),
                                         mode="bilinear", align_corners=False)
                # Ensemble = weighted average (same as DKD10)
                from full_method.train import dual_teacher_ensemble
                ens_logits = dual_teacher_ensemble(t1_logits, t2_logits, cfg)
                s_prob = F.softmax(s_logits / temperature, dim=1)
                ens_prob = F.softmax(ens_logits / temperature, dim=1)
                kl_gap = (ens_prob * (ens_prob.clamp_min(1e-8).log()
                                      - s_prob.clamp_min(1e-8).log())).sum(1)
                gap = kl_gap.mean(dim=(1, 2)).cpu().numpy()

            for i, sid in enumerate(sids):
                results[sid] = {
                    "disagreement": float(disagree[i]),
                    "gap": float(gap[i]),
                    "tier": tier_map.get(sid, -1),
                }

    # Group by tier
    tier_stats = {0: [], 1: [], 2: []}
    for sid, r in results.items():
        t = r["tier"]
        if t in tier_stats:
            tier_stats[t].append(r["disagreement"])

    print("\n=== Teacher Disagreement by Tier ===")
    tier_names = {0: "Easy", 1: "Medium", 2: "Hard"}
    for t in sorted(tier_stats.keys()):
        vals = np.array(tier_stats[t])
        if len(vals) > 0:
            print(f"  Tier {t} ({tier_names[t]:>6s}): n={len(vals):4d}  "
                  f"mean={vals.mean():.6f}  std={vals.std():.6f}  "
                  f"min={vals.min():.6f}  max={vals.max():.6f}")

    # Also report gap stats
    if student_model is not None:
        print("\n=== Student-Teacher Gap by Tier ===")
        gap_stats = {0: [], 1: [], 2: []}
        for sid, r in results.items():
            t = r["tier"]
            if t in gap_stats:
                gap_stats[t].append(r["gap"])
        for t in sorted(gap_stats.keys()):
            vals = np.array(gap_stats[t])
            if len(vals) > 0:
                print(f"  Tier {t} ({tier_names[t]:>6s}): n={len(vals):4d}  "
                      f"mean={vals.mean():.6f}  std={vals.std():.6f}  "
                      f"min={vals.min():.6f}  max={vals.max():.6f}")

    # Save results
    out_dir = Path("results/dgacl")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "disagreement_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[analysis] saved to {out_path}")

    # Go/no-go verdict
    all_disagrees = [r["disagreement"] for r in results.values()]
    overall_std = np.std(all_disagrees)
    tier_means = [np.mean(tier_stats[t]) for t in sorted(tier_stats.keys()) if tier_stats[t]]
    spread = max(tier_means) - min(tier_means) if len(tier_means) > 1 else 0
    print(f"\n[verdict] overall_std={overall_std:.6f}  tier_spread={spread:.6f}")
    if spread > 0.001 or overall_std > 0.005:
        print("[verdict] GO — disagreement signal is informative")
    else:
        print("[verdict] CAUTION — disagreement is flat, may need redesign")


if __name__ == "__main__":
    main()

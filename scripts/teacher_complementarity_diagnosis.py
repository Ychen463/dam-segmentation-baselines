"""Diagnose teacher complementarity on the balanced-group validation set.

Computes:
  1. Pixel-level: T2-correct-T1-wrong proportion (per class)
  2. Component-level: spalling/crack components found by T2 but missed by T1
  3. Per-tier teacher comparison (mIoU_fg, IoU per class)
  4. Teacher disagreement vs actual error correlation
  5. Oracle ensemble upper bound (pixel-level best-teacher selection)
  6. Image Oracle: per-image best-teacher selection with per-tier distribution
  7. Component Oracle with Thin Crack Analysis: per-component best-teacher
     selection, T2-rescue counts, and thin crack (median width ≤ 2px) breakdown

Usage (on RunPod, from Codes/):
    python scripts/teacher_complementarity_diagnosis.py

Outputs results to stdout and saves JSON to
    baseline_unet/splits/balanced_group_split/teacher_diagnosis.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset

torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_unet import config as C
from baseline_unet.dataset import (
    build_transforms,
    decode_mask,
    image_path,
    mask_path,
    read_mask_rgb,
)

CODES_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = CODES_DIR / "baseline_unet" / "splits" / "balanced_group_split"
SUFFIX = "bgsplit"

# Teacher checkpoints
T1_CKPT = CODES_DIR / "full_method" / "runs" / f"dscformer_srl_G1_{SUFFIX}" / "best.pt"
T2_CKPT = CODES_DIR / "full_method" / "runs" / f"sam_lora_srl_SAM2_{SUFFIX}" / "best.pt"

CLASS_NAMES = {0: "background", 1: "crack", 2: "spalling"}
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DamSegDataset(Dataset):
    def __init__(self, data_root: Path, rels: list[str], img_size: int = 512):
        self.data_root = data_root
        self.rels = rels
        self.transform = build_transforms(img_size, train=False)

    def __len__(self):
        return len(self.rels)

    def __getitem__(self, idx):
        rel = self.rels[idx]
        img = cv2.imread(str(image_path(self.data_root, rel)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = read_mask_rgb(mask_path(self.data_root, rel))
        label, _ = decode_mask(m)
        out = self.transform(image=img, mask=label)
        image_t = out["image"]
        mask_t = out["mask"]
        if not torch.is_tensor(mask_t):
            mask_t = torch.from_numpy(mask_t)
        return image_t, mask_t.long(), rel


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_teacher(model_name: str, ckpt_path: Path, device: str) -> nn.Module:
    from shared_eval.model_registry import get as get_entry, FullMethodLogitsWrapper
    from full_method.config import ABLATION_PRESETS, RunCfg

    entry = get_entry(model_name)

    # SAM-LoRA needs special build logic (registry build_fn doesn't handle it)
    preset_id = None
    for pid, pcfg in ABLATION_PRESETS.items():
        if pcfg["name"] == model_name:
            preset_id = pid
            break

    if preset_id and ABLATION_PRESETS[preset_id].get("model_type") == "sam_lora":
        from full_method.sam_model import TopoLoRASAM
        cfg = RunCfg()
        pcfg = ABLATION_PRESETS[preset_id]
        for k, v in pcfg.items():
            if k != "name" and hasattr(cfg, k):
                setattr(cfg, k, v)
        model = TopoLoRASAM(
            sam_checkpoint=cfg.sam_checkpoint,
            num_classes=NUM_CLASSES,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            fpn_dim=cfg.sam_fpn_dim,
            sam_img_size=cfg.sam_img_size,
        )
    else:
        model = entry.build_fn()

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "ema_model" in state:
            model.load_state_dict(state["ema_model"])
        else:
            model.load_state_dict(state["model"])

    if entry.inference_wrapper is not None:
        model = entry.inference_wrapper(model, entry.img_size)

    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Component analysis helpers
# ---------------------------------------------------------------------------

def count_components(mask: np.ndarray, class_id: int) -> int:
    binary = (mask == class_id)
    if not binary.any():
        return 0
    _, n = ndimage.label(binary)
    return n


def components_hit(pred: np.ndarray, gt: np.ndarray, class_id: int,
                   threshold: float = 0.5) -> tuple[int, int]:
    """Returns (hit_count, total_gt_components)."""
    gt_mask = (gt == class_id)
    if not gt_mask.any():
        return 0, 0
    pred_mask = (pred == class_id)
    labeled, n_comp = ndimage.label(gt_mask)
    if n_comp == 0:
        return 0, 0
    hit = 0
    for c in range(1, n_comp + 1):
        comp = (labeled == c)
        overlap = float((comp & pred_mask).sum()) / float(comp.sum())
        if overlap >= threshold:
            hit += 1
    return hit, n_comp


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_diagnosis(val_rels: list[str], data_root: Path, device: str):
    # Load teachers
    print("[diag] Loading Teacher 1 (DSConv+SRL G1) ...")
    t1_model = _load_teacher("dscformer_srl_G1", T1_CKPT, device)
    print("[diag] Loading Teacher 2 (SAM-LoRA) ...")
    t2_model = _load_teacher("sam_lora_srl_SAM2", T2_CKPT, device)

    img_size = 512  # both teachers use 512 eval transforms
    dataset = DamSegDataset(data_root, val_rels, img_size=img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Accumulators
    # Pixel-level
    px = {
        "t1_correct_t2_wrong": defaultdict(int),
        "t2_correct_t1_wrong": defaultdict(int),
        "both_correct": defaultdict(int),
        "both_wrong": defaultdict(int),
        "total": defaultdict(int),
    }

    # IoU per teacher per class
    iou_acc = {
        "t1": {"intersection": defaultdict(int), "union": defaultdict(int)},
        "t2": {"intersection": defaultdict(int), "union": defaultdict(int)},
        "oracle": {"intersection": defaultdict(int), "union": defaultdict(int)},
    }

    # Per-tier
    tier_iou = defaultdict(lambda: {
        "t1": {"intersection": defaultdict(int), "union": defaultdict(int)},
        "t2": {"intersection": defaultdict(int), "union": defaultdict(int)},
        "oracle": {"intersection": defaultdict(int), "union": defaultdict(int)},
    })

    # Component-level
    comp = {
        "t1_hit": defaultdict(int), "t1_total": defaultdict(int),
        "t2_hit": defaultdict(int), "t2_total": defaultdict(int),
        "t2_finds_t1_misses": defaultdict(int),  # components T2 hits but T1 misses
    }

    # Disagreement vs error
    disagree_pixels_total = 0
    disagree_and_t1_wrong = 0
    disagree_and_t2_wrong = 0
    agree_pixels_total = 0
    agree_and_wrong = 0

    # Image Oracle accumulators
    img_oracle_iou = {
        "intersection": defaultdict(int),
        "union": defaultdict(int),
    }
    img_oracle_tier_wins = defaultdict(lambda: {"t1": 0, "t2": 0})

    # Component Oracle accumulators
    comp_oracle = {
        "t1_only": defaultdict(int),       # T1 hits, T2 misses
        "t2_only": defaultdict(int),       # T2 hits, T1 misses (rescued)
        "both_detected": defaultdict(int), # both hit
        "neither": defaultdict(int),       # both miss
        "total": defaultdict(int),
        "oracle_hit": defaultdict(int),    # best-teacher hit count
    }

    # Thin crack analysis accumulators
    thin_crack = {
        "total": 0,
        "t1_hit": 0,
        "t2_hit": 0,
        "t2_rescued": 0,       # T2 hits but T1 misses
        "both_detected": 0,
        "neither": 0,
    }

    for images, masks, rels in loader:
        images = images.to(device, non_blocking=True)
        gt = masks.numpy()[0]  # (H, W)

        # T1 prediction
        logits_t1 = t1_model(images)
        pred_t1 = logits_t1.argmax(1).cpu().numpy()[0]

        # T2 prediction
        logits_t2 = t2_model(images)
        pred_t2 = logits_t2.argmax(1).cpu().numpy()[0]

        # Oracle: pixel-level best teacher
        pred_oracle = np.where(pred_t1 == gt, pred_t1,
                               np.where(pred_t2 == gt, pred_t2, pred_t1))

        # Tier
        rel = rels[0]
        tier = rel.split("/")[0]

        # --- Pixel-level analysis ---
        t1_correct = (pred_t1 == gt)
        t2_correct = (pred_t2 == gt)

        for cls_id in range(NUM_CLASSES):
            cls_mask = (gt == cls_id)
            n = int(cls_mask.sum())
            if n == 0:
                continue
            px["total"][cls_id] += n
            px["both_correct"][cls_id] += int((t1_correct & t2_correct & cls_mask).sum())
            px["both_wrong"][cls_id] += int((~t1_correct & ~t2_correct & cls_mask).sum())
            px["t1_correct_t2_wrong"][cls_id] += int((t1_correct & ~t2_correct & cls_mask).sum())
            px["t2_correct_t1_wrong"][cls_id] += int((~t1_correct & t2_correct & cls_mask).sum())

        # --- IoU accumulation ---
        for cls_id in range(NUM_CLASSES):
            for pred, key in [(pred_t1, "t1"), (pred_t2, "t2"), (pred_oracle, "oracle")]:
                inter = int(((pred == cls_id) & (gt == cls_id)).sum())
                union = int(((pred == cls_id) | (gt == cls_id)).sum())
                iou_acc[key]["intersection"][cls_id] += inter
                iou_acc[key]["union"][cls_id] += union
                tier_iou[tier][key]["intersection"][cls_id] += inter
                tier_iou[tier][key]["union"][cls_id] += union

        # --- Component-level analysis ---
        for cls_id in [1, 2]:  # crack, spalling
            t1_h, t1_tot = components_hit(pred_t1, gt, cls_id)
            t2_h, t2_tot = components_hit(pred_t2, gt, cls_id)
            comp["t1_hit"][cls_id] += t1_h
            comp["t1_total"][cls_id] += t1_tot
            comp["t2_hit"][cls_id] += t2_h
            comp["t2_total"][cls_id] += t2_tot

            # Components T2 hits but T1 misses
            gt_mask = (gt == cls_id)
            if gt_mask.any():
                pred_t1_mask = (pred_t1 == cls_id)
                pred_t2_mask = (pred_t2 == cls_id)
                labeled, n_comp = ndimage.label(gt_mask)
                for c in range(1, n_comp + 1):
                    c_mask = (labeled == c)
                    c_sum = float(c_mask.sum())
                    t1_overlap = float((c_mask & pred_t1_mask).sum()) / c_sum
                    t2_overlap = float((c_mask & pred_t2_mask).sum()) / c_sum
                    if t2_overlap >= 0.5 and t1_overlap < 0.5:
                        comp["t2_finds_t1_misses"][cls_id] += 1

        # --- Disagreement vs error ---
        disagree = (pred_t1 != pred_t2)
        agree = ~disagree
        n_disagree = int(disagree.sum())
        n_agree = int(agree.sum())
        disagree_pixels_total += n_disagree
        agree_pixels_total += n_agree
        disagree_and_t1_wrong += int((disagree & ~t1_correct).sum())
        disagree_and_t2_wrong += int((disagree & ~t2_correct).sum())
        agree_and_wrong += int((agree & ~t1_correct).sum())

        # --- Image Oracle: pick teacher with higher present-class macro IoU ---
        present_classes = [c for c in range(NUM_CLASSES) if (gt == c).any()]
        t1_ious_img, t2_ious_img = [], []
        for c in present_classes:
            gt_c = (gt == c)
            inter_t1 = int(((pred_t1 == c) & gt_c).sum())
            union_t1 = int(((pred_t1 == c) | gt_c).sum())
            inter_t2 = int(((pred_t2 == c) & gt_c).sum())
            union_t2 = int(((pred_t2 == c) | gt_c).sum())
            t1_ious_img.append(inter_t1 / union_t1 if union_t1 > 0 else 0.0)
            t2_ious_img.append(inter_t2 / union_t2 if union_t2 > 0 else 0.0)
        t1_macro = sum(t1_ious_img) / len(t1_ious_img) if t1_ious_img else 0.0
        t2_macro = sum(t2_ious_img) / len(t2_ious_img) if t2_ious_img else 0.0
        best_pred = pred_t1 if t1_macro >= t2_macro else pred_t2
        winner = "t1" if t1_macro >= t2_macro else "t2"
        img_oracle_tier_wins[tier][winner] += 1
        # Accumulate image oracle IoU
        for c in range(NUM_CLASSES):
            gt_c = (gt == c)
            inter = int(((best_pred == c) & gt_c).sum())
            union = int(((best_pred == c) | gt_c).sum())
            img_oracle_iou["intersection"][c] += inter
            img_oracle_iou["union"][c] += union

        # --- Component Oracle with Thin Crack Analysis ---
        for cls_id in [1, 2]:
            gt_mask_co = (gt == cls_id)
            if not gt_mask_co.any():
                continue
            pred_t1_mask_co = (pred_t1 == cls_id)
            pred_t2_mask_co = (pred_t2 == cls_id)
            labeled_co, n_comp_co = ndimage.label(gt_mask_co)
            for c in range(1, n_comp_co + 1):
                c_mask = (labeled_co == c)
                c_sum = float(c_mask.sum())
                t1_cov = float((c_mask & pred_t1_mask_co).sum()) / c_sum
                t2_cov = float((c_mask & pred_t2_mask_co).sum()) / c_sum
                best_cov = max(t1_cov, t2_cov)
                t1_hit_flag = (t1_cov >= 0.5)
                t2_hit_flag = (t2_cov >= 0.5)
                comp_oracle["total"][cls_id] += 1
                if best_cov >= 0.5:
                    comp_oracle["oracle_hit"][cls_id] += 1
                if t1_hit_flag and t2_hit_flag:
                    comp_oracle["both_detected"][cls_id] += 1
                elif t2_hit_flag and not t1_hit_flag:
                    comp_oracle["t2_only"][cls_id] += 1
                elif t1_hit_flag and not t2_hit_flag:
                    comp_oracle["t1_only"][cls_id] += 1
                else:
                    comp_oracle["neither"][cls_id] += 1

                # Thin crack analysis (class 1 only)
                if cls_id == 1:
                    # Skeletonize and measure width via distance transform
                    skel = skeletonize(c_mask)
                    if skel.any():
                        dt = distance_transform_edt(c_mask)
                        skel_widths = dt[skel]
                        median_width = float(np.median(skel_widths))
                        if median_width <= 2.0:
                            thin_crack["total"] += 1
                            if t1_hit_flag:
                                thin_crack["t1_hit"] += 1
                            if t2_hit_flag:
                                thin_crack["t2_hit"] += 1
                            if t1_hit_flag and t2_hit_flag:
                                thin_crack["both_detected"] += 1
                            elif t2_hit_flag and not t1_hit_flag:
                                thin_crack["t2_rescued"] += 1
                            elif not t1_hit_flag and not t2_hit_flag:
                                thin_crack["neither"] += 1

    # --- Compute results ---
    results = {}

    # 1. Pixel-level complementarity
    print("\n" + "=" * 70)
    print("  1. PIXEL-LEVEL COMPLEMENTARITY")
    print("=" * 70)
    px_results = {}
    for cls_id in range(NUM_CLASSES):
        name = CLASS_NAMES[cls_id]
        total = px["total"][cls_id]
        if total == 0:
            continue
        t2_unique = px["t2_correct_t1_wrong"][cls_id]
        t1_unique = px["t1_correct_t2_wrong"][cls_id]
        both_c = px["both_correct"][cls_id]
        both_w = px["both_wrong"][cls_id]
        px_results[name] = {
            "total_pixels": total,
            "both_correct_pct": round(both_c / total * 100, 2),
            "t1_only_correct_pct": round(t1_unique / total * 100, 2),
            "t2_only_correct_pct": round(t2_unique / total * 100, 2),
            "both_wrong_pct": round(both_w / total * 100, 2),
        }
        print(f"\n  {name}:")
        print(f"    Both correct:       {both_c / total * 100:6.2f}%")
        print(f"    T1-only correct:    {t1_unique / total * 100:6.2f}%")
        print(f"    T2-only correct:    {t2_unique / total * 100:6.2f}%  <-- T2 unique contribution")
        print(f"    Both wrong:         {both_w / total * 100:6.2f}%")
    results["pixel_complementarity"] = px_results

    # 2. Component-level
    print("\n" + "=" * 70)
    print("  2. COMPONENT-LEVEL ANALYSIS")
    print("=" * 70)
    comp_results = {}
    for cls_id in [1, 2]:
        name = CLASS_NAMES[cls_id]
        t1_recall = comp["t1_hit"][cls_id] / comp["t1_total"][cls_id] * 100 if comp["t1_total"][cls_id] > 0 else 0
        t2_recall = comp["t2_hit"][cls_id] / comp["t2_total"][cls_id] * 100 if comp["t2_total"][cls_id] > 0 else 0
        t2_unique_comps = comp["t2_finds_t1_misses"][cls_id]
        comp_results[name] = {
            "gt_components": comp["t1_total"][cls_id],
            "t1_recall_pct": round(t1_recall, 2),
            "t2_recall_pct": round(t2_recall, 2),
            "t2_finds_t1_misses": t2_unique_comps,
        }
        print(f"\n  {name}:")
        print(f"    GT components:         {comp['t1_total'][cls_id]}")
        print(f"    T1 comp recall:        {t1_recall:.1f}%  ({comp['t1_hit'][cls_id]}/{comp['t1_total'][cls_id]})")
        print(f"    T2 comp recall:        {t2_recall:.1f}%  ({comp['t2_hit'][cls_id]}/{comp['t2_total'][cls_id]})")
        print(f"    T2 finds, T1 misses:   {t2_unique_comps} components")
    results["component_analysis"] = comp_results

    # 3. Per-tier IoU
    print("\n" + "=" * 70)
    print("  3. PER-TIER TEACHER COMPARISON (IoU)")
    print("=" * 70)
    tier_results = {}
    for tier in ["Easy", "Medium", "Hard"]:
        if tier not in tier_iou:
            continue
        tier_data = tier_iou[tier]
        tier_row = {}
        print(f"\n  {tier}:")
        for key in ["t1", "t2", "oracle"]:
            ious = []
            for cls_id in range(NUM_CLASSES):
                u = tier_data[key]["union"][cls_id]
                iou = tier_data[key]["intersection"][cls_id] / u * 100 if u > 0 else 0
                ious.append(iou)
            miou_fg = (ious[1] + ious[2]) / 2
            label = {"t1": "Teacher 1", "t2": "Teacher 2", "oracle": "Oracle  "}[key]
            print(f"    {label}:  mIoU_fg={miou_fg:.1f}%  IoU_cr={ious[1]:.1f}%  IoU_sp={ious[2]:.1f}%")
            tier_row[key] = {
                "mIoU_fg": round(miou_fg, 2),
                "IoU_crack": round(ious[1], 2),
                "IoU_spalling": round(ious[2], 2),
            }
        tier_results[tier] = tier_row
    results["per_tier"] = tier_results

    # 4. Disagreement vs error
    print("\n" + "=" * 70)
    print("  4. TEACHER DISAGREEMENT vs ACTUAL ERROR")
    print("=" * 70)
    disagree_t1_err_rate = disagree_and_t1_wrong / disagree_pixels_total * 100 if disagree_pixels_total > 0 else 0
    disagree_t2_err_rate = disagree_and_t2_wrong / disagree_pixels_total * 100 if disagree_pixels_total > 0 else 0
    agree_err_rate = agree_and_wrong / agree_pixels_total * 100 if agree_pixels_total > 0 else 0
    print(f"  Disagreement pixels:     {disagree_pixels_total:,} ({disagree_pixels_total / (disagree_pixels_total + agree_pixels_total) * 100:.1f}%)")
    print(f"  Agreement pixels:        {agree_pixels_total:,}")
    print(f"  When teachers disagree:")
    print(f"    T1 error rate:         {disagree_t1_err_rate:.1f}%")
    print(f"    T2 error rate:         {disagree_t2_err_rate:.1f}%")
    print(f"  When teachers agree:")
    print(f"    Error rate:            {agree_err_rate:.1f}%")
    results["disagreement_analysis"] = {
        "disagree_pixels": disagree_pixels_total,
        "agree_pixels": agree_pixels_total,
        "disagree_t1_error_pct": round(disagree_t1_err_rate, 2),
        "disagree_t2_error_pct": round(disagree_t2_err_rate, 2),
        "agree_error_pct": round(agree_err_rate, 2),
    }

    # 5. Oracle ensemble upper bound
    print("\n" + "=" * 70)
    print("  5. ORACLE ENSEMBLE UPPER BOUND")
    print("=" * 70)
    oracle_results = {}
    for cls_id in range(NUM_CLASSES):
        name = CLASS_NAMES[cls_id]
        t1_u = iou_acc["t1"]["union"][cls_id]
        t2_u = iou_acc["t2"]["union"][cls_id]
        or_u = iou_acc["oracle"]["union"][cls_id]
        t1_iou = iou_acc["t1"]["intersection"][cls_id] / t1_u * 100 if t1_u > 0 else 0
        t2_iou = iou_acc["t2"]["intersection"][cls_id] / t2_u * 100 if t2_u > 0 else 0
        or_iou = iou_acc["oracle"]["intersection"][cls_id] / or_u * 100 if or_u > 0 else 0
        oracle_results[name] = {
            "t1_iou": round(t1_iou, 2),
            "t2_iou": round(t2_iou, 2),
            "oracle_iou": round(or_iou, 2),
            "oracle_gain_over_t1": round(or_iou - t1_iou, 2),
        }
        print(f"  {name}:  T1={t1_iou:.1f}%  T2={t2_iou:.1f}%  Oracle={or_iou:.1f}%  (Oracle - T1 = {or_iou - t1_iou:+.1f})")

    # Overall mIoU_fg
    for key, label in [("t1", "T1"), ("t2", "T2"), ("oracle", "Oracle")]:
        fg_ious = []
        for cls_id in [1, 2]:
            u = iou_acc[key]["union"][cls_id]
            fg_ious.append(iou_acc[key]["intersection"][cls_id] / u * 100 if u > 0 else 0)
        print(f"  {label} mIoU_fg = {sum(fg_ious) / 2:.1f}%")
        oracle_results[f"{key}_mIoU_fg"] = round(sum(fg_ious) / 2, 2)

    oracle_gain = oracle_results["oracle_mIoU_fg"] - oracle_results["t1_mIoU_fg"]
    print(f"\n  Oracle gain over T1: {oracle_gain:+.1f} mIoU_fg")
    oracle_results["oracle_gain_mIoU_fg"] = round(oracle_gain, 2)
    results["oracle_ensemble"] = oracle_results

    # 6. Image Oracle
    print("\n" + "=" * 70)
    print("  6. IMAGE ORACLE (per-image best-teacher selection)")
    print("=" * 70)
    img_oracle_results = {}

    # Macro IoU: average of per-class IoUs
    img_oracle_ious = []
    img_oracle_fg_ious = []
    for cls_id in range(NUM_CLASSES):
        u = img_oracle_iou["union"][cls_id]
        iou_val = img_oracle_iou["intersection"][cls_id] / u * 100 if u > 0 else 0.0
        img_oracle_ious.append(iou_val)
        if cls_id > 0:
            img_oracle_fg_ious.append(iou_val)
    img_oracle_macro = sum(img_oracle_ious) / len(img_oracle_ious) if img_oracle_ious else 0.0
    img_oracle_micro_inter = sum(img_oracle_iou["intersection"][c] for c in range(NUM_CLASSES))
    img_oracle_micro_union = sum(img_oracle_iou["union"][c] for c in range(NUM_CLASSES))
    img_oracle_micro = img_oracle_micro_inter / img_oracle_micro_union * 100 if img_oracle_micro_union > 0 else 0.0
    img_oracle_mIoU_fg = sum(img_oracle_fg_ious) / len(img_oracle_fg_ious) if img_oracle_fg_ious else 0.0

    print(f"  Image-oracle macro IoU (all classes):  {img_oracle_macro:.1f}%")
    print(f"  Image-oracle micro IoU (all classes):  {img_oracle_micro:.1f}%")
    print(f"  Image-oracle mIoU_fg:                  {img_oracle_mIoU_fg:.1f}%")
    for cls_id in range(NUM_CLASSES):
        name = CLASS_NAMES[cls_id]
        u = img_oracle_iou["union"][cls_id]
        iou_val = img_oracle_iou["intersection"][cls_id] / u * 100 if u > 0 else 0.0
        print(f"    {name}: {iou_val:.1f}%")

    img_oracle_results["macro_iou"] = round(img_oracle_macro, 2)
    img_oracle_results["micro_iou"] = round(img_oracle_micro, 2)
    img_oracle_results["mIoU_fg"] = round(img_oracle_mIoU_fg, 2)

    print(f"\n  Per-tier teacher selection distribution:")
    tier_sel = {}
    for tier_name in ["Easy", "Medium", "Hard"]:
        wins = img_oracle_tier_wins[tier_name]
        total_imgs = wins["t1"] + wins["t2"]
        if total_imgs == 0:
            continue
        t1_pct = wins["t1"] / total_imgs * 100
        t2_pct = wins["t2"] / total_imgs * 100
        print(f"    {tier_name:8s}:  T1 wins {wins['t1']:3d} ({t1_pct:.0f}%)  |  T2 wins {wins['t2']:3d} ({t2_pct:.0f}%)  |  total {total_imgs}")
        tier_sel[tier_name] = {"t1_wins": wins["t1"], "t2_wins": wins["t2"], "total": total_imgs}
    img_oracle_results["tier_selection"] = tier_sel
    results["image_oracle"] = img_oracle_results

    # 7. Component Oracle with Thin Crack Analysis
    print("\n" + "=" * 70)
    print("  7. COMPONENT ORACLE WITH THIN CRACK ANALYSIS")
    print("=" * 70)
    comp_oracle_results = {}
    for cls_id in [1, 2]:
        name = CLASS_NAMES[cls_id]
        total_c = comp_oracle["total"][cls_id]
        if total_c == 0:
            continue
        oracle_hit_c = comp_oracle["oracle_hit"][cls_id]
        oracle_recall = oracle_hit_c / total_c * 100
        both_c = comp_oracle["both_detected"][cls_id]
        t1_only_c = comp_oracle["t1_only"][cls_id]
        t2_only_c = comp_oracle["t2_only"][cls_id]
        neither_c = comp_oracle["neither"][cls_id]
        comp_oracle_results[name] = {
            "total_components": total_c,
            "oracle_recall_pct": round(oracle_recall, 2),
            "both_detected": both_c,
            "t1_only": t1_only_c,
            "t2_rescued": t2_only_c,
            "neither_detected": neither_c,
        }
        print(f"\n  {name}:")
        print(f"    GT components:           {total_c}")
        print(f"    Oracle CompR:            {oracle_recall:.1f}%  ({oracle_hit_c}/{total_c})")
        print(f"    Both detected:           {both_c}")
        print(f"    T1-only:                 {t1_only_c}")
        print(f"    T2-only (rescued):       {t2_only_c}")
        print(f"    Neither detected:        {neither_c}")

    # Thin crack analysis
    print(f"\n  Thin crack analysis (median skeleton width <= 2px):")
    thin_total = thin_crack["total"]
    if thin_total > 0:
        t1_thin_recall = thin_crack["t1_hit"] / thin_total * 100
        t2_thin_recall = thin_crack["t2_hit"] / thin_total * 100
        t2_rescue_rate = thin_crack["t2_rescued"] / thin_total * 100
        print(f"    Total thin crack components:  {thin_total}")
        print(f"    T1 recall on thin cracks:     {t1_thin_recall:.1f}%  ({thin_crack['t1_hit']}/{thin_total})")
        print(f"    T2 recall on thin cracks:     {t2_thin_recall:.1f}%  ({thin_crack['t2_hit']}/{thin_total})")
        print(f"    T2-rescued thin cracks:       {thin_crack['t2_rescued']}  ({t2_rescue_rate:.1f}%)")
        print(f"    Both detected:                {thin_crack['both_detected']}")
        print(f"    Neither detected:             {thin_crack['neither']}")
        comp_oracle_results["thin_crack"] = {
            "total": thin_total,
            "t1_recall_pct": round(t1_thin_recall, 2),
            "t2_recall_pct": round(t2_thin_recall, 2),
            "t2_rescue_rate_pct": round(t2_rescue_rate, 2),
            "t2_rescued": thin_crack["t2_rescued"],
            "both_detected": thin_crack["both_detected"],
            "neither_detected": thin_crack["neither"],
        }
    else:
        print(f"    No thin crack components found.")
        comp_oracle_results["thin_crack"] = {"total": 0}

    results["component_oracle"] = comp_oracle_results

    # Verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    if oracle_gain < 1.0:
        print(f"  Oracle gain = {oracle_gain:+.1f} < 1.0 points.")
        print("  T2 provides INSUFFICIENT complementary knowledge.")
        print("  Curriculum distillation is unlikely to recover meaningful gains.")
        verdict = "insufficient_complementarity"
    elif oracle_gain < 2.0:
        print(f"  Oracle gain = {oracle_gain:+.1f} (1-2 point range).")
        print("  T2 provides MARGINAL complementary knowledge.")
        print("  Curriculum distillation may help but gains will be modest.")
        verdict = "marginal_complementarity"
    else:
        print(f"  Oracle gain = {oracle_gain:+.1f} >= 2.0 points.")
        print("  T2 provides SUBSTANTIAL complementary knowledge.")
        print("  Current equal-weight full-stage distillation is suboptimal.")
        print("  Curriculum distillation is promising.")
        verdict = "substantial_complementarity"

    results["verdict"] = verdict
    results["oracle_gain_mIoU_fg"] = round(oracle_gain, 2)

    return results


def main():
    # Load val set
    val_file = SPLIT_DIR / "val.txt"
    if not val_file.exists():
        print(f"[ERROR] Val split not found: {val_file}")
        sys.exit(1)

    with open(val_file) as f:
        val_rels = [ln.strip() for ln in f if ln.strip()]

    print(f"[diag] Val set: {len(val_rels)} images")
    print(f"[diag] T1 checkpoint: {T1_CKPT}")
    print(f"[diag] T2 checkpoint: {T2_CKPT}")

    for ckpt in [T1_CKPT, T2_CKPT]:
        if not ckpt.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt}")
            sys.exit(1)

    device = _pick_device()
    print(f"[diag] Device: {device}")

    results = run_diagnosis(val_rels, C.DATA_ROOT, device)

    # Save
    out_path = SPLIT_DIR / "teacher_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[diag] Results saved to {out_path}")


if __name__ == "__main__":
    main()

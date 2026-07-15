"""Diagnose whether selective T2 rescue routing is feasible on the
balanced-group validation set.

Computes:
  1. Routing signal quality: AUC of delta_c / margin / entropy for detecting
     T2-better pixels among disagreement pixels
  2. Threshold sweep: coverage, precision, recall at candidate thresholds
  3. Replay mIoU: actual mIoU_fg when routing T2 predictions at each threshold
  4. Success gate: top-10% precision > 60% AND replay mIoU > T1 mIoU_fg

Usage (on RunPod, from Codes/):
    python scripts/confidence_routing_diagnosis.py

Outputs results to stdout and saves JSON to
    baseline_unet/splits/balanced_group_split/routing_diagnosis.json
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
import torch.nn.functional as F
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
# Helpers
# ---------------------------------------------------------------------------

def _compute_miou_fg(preds: np.ndarray, gts: np.ndarray) -> float:
    """Compute mIoU over foreground classes (1=crack, 2=spalling)."""
    ious = []
    for cls_id in [1, 2]:
        inter = int(((preds == cls_id) & (gts == cls_id)).sum())
        union = int(((preds == cls_id) | (gts == cls_id)).sum())
        if union > 0:
            ious.append(inter / union)
        else:
            ious.append(0.0)
    return sum(ious) / len(ious) * 100


def _pr_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, list, list]:
    """Compute precision-recall AUC. Returns (auc, precisions, recalls).
    Uses sklearn if available, otherwise manual computation."""
    try:
        from sklearn.metrics import precision_recall_curve, auc
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        return float(pr_auc), precision.tolist(), recall.tolist()
    except ImportError:
        pass

    # Manual computation
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp_cum = np.cumsum(sorted_labels)
    fp_cum = np.cumsum(1 - sorted_labels)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / max(sorted_labels.sum(), 1)
    # Prepend (recall=0, precision=1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    # Trapezoidal AUC
    pr_auc = float(np.trapz(precision, recall))
    return abs(pr_auc), precision.tolist(), recall.tolist()


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC AUC for a binary signal."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        pass

    # Manual: sort by descending score, compute via trapezoidal rule
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    n_pos = sorted_labels.sum()
    n_neg = len(sorted_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr_cum = np.cumsum(sorted_labels) / n_pos
    fpr_cum = np.cumsum(1 - sorted_labels) / n_neg
    tpr = np.concatenate([[0.0], tpr_cum])
    fpr = np.concatenate([[0.0], fpr_cum])
    return float(np.trapz(tpr, fpr))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_routing_diagnosis(
    val_rels: list[str],
    data_root: Path,
    device: str,
    routing_temperature: float = 1.0,
):
    # Load teachers
    print("[routing] Loading Teacher 1 (DSConv+SRL G1) ...")
    t1_model = _load_teacher("dscformer_srl_G1", T1_CKPT, device)
    print("[routing] Loading Teacher 2 (SAM-LoRA) ...")
    t2_model = _load_teacher("sam_lora_srl_SAM2", T2_CKPT, device)

    img_size = 512
    dataset = DamSegDataset(data_root, val_rels, img_size=img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Accumulators (collect all on CPU as numpy for threshold sweep)
    all_disagree = []       # bool
    all_delta_c = []        # float: t2_max_prob - t1_max_prob
    all_t2_better = []      # bool: t2_correct & ~t1_correct among disagree
    all_t1_pred = []        # int
    all_t2_pred = []        # int
    all_gt = []             # int
    all_tier = []           # str per pixel (for per-tier analysis, stored per image)

    # Per-class / per-tier disagree breakdown
    disagree_by_class = defaultdict(lambda: {"total": 0, "both_wrong": 0,
                                              "t2_better": 0, "t1_better": 0})
    disagree_by_tier = defaultdict(lambda: {"total": 0, "both_wrong": 0,
                                             "t2_better": 0, "t1_better": 0})

    # Routing signal arrays for AUC comparison
    all_margin_diff = []    # t2_margin - t1_margin
    all_entropy_diff = []   # t1_entropy - t2_entropy (higher = more confused T1)
    all_max_prob_diff = []  # same as delta_c but kept for clarity

    print(f"[routing] Processing {len(dataset)} images (temperature={routing_temperature}) ...")

    for images, masks, rels in loader:
        images = images.to(device, non_blocking=True)
        gt = masks.numpy()[0]  # (H, W)
        rel = rels[0]
        tier = rel.split("/")[0]

        # Get logits
        t1_logits = t1_model(images)  # (1, C, H, W)
        t2_logits = t2_model(images)

        # Compute routing probabilities at temperature
        t1_rp = F.softmax(t1_logits / routing_temperature, dim=1)  # (1, C, H, W)
        t2_rp = F.softmax(t2_logits / routing_temperature, dim=1)

        t1_rp_np = t1_rp.cpu().numpy()[0]  # (C, H, W)
        t2_rp_np = t2_rp.cpu().numpy()[0]

        # Predictions
        t1_pred = t1_rp_np.argmax(0)  # (H, W)
        t2_pred = t2_rp_np.argmax(0)

        # Disagree mask
        disagree = (t1_pred != t2_pred)

        # Max probabilities
        t1_max_prob = t1_rp_np.max(0)  # (H, W)
        t2_max_prob = t2_rp_np.max(0)
        delta_c = t2_max_prob - t1_max_prob

        # Correctness
        t1_correct = (t1_pred == gt)
        t2_correct = (t2_pred == gt)
        t2_better = t2_correct & ~t1_correct

        # Margin: difference between top-2 probabilities
        sorted_t1 = np.sort(t1_rp_np, axis=0)[::-1]  # descending along class dim
        sorted_t2 = np.sort(t2_rp_np, axis=0)[::-1]
        t1_margin = sorted_t1[0] - sorted_t1[1]  # (H, W)
        t2_margin = sorted_t2[0] - sorted_t2[1]
        margin_diff = t2_margin - t1_margin

        # Entropy
        eps = 1e-8
        t1_entropy = -(t1_rp_np * np.log(t1_rp_np + eps)).sum(0)  # (H, W)
        t2_entropy = -(t2_rp_np * np.log(t2_rp_np + eps)).sum(0)
        entropy_diff = t1_entropy - t2_entropy  # positive = T1 more confused

        # Accumulate (only disagreement pixels for signal arrays)
        flat_disagree = disagree.ravel()
        all_disagree.append(flat_disagree)
        all_delta_c.append(delta_c.ravel())
        all_t2_better.append(t2_better.ravel())
        all_t1_pred.append(t1_pred.ravel())
        all_t2_pred.append(t2_pred.ravel())
        all_gt.append(gt.ravel())
        all_max_prob_diff.append(delta_c.ravel())
        all_margin_diff.append(margin_diff.ravel())
        all_entropy_diff.append(entropy_diff.ravel())

        # Per-class disagree breakdown
        for cls_id in range(NUM_CLASSES):
            cls_mask = (gt == cls_id) & disagree
            n = int(cls_mask.sum())
            if n == 0:
                continue
            name = CLASS_NAMES[cls_id]
            disagree_by_class[name]["total"] += n
            disagree_by_class[name]["both_wrong"] += int((cls_mask & ~t1_correct & ~t2_correct).sum())
            disagree_by_class[name]["t2_better"] += int((cls_mask & t2_better).sum())
            disagree_by_class[name]["t1_better"] += int((cls_mask & t1_correct & ~t2_correct).sum())

        # Per-tier disagree breakdown
        n_d = int(disagree.sum())
        if n_d > 0:
            disagree_by_tier[tier]["total"] += n_d
            disagree_by_tier[tier]["both_wrong"] += int((disagree & ~t1_correct & ~t2_correct).sum())
            disagree_by_tier[tier]["t2_better"] += int((disagree & t2_better).sum())
            disagree_by_tier[tier]["t1_better"] += int((disagree & t1_correct & ~t2_correct).sum())

    # Concatenate all
    all_disagree = np.concatenate(all_disagree)
    all_delta_c = np.concatenate(all_delta_c)
    all_t2_better = np.concatenate(all_t2_better)
    all_t1_pred = np.concatenate(all_t1_pred)
    all_t2_pred = np.concatenate(all_t2_pred)
    all_gt = np.concatenate(all_gt)
    all_max_prob_diff = np.concatenate(all_max_prob_diff)
    all_margin_diff = np.concatenate(all_margin_diff)
    all_entropy_diff = np.concatenate(all_entropy_diff)

    total_pixels = len(all_disagree)
    n_disagree = int(all_disagree.sum())

    results = {}

    # -----------------------------------------------------------------------
    # 1. Disagreement pixel breakdown
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  1. DISAGREEMENT PIXEL BREAKDOWN")
    print("=" * 70)
    print(f"  Total pixels:       {total_pixels:,}")
    print(f"  Disagree pixels:    {n_disagree:,} ({n_disagree / total_pixels * 100:.2f}%)")

    n_t2_better_all = int((all_disagree & all_t2_better).sum())
    n_t1_better_disagree = int((all_disagree & (all_t1_pred == all_gt) & (all_t2_pred != all_gt)).sum())
    n_both_wrong_disagree = int((all_disagree & (all_t1_pred != all_gt) & (all_t2_pred != all_gt)).sum())
    print(f"\n  Among disagreement pixels:")
    print(f"    T2 better (T2 correct, T1 wrong): {n_t2_better_all:,} ({n_t2_better_all / max(n_disagree, 1) * 100:.1f}%)")
    print(f"    T1 better (T1 correct, T2 wrong): {n_t1_better_disagree:,} ({n_t1_better_disagree / max(n_disagree, 1) * 100:.1f}%)")
    print(f"    Both wrong:                       {n_both_wrong_disagree:,} ({n_both_wrong_disagree / max(n_disagree, 1) * 100:.1f}%)")

    results["disagreement_overview"] = {
        "total_pixels": int(total_pixels),
        "disagree_pixels": int(n_disagree),
        "disagree_pct": round(n_disagree / total_pixels * 100, 2),
        "t2_better_among_disagree_pct": round(n_t2_better_all / max(n_disagree, 1) * 100, 2),
        "t1_better_among_disagree_pct": round(n_t1_better_disagree / max(n_disagree, 1) * 100, 2),
        "both_wrong_among_disagree_pct": round(n_both_wrong_disagree / max(n_disagree, 1) * 100, 2),
    }

    # Per-class breakdown
    print("\n  Per GT class:")
    class_breakdown = {}
    for name in ["background", "crack", "spalling"]:
        info = disagree_by_class.get(name, {"total": 0, "both_wrong": 0, "t2_better": 0, "t1_better": 0})
        t = info["total"]
        if t == 0:
            continue
        class_breakdown[name] = {
            "disagree_pixels": t,
            "both_wrong_pct": round(info["both_wrong"] / t * 100, 2),
            "t2_better_pct": round(info["t2_better"] / t * 100, 2),
            "t1_better_pct": round(info["t1_better"] / t * 100, 2),
        }
        print(f"    {name}: {t:,} disagree px  |  T2-better={info['t2_better'] / t * 100:.1f}%  T1-better={info['t1_better'] / t * 100:.1f}%  both_wrong={info['both_wrong'] / t * 100:.1f}%")
    results["per_class_disagree"] = class_breakdown

    # Per-tier breakdown
    print("\n  Per tier:")
    tier_breakdown = {}
    for tier in ["Easy", "Medium", "Hard"]:
        info = disagree_by_tier.get(tier, {"total": 0, "both_wrong": 0, "t2_better": 0, "t1_better": 0})
        t = info["total"]
        if t == 0:
            continue
        tier_breakdown[tier] = {
            "disagree_pixels": t,
            "both_wrong_pct": round(info["both_wrong"] / t * 100, 2),
            "t2_better_pct": round(info["t2_better"] / t * 100, 2),
            "t1_better_pct": round(info["t1_better"] / t * 100, 2),
        }
        print(f"    {tier}: {t:,} disagree px  |  T2-better={info['t2_better'] / t * 100:.1f}%  T1-better={info['t1_better'] / t * 100:.1f}%  both_wrong={info['both_wrong'] / t * 100:.1f}%")
    results["per_tier_disagree"] = tier_breakdown

    # -----------------------------------------------------------------------
    # 2. Routing signal AUC comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  2. ROUTING SIGNAL AUC (among disagreement pixels)")
    print("=" * 70)

    # Restrict to disagreement pixels
    d_mask = all_disagree.astype(bool)
    d_labels = all_t2_better[d_mask].astype(np.float32)
    positive_prevalence = float(d_labels.mean()) if len(d_labels) > 0 else 0.0
    print(f"  Positive prevalence (T2-better among disagree): {positive_prevalence * 100:.1f}%")
    print(f"  Random baseline AUC: {positive_prevalence:.3f}")

    signal_results = {}
    signals = {
        "max_prob_diff (delta_c)": all_max_prob_diff[d_mask],
        "margin_diff": all_margin_diff[d_mask],
        "entropy_diff (T1-T2)": all_entropy_diff[d_mask],
    }

    for sig_name, sig_values in signals.items():
        if len(sig_values) == 0 or d_labels.sum() == 0:
            auc_val = 0.5
        else:
            auc_val = _roc_auc(sig_values, d_labels)
        signal_results[sig_name] = round(auc_val, 4)
        print(f"  {sig_name:30s}  ROC-AUC = {auc_val:.4f}")

    results["signal_auc"] = signal_results
    results["positive_prevalence"] = round(positive_prevalence, 4)

    # -----------------------------------------------------------------------
    # 3. PR curve + AUC for delta_c signal
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  3. PR CURVE (delta_c signal, among disagreement pixels)")
    print("=" * 70)

    if len(d_labels) > 0 and d_labels.sum() > 0:
        pr_auc, pr_precisions, pr_recalls = _pr_auc(
            all_max_prob_diff[d_mask], d_labels
        )
    else:
        pr_auc = 0.0
        pr_precisions, pr_recalls = [], []

    print(f"  PR-AUC (delta_c): {pr_auc:.4f}")
    results["pr_auc_delta_c"] = round(pr_auc, 4)

    # -----------------------------------------------------------------------
    # 4. Threshold sweep
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  4. THRESHOLD SWEEP")
    print("=" * 70)
    candidate_thresholds = [0.00, 0.02, 0.05, 0.10, 0.15]
    threshold_results = {}

    print(f"  {'Thr':>6s}  {'Cov_dis%':>8s}  {'Cov_all%':>8s}  {'Prec%':>8s}  {'Recall%':>8s}")
    print("  " + "-" * 46)

    for thr in candidate_thresholds:
        # Route T2 where: disagree AND delta_c > threshold
        route_t2 = all_disagree & (all_delta_c > thr)
        n_routed = int(route_t2.sum())

        # Coverage
        cov_disagree = n_routed / max(n_disagree, 1) * 100
        cov_all = n_routed / total_pixels * 100

        # Among routed pixels: how many are T2-better?
        n_t2_better_routed = int((route_t2 & all_t2_better).sum())
        precision = n_t2_better_routed / max(n_routed, 1) * 100

        # Among all T2-better pixels: how many are routed?
        n_t2_better_total = int((all_disagree & all_t2_better).sum())
        recall = n_t2_better_routed / max(n_t2_better_total, 1) * 100

        threshold_results[str(thr)] = {
            "routed_pixels": int(n_routed),
            "coverage_among_disagree_pct": round(cov_disagree, 2),
            "coverage_among_all_pct": round(cov_all, 2),
            "t2_better_precision_pct": round(precision, 2),
            "t2_better_recall_pct": round(recall, 2),
        }

        print(f"  {thr:6.2f}  {cov_disagree:8.2f}  {cov_all:8.2f}  {precision:8.2f}  {recall:8.2f}")

    results["threshold_sweep"] = threshold_results

    # -----------------------------------------------------------------------
    # 5. Replay mIoU per threshold
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  5. REPLAY mIoU_fg PER THRESHOLD")
    print("=" * 70)

    # T1-only mIoU_fg baseline
    t1_miou_fg = _compute_miou_fg(all_t1_pred, all_gt)
    t2_miou_fg = _compute_miou_fg(all_t2_pred, all_gt)
    print(f"  T1-only mIoU_fg: {t1_miou_fg:.2f}%")
    print(f"  T2-only mIoU_fg: {t2_miou_fg:.2f}%")

    replay_results = {"t1_only": round(t1_miou_fg, 2), "t2_only": round(t2_miou_fg, 2)}

    print(f"\n  {'Thr':>6s}  {'mIoU_fg%':>10s}  {'Delta':>8s}")
    print("  " + "-" * 30)

    best_replay_miou = t1_miou_fg
    best_thr = None

    for thr in candidate_thresholds:
        route_t2 = all_disagree & (all_delta_c > thr)
        combined = all_t1_pred.copy()
        combined[route_t2] = all_t2_pred[route_t2]
        miou_fg = _compute_miou_fg(combined, all_gt)
        delta = miou_fg - t1_miou_fg
        replay_results[str(thr)] = {"mIoU_fg": round(miou_fg, 2), "delta": round(delta, 2)}
        print(f"  {thr:6.2f}  {miou_fg:10.2f}  {delta:+8.2f}")

        if miou_fg > best_replay_miou:
            best_replay_miou = miou_fg
            best_thr = thr

    results["replay_miou"] = replay_results

    # -----------------------------------------------------------------------
    # 6. Success gate
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  6. SUCCESS GATE")
    print("=" * 70)

    # Top-10% precision: among disagree pixels, take the top 10% by delta_c
    # and check what fraction are T2-better
    if n_disagree > 0:
        d_delta_c = all_delta_c[d_mask]
        k = max(1, int(0.1 * len(d_delta_c)))
        top_indices = np.argpartition(d_delta_c, -k)[-k:]
        top_labels = d_labels[top_indices]
        top10_precision = float(top_labels.mean()) * 100
    else:
        top10_precision = 0.0

    replay_beats_t1 = best_replay_miou > t1_miou_fg

    gate_pass = (top10_precision > 60.0) and replay_beats_t1

    print(f"  Top-10% precision:   {top10_precision:.1f}%  (gate: > 60%)")
    print(f"  Best replay mIoU:    {best_replay_miou:.2f}%  at thr={best_thr}  (T1={t1_miou_fg:.2f}%)")
    print(f"  Replay beats T1:     {'YES' if replay_beats_t1 else 'NO'}")
    print(f"\n  >>> GATE: {'PASS' if gate_pass else 'FAIL'} <<<")

    if gate_pass:
        print("  Selective T2 routing is FEASIBLE.")
    else:
        print("  Selective T2 routing is NOT feasible with current signals.")

    results["success_gate"] = {
        "top10_precision_pct": round(top10_precision, 2),
        "best_replay_miou": round(best_replay_miou, 2),
        "best_threshold": best_thr,
        "t1_miou_fg": round(t1_miou_fg, 2),
        "replay_beats_t1": replay_beats_t1,
        "gate_pass": gate_pass,
    }

    return results


def main():
    # Load val set
    val_file = SPLIT_DIR / "val.txt"
    if not val_file.exists():
        print(f"[ERROR] Val split not found: {val_file}")
        sys.exit(1)

    with open(val_file) as f:
        val_rels = [ln.strip() for ln in f if ln.strip()]

    print(f"[routing] Val set: {len(val_rels)} images")
    print(f"[routing] T1 checkpoint: {T1_CKPT}")
    print(f"[routing] T2 checkpoint: {T2_CKPT}")

    for ckpt in [T1_CKPT, T2_CKPT]:
        if not ckpt.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt}")
            sys.exit(1)

    device = _pick_device()
    print(f"[routing] Device: {device}")

    results = run_routing_diagnosis(val_rels, C.DATA_ROOT, device)

    # Save
    out_path = SPLIT_DIR / "routing_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[routing] Results saved to {out_path}")


if __name__ == "__main__":
    main()

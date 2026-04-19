"""Step-5 Full Method: Dynamic Difficulty-Aware Curriculum + Boundary Refinement.

SegFormer-B2 backbone with:
  Module 1: Online difficulty estimation (EMA loss + uncertainty + boundary + sparsity)
  Module 2: Class-aware dynamic sampling (tier gating + spalling/crack bonuses)
  Module 3: Crack boundary refinement head + Tversky loss

Usage:
    python -m full_method.train --dry-run
    python -m full_method.train --epochs 5          # probe
    python -m full_method.train                     # full 100 epochs
    python -m full_method.train --resume full_method/runs/segformer_b2_full_512/last.pt
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Workaround: cuDNN may fail to initialize on some CUDA 12.x / driver combos.
torch.backends.cudnn.enabled = False

from baseline_unet.dataset import (
    build_transforms,
    compute_class_weights,
    read_split_file,
)
from baseline_unet.visualize import pick_viz_samples, save_preview
from baseline_deeplab.metrics import SegMetricsBF1, format_metrics

from full_method import config as C
from full_method.config import ABLATION_PRESETS, apply_preset
from full_method.dataset import FullMethodDataset, build_records, dict_collate
from full_method.model import SegFormerWithBoundary, DSCformerDam, _PreviewWrapper
from full_method.losses import CompositeLoss
from full_method.difficulty import DifficultyEstimator, SampleState
from full_method.sampler import TierAwareDynamicSampler
from full_method.scheduler import CurriculumScheduler, AdaptivePacer, ClassLossScheduler


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(pref: str) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_train_loader(records: List[Dict], sampler: TierAwareDynamicSampler,
                       cfg: C.RunCfg, device: str) -> DataLoader:
    ds = FullMethodDataset(C.DATA_ROOT, records,
                           build_transforms(cfg.img_size, train=True,
                                          aug_level=getattr(cfg, 'aug_level', 'basic')),
                           compute_skel=cfg.use_srl_loss)
    pin = (device == "cuda")
    return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler,
                      num_workers=4, pin_memory=pin, drop_last=False,
                      collate_fn=dict_collate)


def build_val_loader(val_files: List[str], cfg: C.RunCfg, device: str) -> DataLoader:
    """Val/test loader uses plain records (tier/has_spalling not needed)."""
    records = [{"id": f, "rel": f, "tier": 0, "has_spalling": False} for f in val_files]
    ds = FullMethodDataset(C.DATA_ROOT, records,
                           build_transforms(cfg.img_size, train=False))
    pin = (device == "cuda")
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                      num_workers=4, pin_memory=pin, drop_last=False,
                      collate_fn=dict_collate)


# ---------------------------------------------------------------------------
# OOM probe (adapted for dict-output model)
# ---------------------------------------------------------------------------

def oom_probe(model: nn.Module, cfg: C.RunCfg, device: str) -> bool:
    try:
        if device == "cuda":
            _w = torch.randn(1, 3, 8, 8, device=device)
            _ = torch.nn.functional.conv2d(_w, torch.randn(3, 3, 3, 3, device=device), padding=1)
            del _w, _
        x = torch.zeros(cfg.batch_size, 3, cfg.img_size, cfg.img_size, device=device)
        y = torch.zeros(cfg.batch_size, cfg.img_size, cfg.img_size,
                        device=device, dtype=torch.long)
        outputs = model(x)
        seg_logits = F.interpolate(outputs["seg_logits"], size=y.shape[-2:],
                                   mode="bilinear", align_corners=False)
        loss = F.cross_entropy(seg_logits, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        del x, y, outputs, seg_logits, loss
        if device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        return True
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "mps" in msg or "allocat" in msg:
            print(f"[probe] OOM-like error: {e}")
            return False
        raise


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

METRIC_KEYS: List[str] = [
    "epoch", "split", "train_loss", "val_loss",
    "loss_ce", "loss_dice", "loss_tversky", "loss_bd", "loss_cldice", "loss_snake",
    "IoU_background", "IoU_crack", "IoU_spalling",
    "Dice_background", "Dice_crack", "Dice_spalling",
    "mIoU_fg", "mIoU_all", "pixel_acc",
    "BF1_crack", "BF1_spalling", "BF1_fg_mean",
    "sampled_t0", "sampled_t1", "sampled_t2",
]


def train_one_epoch(model, loader, optimizer, criterion: CompositeLoss, device,
                    metrics: SegMetricsBF1, grad_accum: int,
                    sample_bank: Dict[str, SampleState],
                    estimator: DifficultyEstimator,
                    curriculum_scheduler: CurriculumScheduler,
                    epoch: int, total_epochs: int,
                    cfg: C.RunCfg = None,
                    scaler: torch.amp.GradScaler = None,
                    mac_ce_multipliers: tuple = None,
                    mac_topo_multiplier: float = 1.0) -> Dict[str, float]:
    model.train()
    metrics.reset()
    loss_sum = 0.0
    loss_parts = {"loss_ce": 0.0, "loss_dice": 0.0, "loss_tversky": 0.0, "loss_bd": 0.0, "loss_cldice": 0.0, "loss_snake": 0.0}
    n_batches = 0
    total_steps = len(loader)
    log_every = max(1, total_steps // 10)
    t_start = time.time()
    use_amp = scaler is not None

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        imgs = batch["image"].to(device, non_blocking=True).float()
        masks = batch["mask"].to(device, non_blocking=True).long()
        sample_ids = batch["sample_id"]

        # Crack skeleton for SRL (if available)
        crack_skel = batch.get("crack_skel")
        if crack_skel is not None:
            crack_skel = crack_skel.to(device, non_blocking=True).float()

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(imgs)

            # Compute per-sample loss weights from difficulty scores
            if cfg is not None and cfg.use_dynamic_loss_reweight:
                diffs = torch.tensor([sample_bank.get(sid, SampleState()).difficulty
                                      for sid in sample_ids], device=device)
                d_norm = torch.sigmoid(diffs)  # z-scored → [0,1], batch-invariant
                sample_weights = 1.0 + cfg.loss_reweight_lambda * d_norm
                sample_weights = sample_weights / sample_weights.mean()
            else:
                sample_weights = None

            total_loss, info = criterion(outputs, masks, curriculum_scheduler, epoch,
                                         sample_weights=sample_weights,
                                         crack_skel=crack_skel,
                                         mac_ce_multipliers=mac_ce_multipliers,
                                         mac_topo_multiplier=mac_topo_multiplier)
            total_loss = total_loss / grad_accum

        if use_amp:
            scaler.scale(total_loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            total_loss.backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        loss_sum += float(total_loss.detach().cpu()) * grad_accum
        for k in loss_parts:
            loss_parts[k] += info[k]
        n_batches += 1

        # Update metrics with seg_logits
        seg_logits = F.interpolate(outputs["seg_logits"].float(), masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
        metrics.update(seg_logits, masks)

        # Update sample bank (only when dynamic difficulty is enabled)
        _use_dynamic = cfg is None or cfg.use_dynamic_difficulty
        if _use_dynamic:
            ps_ce = info["per_sample_ce"]    # (B,) cpu tensor
            ps_ent = info["per_sample_ent"]  # (B,) cpu tensor
            masks_np = masks.detach().cpu().numpy()
            for b_idx in range(len(sample_ids)):
                sid = sample_ids[b_idx]
                if sid not in sample_bank:
                    sample_bank[sid] = SampleState()
                estimator.update(
                    sample_bank[sid],
                    float(ps_ce[b_idx]),
                    float(ps_ent[b_idx]),
                    masks_np[b_idx],
                )
                # MAC: update per-class EMA losses
                if cfg is not None and cfg.use_mac and "per_sample_ce_crack" in info:
                    estimator.update_class_loss(
                        sample_bank[sid],
                        float(info["per_sample_ce_crack"][b_idx]),
                        float(info["per_sample_ce_spalling"][b_idx]),
                    )

        done = step + 1
        if done % log_every == 0 or done == total_steps:
            elapsed = time.time() - t_start
            eta = elapsed / done * (total_steps - done)
            run_tag = cfg.name if cfg else "full"
            print(f"  [{run_tag}] [epoch {epoch}/{total_epochs}] batch {done}/{total_steps}"
                  f" ({done*100//total_steps}%) loss={loss_sum/n_batches:.4f}"
                  f" elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    # Flush any tail micro-batches
    if n_batches % grad_accum != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = loss_sum / max(n_batches, 1)
    m = metrics.compute()
    m["loss"] = avg_loss
    for k in loss_parts:
        m[k] = loss_parts[k] / max(n_batches, 1)
    return m


@torch.no_grad()
def evaluate(model, loader, criterion: CompositeLoss, device,
             metrics: SegMetricsBF1,
             curriculum_scheduler: CurriculumScheduler,
             epoch: int, use_amp: bool = False) -> Dict[str, float]:
    model.eval()
    metrics.reset()
    loss_sum = 0.0
    n_batches = 0
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True).float()
        masks = batch["mask"].to(device, non_blocking=True).long()

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(imgs)
            total_loss, _ = criterion(outputs, masks, curriculum_scheduler, epoch)
        loss_sum += float(total_loss.detach().cpu())
        n_batches += 1

        seg_logits = F.interpolate(outputs["seg_logits"].float(), masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
        metrics.update(seg_logits, masks)

    m = metrics.compute()
    m["loss"] = loss_sum / max(n_batches, 1)
    return m


def write_metrics_row(csv_path: Path, row: Dict[str, object]) -> None:
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_KEYS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in METRIC_KEYS})


# ---------------------------------------------------------------------------
# Curves rendering
# ---------------------------------------------------------------------------

def render_curves(csv_path: Path, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_list: List[int] = []
    train_loss: List[float] = []
    val_loss: List[float] = []
    iou_crack: List[float] = []
    iou_spalling: List[float] = []
    miou_fg: List[float] = []
    bf1_fg: List[float] = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["epoch"])
            except (ValueError, KeyError):
                continue
            epochs_list.append(ep)
            train_loss.append(float(row.get("train_loss") or "nan"))
            val_loss.append(float(row.get("val_loss") or "nan"))
            iou_crack.append(float(row.get("IoU_crack") or "nan"))
            iou_spalling.append(float(row.get("IoU_spalling") or "nan"))
            miou_fg.append(float(row.get("mIoU_fg") or "nan"))
            try:
                bf1_fg.append(float(row.get("BF1_fg_mean") or "nan"))
            except ValueError:
                bf1_fg.append(float("nan"))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(epochs_list, train_loss, label="train")
    axs[0].plot(epochs_list, val_loss, label="val")
    axs[0].set_title("loss"); axs[0].set_xlabel("epoch"); axs[0].legend()
    axs[1].plot(epochs_list, iou_crack, label="IoU_crack")
    axs[1].plot(epochs_list, iou_spalling, label="IoU_spalling")
    axs[1].plot(epochs_list, miou_fg, label="mIoU_fg")
    axs[1].set_title("IoU / mIoU_fg (val)"); axs[1].set_xlabel("epoch"); axs[1].legend()
    axs[2].plot(epochs_list, bf1_fg, label="BF1_fg_mean")
    axs[2].set_title("Boundary F1 (val)"); axs[2].set_xlabel("epoch"); axs[2].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override default epochs (e.g. 5 for probe)")
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint path to resume from")
    # Ablation controls
    parser.add_argument("--ablation", type=str, default=None,
                        choices=sorted(ABLATION_PRESETS.keys()),
                        help="named ablation preset (e.g. A2, A3, B0, S1, S2)")
    parser.add_argument("--no-dynamic-difficulty", action="store_true")
    parser.add_argument("--no-class-sampling-bonus", action="store_true")
    parser.add_argument("--no-class-loss-schedule", action="store_true")
    parser.add_argument("--no-boundary-loss", action="store_true")
    parser.add_argument("--no-tversky-loss", action="store_true")
    parser.add_argument("--no-cldice-loss", action="store_true")
    parser.add_argument("--diff-alpha", type=float, default=None)
    parser.add_argument("--diff-beta", type=float, default=None)
    parser.add_argument("--diff-gamma", type=float, default=None)
    parser.add_argument("--diff-delta", type=float, default=None)
    parser.add_argument("--name", type=str, default=None,
                        help="override run directory name")
    # New soft curriculum / reweight controls
    parser.add_argument("--no-soft-curriculum", action="store_true")
    parser.add_argument("--no-dynamic-loss-reweight", action="store_true")
    parser.add_argument("--loss-reweight-lambda", type=float, default=None)
    args = parser.parse_args()

    # Config resolution: defaults -> preset -> CLI flags -> CLI name
    cfg = C.RunCfg()
    if args.ablation is not None:
        apply_preset(cfg, args.ablation)
    if args.no_dynamic_difficulty:
        cfg.use_dynamic_difficulty = False
    if args.no_class_sampling_bonus:
        cfg.use_class_sampling_bonus = False
    if args.no_class_loss_schedule:
        cfg.use_class_loss_schedule = False
    if args.no_boundary_loss:
        cfg.use_boundary_loss = False
    if args.no_tversky_loss:
        cfg.use_tversky_loss = False
    if args.no_cldice_loss:
        cfg.use_cldice_loss = False
    if args.diff_alpha is not None:
        cfg.diff_alpha = args.diff_alpha
    if args.diff_beta is not None:
        cfg.diff_beta = args.diff_beta
    if args.diff_gamma is not None:
        cfg.diff_gamma = args.diff_gamma
    if args.diff_delta is not None:
        cfg.diff_delta = args.diff_delta
    if args.no_soft_curriculum:
        cfg.use_soft_curriculum = False
    if args.no_dynamic_loss_reweight:
        cfg.use_dynamic_loss_reweight = False
    if args.loss_reweight_lambda is not None:
        cfg.loss_reweight_lambda = args.loss_reweight_lambda
    if args.name is not None:
        cfg.name = args.name
    if args.grad_accum is not None:
        cfg.grad_accum = args.grad_accum
    total_epochs = args.epochs if args.epochs is not None else cfg.epochs

    set_seed(C.SEED)
    device = pick_device(C.DEVICE)

    # cuDNN warmup / fallback (RunPod compatibility)
    if device == "cuda":
        try:
            _w = torch.randn(1, 3, 8, 8, device=device)
            _ = torch.nn.functional.conv2d(_w, torch.randn(3, 3, 3, 3, device=device), padding=1)
            del _w, _
            print("[train] cuDNN warmup OK")
        except RuntimeError as e:
            print(f"[train] cuDNN warmup failed ({e}); disabling cuDNN")
            torch.backends.cudnn.enabled = False

    import torch as _t
    try:
        import transformers as _tf
        tf_v = _tf.__version__
    except Exception:
        tf_v = "?"
    print(f"[train] full_method ({cfg.name})")
    print(f"[train] device={device}  torch={_t.__version__}  transformers={tf_v}")
    print(f"[train] cfg: pretrained={cfg.pretrained} img={cfg.img_size} bs={cfg.batch_size}"
          f" grad_accum={cfg.grad_accum} epochs={total_epochs} lr={cfg.lr}"
          f" warmup={cfg.warmup_epochs}")
    print(f"[train] difficulty: alpha={cfg.diff_alpha} beta={cfg.diff_beta}"
          f" gamma={cfg.diff_gamma} delta={cfg.diff_delta}"
          f" ema={cfg.diff_ema} tau={cfg.diff_tau}")
    print(f"[train] bonuses: spalling={cfg.spalling_bonus}"
          f" late_hard_crack={cfg.late_hard_crack_bonus}")
    print(f"[train] ablation switches: dynamic_difficulty={cfg.use_dynamic_difficulty}"
          f" class_sampling_bonus={cfg.use_class_sampling_bonus}"
          f" class_loss_schedule={cfg.use_class_loss_schedule}"
          f" boundary_loss={cfg.use_boundary_loss}"
          f" tversky_loss={cfg.use_tversky_loss}")
    print(f"[train] new switches: soft_curriculum={cfg.use_soft_curriculum}"
          f" softmax_sampling={cfg.use_softmax_sampling}"
          f" dynamic_loss_reweight={cfg.use_dynamic_loss_reweight}"
          f" (lambda={cfg.loss_reweight_lambda})"
          f" soft_boundary={cfg.use_soft_boundary_schedule}"
          f" (start={cfg.boundary_start_ratio} max={cfg.boundary_max_weight})")
    print(f"[train] cldice: use={cfg.use_cldice_loss}"
          f" weight={cfg.cldice_weight} start_epoch={cfg.cldice_start_epoch}"
          f" iters={cfg.cldice_iters}")
    print(f"[train] srl: use={cfg.use_srl_loss}"
          f" weight={cfg.cldice_weight} start_epoch={cfg.cldice_start_epoch}")
    print(f"[train] snake_aux: use={cfg.use_snake_aux_loss}"
          f" weight={cfg.snake_aux_weight}")
    print(f"[train] model_type={cfg.model_type}"
          f" snake_channels={cfg.snake_channels} snake_kernel={cfg.snake_kernel_size}")
    print(f"[train] no_curriculum={cfg.no_curriculum}")
    print(f"[train] competence: hard={cfg.use_competence_curriculum}"
          f" soft={cfg.use_competence_soft_mixing}"
          f" c0={cfg.competence_c0} duration={cfg.competence_duration}"
          f" floors=[{cfg.competence_floor_easy},{cfg.competence_floor_medium},{cfg.competence_floor_hard}]")
    print(f"[train] MAC: use={cfg.use_mac} morph_diff={cfg.use_mac_morph_difficulty}"
          f" adaptive_pacing={cfg.use_mac_adaptive_pacing}"
          f" class_loss={cfg.use_mac_class_loss}"
          f" diff_gamma={cfg.mac_diff_gamma}"
          f" weights=[{cfg.mac_morph_width_w},{cfg.mac_morph_topo_w},"
          f"{cfg.mac_morph_prox_w},{cfg.mac_morph_sparse_w}]")

    rdir = C.run_dir(cfg)
    samples_dir = rdir / "samples"

    # ----- read split files -----
    all_train_files = read_split_file(C.SPLIT_FILES["train"])
    val_files = read_split_file(C.SPLIT_FILES["val"])
    test_files = read_split_file(C.SPLIT_FILES["test"])
    print(f"[train] sizes: train={len(all_train_files)} val={len(val_files)} test={len(test_files)}")

    # ----- build records -----
    print("[train] building records (scanning masks for has_spalling) ...")
    records = build_records(all_train_files, C.DATA_ROOT)
    tier_counts = {0: 0, 1: 0, 2: 0}
    sp_count = 0
    for r in records:
        tier_counts[r["tier"]] += 1
        if r["has_spalling"]:
            sp_count += 1
    print(f"[train] records: {tier_counts}  has_spalling={sp_count}/{len(records)}")

    # ----- class weights -----
    raw_freq, smoothed, clipped = compute_class_weights(C.DATA_ROOT, all_train_files)
    print("[train] raw pixel frequency:", dict(zip(C.CLASS_NAMES, [float(x) for x in raw_freq])))
    print("[train] clipped weights    :", dict(zip(C.CLASS_NAMES, [float(x) for x in clipped])))
    w = torch.tensor(clipped, dtype=torch.float32)

    # ----- init sample bank + estimator + scheduler + sampler -----
    sample_bank: Dict[str, SampleState] = {r["id"]: SampleState() for r in records}

    estimator = DifficultyEstimator(
        alpha=cfg.diff_alpha, beta=cfg.diff_beta,
        gamma=cfg.diff_gamma, delta=cfg.diff_delta,
        ema_decay=cfg.diff_ema,
        use_mac_morph_difficulty=cfg.use_mac_morph_difficulty,
        mac_diff_gamma=cfg.mac_diff_gamma,
    )

    curriculum_scheduler = CurriculumScheduler(total_epochs, cfg=cfg)

    sampler = TierAwareDynamicSampler(
        records, sample_bank,
        tau=cfg.diff_tau,
        spalling_bonus=cfg.spalling_bonus if cfg.use_class_sampling_bonus else 0.0,
        late_hard_crack_bonus=cfg.late_hard_crack_bonus if cfg.use_class_sampling_bonus else 0.0,
        enable_dynamic=cfg.use_dynamic_difficulty,
        use_soft_curriculum=cfg.use_soft_curriculum,
        use_softmax_sampling=cfg.use_softmax_sampling,
        no_curriculum=cfg.no_curriculum,
    )

    # ----- MAC: morph cache + adaptive pacer + class loss scheduler -----
    mac_pacer = None
    mac_class_sched = None
    if cfg.use_mac:
        from full_method.morphology import precompute_morph_cache, MorphFeatures
        morph_cache_path = C.DATA_ROOT / ".morph_cache.pkl"
        morph_cache = precompute_morph_cache(records, C.DATA_ROOT, morph_cache_path)

        # Compute per-sample morph_difficulty from features
        w1 = cfg.mac_morph_width_w
        w2 = cfg.mac_morph_topo_w
        w3 = cfg.mac_morph_prox_w
        w4 = cfg.mac_morph_sparse_w
        # Collect raw feature vectors for z-score normalization
        sids = [r["id"] for r in records]
        raw_inv_width = np.array([1.0 / (morph_cache[s].crack_mean_width + 1e-6)
                                  if morph_cache[s].has_crack else 0.0 for s in sids])
        raw_junc = np.array([morph_cache[s].junction_density for s in sids])
        raw_prox = np.array([morph_cache[s].crack_spalling_proximity for s in sids])
        raw_comp = np.array([np.log(morph_cache[s].crack_components + 1) for s in sids])

        def _zscore(arr):
            return (arr - arr.mean()) / (arr.std() + 1e-6)

        z_iw = _zscore(raw_inv_width)
        z_junc = _zscore(raw_junc)
        z_prox = _zscore(raw_prox)
        z_comp = _zscore(raw_comp)

        for i, sid in enumerate(sids):
            morph_d = float(w1 * z_iw[i] + w2 * z_junc[i] + w3 * z_prox[i] + w4 * z_comp[i])
            estimator.init_morph(sample_bank[sid], morph_d)

        print(f"[MAC] morph difficulty initialized for {len(sids)} samples")
        morph_ds = [sample_bank[s].morph_difficulty for s in sids]
        print(f"[MAC] morph_d stats: mean={np.mean(morph_ds):.3f} std={np.std(morph_ds):.3f}"
              f" min={np.min(morph_ds):.3f} max={np.max(morph_ds):.3f}")

        if cfg.use_mac_adaptive_pacing:
            mac_pacer = AdaptivePacer(cfg)
            print("[MAC] adaptive pacer enabled")
        if cfg.use_mac_class_loss:
            mac_class_sched = ClassLossScheduler(cfg)
            print("[MAC] class loss scheduler enabled")

    # ----- build loaders -----
    train_loader = build_train_loader(records, sampler, cfg, device)
    val_loader = build_val_loader(val_files, cfg, device)
    test_loader = build_val_loader(test_files, cfg, device)

    # ----- model -----
    if cfg.model_type == "dscformer":
        model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)
    else:
        model = SegFormerWithBoundary(cfg.pretrained).to(device)
    preview_model = _PreviewWrapper(model)
    criterion = CompositeLoss(ce_weight=w, cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    # Warmup + cosine scheduler
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup_sched = LinearLR(optimizer, start_factor=1e-2, total_iters=cfg.warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_epochs - cfg.warmup_epochs)
    lr_scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                                milestones=[cfg.warmup_epochs])

    # ----- OOM probe -----
    accum_notice = None
    if not args.dry_run and cfg.grad_accum == 1 and cfg.batch_size > 2:
        print("[train] running OOM probe ...")
        ok = oom_probe(model, cfg, device)
        if not ok:
            new_bs, new_ga = 2, max(1, cfg.batch_size // 2)
            accum_notice = (f"OOM at batch_size={cfg.batch_size}; falling back to"
                            f" batch_size={new_bs} grad_accum={new_ga}"
                            f" (effective batch = {new_bs * new_ga})")
            print(f"[train] {accum_notice}")
            cfg.batch_size = new_bs
            cfg.grad_accum = new_ga
            train_loader = build_train_loader(records, sampler, cfg, device)
            val_loader = build_val_loader(val_files, cfg, device)
            test_loader = build_val_loader(test_files, cfg, device)
            del model, optimizer, lr_scheduler, preview_model
            if device == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            if cfg.model_type == "dscformer":
                model = DSCformerDam(cfg.pretrained, cfg=cfg).to(device)
            else:
                model = SegFormerWithBoundary(cfg.pretrained).to(device)
            preview_model = _PreviewWrapper(model)
            criterion = CompositeLoss(ce_weight=w, cfg=cfg).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)
            warmup_sched = LinearLR(optimizer, start_factor=1e-2,
                                    total_iters=cfg.warmup_epochs)
            cosine_sched = CosineAnnealingLR(optimizer,
                                             T_max=total_epochs - cfg.warmup_epochs)
            lr_scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                                        milestones=[cfg.warmup_epochs])
        else:
            print("[train] OOM probe OK")

    # Mixed precision
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp:
        print("[train] mixed precision (AMP) enabled")

    train_metrics = SegMetricsBF1(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    eval_metrics = SegMetricsBF1(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)

    # ----- dry run -----
    if args.dry_run:
        print("[train] DRY RUN: 1 train step + 1 val batch")
        # Set sampler for epoch 1
        stage = curriculum_scheduler.stage(1)
        if cfg.use_competence_soft_mixing:
            weights = curriculum_scheduler.competence_tier_weights(1)
            sampler.set_epoch(1, total_epochs, cfg.warmup_epochs, stage,
                              competence_tier_weights=weights)
        elif cfg.use_competence_curriculum:
            max_tier = curriculum_scheduler.max_allowed_tier(1)
            sampler.set_epoch(1, total_epochs, cfg.warmup_epochs, stage,
                              competence_tier=max_tier)
        else:
            tier_mix = curriculum_scheduler.tier_mix(1) if cfg.use_soft_curriculum else None
            sampler.set_epoch(1, total_epochs, cfg.warmup_epochs, stage, tier_mix=tier_mix)

        batch = next(iter(train_loader))
        imgs = batch["image"].to(device).float()
        masks = batch["mask"].to(device).long()
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(imgs)

        print(f"  seg_logits shape: {outputs['seg_logits'].shape}")
        print(f"  boundary_logits shape: {outputs['boundary_logits'].shape}")
        print(f"  fuse_feat shape: {model._fuse_feat.shape}")

        with torch.amp.autocast("cuda", enabled=use_amp):
            total_loss, info = criterion(outputs, masks, curriculum_scheduler, 1)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        print(f"  train loss = {float(total_loss):.4f}")
        print(f"  loss_ce={info['loss_ce']:.4f} loss_dice={info['loss_dice']:.4f}"
              f" loss_tversky={info['loss_tversky']:.4f} loss_bd={info['loss_bd']:.4f}"
              f" loss_cldice={info['loss_cldice']:.4f} loss_snake={info['loss_snake']:.4f}")
        print(f"  per_sample_ce shape: {info['per_sample_ce'].shape}")

        # Check sampler warmup bypass
        print(f"  sampler._use_dynamic = {sampler._use_dynamic} (should be False during warmup)")
        ss = sampler.get_sampling_stats()
        if ss:
            print(f"  sampler stats: {ss['tier_hist']} has_spalling={ss['has_spalling_ratio']:.2%}")

        # Val batch
        model.eval()
        with torch.no_grad():
            v_batch = next(iter(val_loader))
            v_imgs = v_batch["image"].to(device).float()
            v_masks = v_batch["mask"].to(device).long()
            v_out = model(v_imgs)
            v_seg = F.interpolate(v_out["seg_logits"], v_masks.shape[-2:],
                                  mode="bilinear", align_corners=False)
            eval_metrics.reset()
            eval_metrics.update(v_seg, v_masks)
            print(format_metrics(eval_metrics.compute()))

        # Check difficulty update
        for sid in list(sample_bank.keys())[:3]:
            s = sample_bank[sid]
            print(f"  sample_bank[{sid[:30]}...]: ema_loss={s.ema_loss:.4f}"
                  f" boundary_complex={s.boundary_complexity:.4f}")

        print("[train] dry-run OK")
        return

    # ----- viz samples -----
    viz_files = pick_viz_samples(val_files, C.DATA_ROOT, seed=C.SEED)
    print(f"[train] viz samples ({len(viz_files)}):")
    for r in viz_files:
        print(f"    {r}")

    # ----- resume -----
    start_epoch = 1
    best_miou = -1.0
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = (rdir / resume_path).resolve() if not resume_path.exists() \
                else resume_path.resolve()
        print(f"[train] resuming from {resume_path}")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            try:
                lr_scheduler.load_state_dict(state["scheduler"])
            except Exception as e:
                print(f"[train] scheduler state ignored: {e}")
        if "sample_bank" in state:
            for sid, sdata in state["sample_bank"].items():
                if sid in sample_bank:
                    # Filter to valid SampleState fields for backward compat
                    valid_keys = {f.name for f in SampleState.__dataclass_fields__.values()}
                    filtered = {k: v for k, v in sdata.items() if k in valid_keys}
                    sample_bank[sid] = SampleState(**filtered)
            print(f"[train] restored sample_bank ({len(state['sample_bank'])} entries)")
        start_epoch = int(state.get("epoch", 0)) + 1
        best_miou = float(state.get("best_miou_fg", state.get("mIoU_fg", -1.0) or -1.0))
        print(f"[train] resumed at epoch={start_epoch} best_miou_fg={best_miou:.4f}")

    # ----- file outputs -----
    csv_path = rdir / "metrics.csv"
    if args.resume is None and csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"[ERROR] Run directory {rdir} already has metrics.csv with data. "
              f"Delete the directory manually or use --resume to continue.")
        _sys.exit(1)
    if args.resume is None and accum_notice is not None:
        with open(csv_path, "w") as f:
            f.write(f"# {accum_notice}\n")

    last_pt = rdir / "last.pt"
    best_pt = rdir / "best.pt"

    if args.resume is None:
        save_preview(preview_model, viz_files, C.DATA_ROOT,
                     samples_dir / "epoch_000_init.png", device, cfg.img_size)

    # ----- training loop -----
    run_t0 = time.time()
    epochs_done = 0
    for epoch in range(start_epoch, total_epochs + 1):
        # 1. Stage + sampler update
        stage = curriculum_scheduler.stage(epoch)

        if mac_pacer is not None:
            # MAC adaptive pacing: use pacer's tier weights
            weights = mac_pacer.tier_weights()
            sampler.set_epoch(epoch, total_epochs, cfg.warmup_epochs, stage,
                              competence_tier_weights=weights)
            print(f"[{cfg.name}] [MAC pacer] epoch {epoch}: stage={mac_pacer.stage} "
                  f"tier_weights={{0:{weights[0]:.3f}, 1:{weights[1]:.3f}, 2:{weights[2]:.3f}}}")
        elif cfg.use_competence_soft_mixing:
            comp = curriculum_scheduler.competence(epoch)
            weights = curriculum_scheduler.competence_tier_weights(epoch)
            sampler.set_epoch(epoch, total_epochs, cfg.warmup_epochs, stage,
                              competence_tier_weights=weights)
            print(f"[{cfg.name}] [curriculum] epoch {epoch}: c={comp:.3f} "
                  f"target_weights={{0:{weights[0]:.3f}, 1:{weights[1]:.3f}, 2:{weights[2]:.3f}}}")
        elif cfg.use_competence_curriculum:
            comp = curriculum_scheduler.competence(epoch)
            max_tier = curriculum_scheduler.max_allowed_tier(epoch)
            sampler.set_epoch(epoch, total_epochs, cfg.warmup_epochs, stage,
                              competence_tier=max_tier)
            print(f"[{cfg.name}] [curriculum] epoch {epoch}: c={comp:.3f} max_tier={max_tier}")
        elif cfg.use_soft_curriculum:
            tier_mix = curriculum_scheduler.tier_mix(epoch)
            sampler.set_epoch(epoch, total_epochs, cfg.warmup_epochs, stage, tier_mix=tier_mix)
            mix_str = " ".join(f"t{t}={r:.0%}" for t, r in tier_mix.items())
            print(f"[{cfg.name}] [curriculum] epoch {epoch}: stage={stage} tier_mix=[{mix_str}]")
        else:
            sampler.set_epoch(epoch, total_epochs, cfg.warmup_epochs, stage)
            allowed = sampler._allowed_tiers()
            pool_counts = {t: sum(1 for r in records if r["tier"] == t and t in allowed)
                           for t in range(3)}
            print(f"[{cfg.name}] [curriculum] epoch {epoch}: stage={stage} tiers={sorted(allowed)}"
                  f" pool={pool_counts}")

        # 2. Train (with MAC CE/topo multipliers if active)
        _mac_ce_mults = None
        _mac_topo_mult = 1.0
        if mac_class_sched is not None:
            _mac_ce_mults = mac_class_sched.get_ce_weight_multipliers()
            _mac_topo_mult = mac_class_sched.get_topo_multiplier()
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             train_metrics, cfg.grad_accum,
                             sample_bank, estimator,
                             curriculum_scheduler, epoch, total_epochs,
                             cfg=cfg, scaler=scaler,
                             mac_ce_multipliers=_mac_ce_mults,
                             mac_topo_multiplier=_mac_topo_mult)

        # 3. Print sampler stats
        ss = sampler.get_sampling_stats()
        if ss:
            print(f"[{cfg.name}] [sampler] sampled: {ss['tier_hist']} total={ss['total']}"
                  f" spalling_ratio={ss['has_spalling_ratio']:.2%}")
            if 'unique_hist' in ss:
                print(f"[{cfg.name}] [sampler] unique: {ss['unique_hist']} "
                      f"unique_total={ss['unique_total']} dup_ratio={ss['dup_ratio']:.2f}x")

        # 4. Epoch-level z-score normalize + rescore (before val)
        if cfg.use_dynamic_difficulty:
            estimator.normalize_and_score(sample_bank)

            # 5. Difficulty distribution stats
            diffs = [s.difficulty for s in sample_bank.values()]
            print(f"  difficulty: mean={np.mean(diffs):.3f} std={np.std(diffs):.3f}")

        # 6. Val (every 5 epochs, or every epoch in last 10, or epoch 1)
        #    MAC adaptive pacing: validate every epoch for responsive stage transitions
        do_val = (epoch % 5 == 0) or (epoch > total_epochs - 10) or (epoch == 1)
        if mac_pacer is not None:
            do_val = True  # per-epoch val for adaptive pacing
        if do_val:
            va = evaluate(model, val_loader, criterion, device, eval_metrics,
                          curriculum_scheduler, epoch, use_amp=use_amp)
            # MAC: update pacer and class loss scheduler after validation
            if mac_pacer is not None:
                old_stage = mac_pacer.stage
                new_stage = mac_pacer.update(va, epoch)
                if new_stage != old_stage:
                    # Re-set sampler with new tier weights from pacer
                    pacer_weights = mac_pacer.tier_weights()
                    sampler.set_epoch(epoch, total_epochs, cfg.warmup_epochs, stage,
                                      competence_tier_weights=pacer_weights)
                    print(f"[MAC] pacer stage {old_stage}→{new_stage}, "
                          f"new tier_weights={pacer_weights}")
            if mac_class_sched is not None:
                mac_class_sched.update(va)
                cm, sm = mac_class_sched.get_ce_weight_multipliers()
                tm = mac_class_sched.get_topo_multiplier()
                print(f"[MAC] class_loss: crack_mult={cm:.3f} spalling_mult={sm:.3f}"
                      f" topo_mult={tm:.3f}")
        lr_scheduler.step()
        dt = time.time() - t0
        epochs_done += 1
        avg_ep = (time.time() - run_t0) / epochs_done
        remain = total_epochs - epoch
        run_eta = avg_ep * remain
        print(f"[{cfg.name}] [epoch {epoch:03d}/{total_epochs}] remain={remain} epochs"
              f" ep_eta={run_eta/60:.1f}min (avg {avg_ep:.0f}s/ep)", flush=True)

        print(f"[{cfg.name}] [epoch {epoch:03d}/{total_epochs}] lr={optimizer.param_groups[0]['lr']:.6f}"
              f"  dt={dt:.1f}s")
        print(f"  train loss={tr['loss']:.4f}"
              f"  ce={tr.get('loss_ce', 0):.4f} dice={tr.get('loss_dice', 0):.4f}"
              f"  tversky={tr.get('loss_tversky', 0):.4f} bd={tr.get('loss_bd', 0):.4f}"
              f"  cldice={tr.get('loss_cldice', 0):.4f} snake={tr.get('loss_snake', 0):.4f}")
        if do_val:
            print("  val   loss={:.4f}".format(va['loss']))
            print("  " + format_metrics(va).replace("\n", "\n  "))
        else:
            print("  val   skipped (next val at epoch {})".format(
                epoch + (5 - epoch % 5)))

        if do_val:
            row = {
                "epoch": epoch, "split": "val",
                "train_loss": tr["loss"], "val_loss": va["loss"],
                "loss_ce": tr.get("loss_ce", ""),
                "loss_dice": tr.get("loss_dice", ""),
                "loss_tversky": tr.get("loss_tversky", ""),
                "loss_bd": tr.get("loss_bd", ""),
                "loss_cldice": tr.get("loss_cldice", ""),
                "loss_snake": tr.get("loss_snake", ""),
                "IoU_background": va["IoU_background"],
                "IoU_crack": va["IoU_crack"],
                "IoU_spalling": va["IoU_spalling"],
                "Dice_background": va["Dice_background"],
                "Dice_crack": va["Dice_crack"],
                "Dice_spalling": va["Dice_spalling"],
                "mIoU_fg": va["mIoU_fg"],
                "mIoU_all": va["mIoU_all"],
                "pixel_acc": va["pixel_acc"],
                "BF1_crack": va["BF1_crack"],
                "BF1_spalling": va["BF1_spalling"],
                "BF1_fg_mean": va["BF1_fg_mean"],
                "sampled_t0": ss.get("tier_hist", {}).get(0, "") if ss else "",
                "sampled_t1": ss.get("tier_hist", {}).get(1, "") if ss else "",
                "sampled_t2": ss.get("tier_hist", {}).get(2, "") if ss else "",
            }
            write_metrics_row(csv_path, row)

        # Save checkpoint (with sample_bank for resume)
        bank_serializable = {
            sid: {"ema_loss": s.ema_loss, "ema_uncertainty": s.ema_uncertainty,
                  "boundary_complexity": s.boundary_complexity, "sparsity": s.sparsity,
                  "difficulty": s.difficulty,
                  "ema_loss_crack": s.ema_loss_crack,
                  "ema_loss_spalling": s.ema_loss_spalling,
                  "morph_difficulty": s.morph_difficulty}
            for sid, s in sample_bank.items()
        }
        torch.save(
            {"model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scheduler": lr_scheduler.state_dict(),
             "epoch": epoch,
             "best_miou_fg": best_miou,
             "sample_bank": bank_serializable},
            last_pt,
        )
        if do_val and va["mIoU_fg"] > best_miou:
            best_miou = va["mIoU_fg"]
            torch.save(
                {"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": lr_scheduler.state_dict(),
                 "epoch": epoch,
                 "mIoU_fg": best_miou,
                 "best_miou_fg": best_miou,
                 "sample_bank": bank_serializable},
                best_pt,
            )
            print(f"  [{cfg.name}] [best] mIoU_fg={best_miou:.4f}  saved -> {best_pt.name}")

        if epoch % 5 == 0 or epoch == 1 or epoch == total_epochs:
            save_preview(preview_model, viz_files, C.DATA_ROOT,
                         samples_dir / f"epoch_{epoch:03d}.png", device, cfg.img_size)

    # ----- final test eval using best -----
    print("\n[train] final test eval using best checkpoint")
    if not best_pt.exists():
        print("[train] WARNING: best.pt missing; using current model")
    else:
        state = torch.load(best_pt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])

    # Use SegMetricsFull for final test (adds clDice + connectivity)
    try:
        from shared_eval.metrics_full import SegMetricsFull
        test_eval_metrics = SegMetricsFull(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    except ImportError:
        print("[train] WARNING: shared_eval not available; using SegMetricsBF1 for test")
        test_eval_metrics = eval_metrics

    test_m = evaluate(model, test_loader, criterion, device, test_eval_metrics,
                      curriculum_scheduler, total_epochs, use_amp=use_amp)
    print(format_metrics(test_m))

    report = rdir / "test_report.txt"
    with open(report, "w") as f:
        f.write(f"run: {cfg.name}\n")
        f.write(f"pretrained: {cfg.pretrained}  img_size: {cfg.img_size}  "
                f"batch: {cfg.batch_size}  grad_accum: {cfg.grad_accum}  "
                f"epochs: {total_epochs}  lr: {cfg.lr}  "
                f"warmup: {cfg.warmup_epochs}\n")
        f.write(f"difficulty: alpha={cfg.diff_alpha} beta={cfg.diff_beta}"
                f" gamma={cfg.diff_gamma} delta={cfg.diff_delta}"
                f" ema={cfg.diff_ema} tau={cfg.diff_tau}\n")
        f.write(f"bonuses: spalling={cfg.spalling_bonus}"
                f" late_hard_crack={cfg.late_hard_crack_bonus}\n")
        f.write(f"ablation: dynamic_difficulty={cfg.use_dynamic_difficulty}"
                f" class_sampling_bonus={cfg.use_class_sampling_bonus}"
                f" class_loss_schedule={cfg.use_class_loss_schedule}"
                f" boundary_loss={cfg.use_boundary_loss}"
                f" tversky_loss={cfg.use_tversky_loss}\n")
        f.write(f"new switches: soft_curriculum={cfg.use_soft_curriculum}"
                f" softmax_sampling={cfg.use_softmax_sampling}"
                f" dynamic_loss_reweight={cfg.use_dynamic_loss_reweight}"
                f" (lambda={cfg.loss_reweight_lambda})"
                f" soft_boundary={cfg.use_soft_boundary_schedule}\n")
        f.write(f"cldice: use={cfg.use_cldice_loss}"
                f" weight={cfg.cldice_weight} start_epoch={cfg.cldice_start_epoch}"
                f" iters={cfg.cldice_iters}\n")
        f.write(f"srl: use={cfg.use_srl_loss}\n")
        f.write(f"loss: ce_w={cfg.loss_ce_w} dice_w={cfg.loss_dice_w}"
                f" tversky_alpha={cfg.loss_tversky_alpha}"
                f" tversky_beta={cfg.loss_tversky_beta}\n")
        if accum_notice:
            f.write(f"note: {accum_notice}\n")
        if best_pt.exists():
            f.write(f"best epoch: {state.get('epoch')}\n")
            f.write(f"best val mIoU_fg: {state.get('mIoU_fg')}\n")
        f.write("test set metrics:\n")
        for k in ("IoU_background", "IoU_crack", "IoU_spalling",
                  "Dice_background", "Dice_crack", "Dice_spalling",
                  "mIoU_fg", "mIoU_all", "pixel_acc",
                  "BF1_crack", "BF1_spalling", "BF1_fg_mean",
                  "clDice_crack", "clDice_spalling", "clDice_fg_mean",
                  "ConnR_crack", "ConnR_spalling", "ConnR_fg_mean"):
            f.write(f"  {k}: {test_m.get(k)}\n")
        f.write("confusion matrix (rows=gt, cols=pred):\n")
        f.write(str(test_eval_metrics.cm) + "\n")
    print(f"[train] wrote {report}")

    try:
        render_curves(csv_path, rdir / "curves.png")
        print(f"[train] wrote {rdir / 'curves.png'}")
    except Exception as e:
        print(f"[train] WARNING: curves render failed: {e}")


if __name__ == "__main__":
    main()

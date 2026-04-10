"""Step-4 SegFormer-B2 training.

Two presets selectable with --preset {A,B}:
    A = segformer_b2_plain_512      (plain baseline, 100 ep, lr=6e-5)
    B = segformer_b2_staticcurr_512 (static curriculum, 100 ep, lr=6e-5)

Reuses baseline_unet's dataset, losses, splits and visualize modules verbatim.

Usage:
    python -m baseline_segformer.train --preset A --dry-run
    python -m baseline_segformer.train --preset A --epochs 10       # probe
    python -m baseline_segformer.train --preset A
    python -m baseline_segformer.train --preset B
    python -m baseline_segformer.train --preset A --resume runs/segformer_b2_plain_512/last.pt
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

from baseline_unet.dataset import (
    DamSegmentDataset,
    build_transforms,
    compute_class_weights,
    read_split_file,
)
from baseline_unet.losses import CEDiceLoss
from baseline_unet.visualize import pick_viz_samples, save_preview

from baseline_deeplab.metrics import SegMetricsBF1, format_metrics

from baseline_segformer import config as C


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
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: C.RunCfg) -> nn.Module:
    from transformers import SegformerForSemanticSegmentation
    return SegformerForSemanticSegmentation.from_pretrained(
        cfg.pretrained,
        num_labels=C.NUM_CLASSES,
        ignore_mismatched_sizes=True,
        id2label={0: "background", 1: "crack", 2: "spalling"},
        label2id={"background": 0, "crack": 1, "spalling": 2},
    )


class _PreviewWrapper(nn.Module):
    """Wraps HF SegFormer so save_preview() gets a raw logits tensor."""
    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.m = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.m(x).logits
        return F.interpolate(logits, size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Data loaders + curriculum helpers
# ---------------------------------------------------------------------------

def get_curriculum_tiers(epoch: int, total_epochs: int) -> List[str]:
    s1 = round(0.3 * total_epochs)   # 30 % Easy-only
    s2 = round(0.6 * total_epochs)   # 30 % Easy+Medium
    if epoch <= s1:
        return ["Easy"]
    elif epoch <= s2:
        return ["Easy", "Medium"]
    else:
        return ["Easy", "Medium", "Hard"]


def filter_by_tier(files: List[str], tiers: List[str]) -> List[str]:
    return [f for f in files if any(f.startswith(t + "/") for t in tiers)]


def build_loader(files: List[str], cfg: C.RunCfg, device: str,
                 train: bool) -> DataLoader:
    ds = DamSegmentDataset(C.DATA_ROOT, files,
                           build_transforms(cfg.img_size, train=train))
    pin = (device == "cuda")
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train,
                      num_workers=2, pin_memory=pin, drop_last=False)


def build_loaders(cfg: C.RunCfg, device: str, train_files: List[str],
                  val_files: List[str], test_files: List[str]):
    train_loader = build_loader(train_files, cfg, device, train=True)
    val_loader = build_loader(val_files, cfg, device, train=False)
    test_loader = build_loader(test_files, cfg, device, train=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# OOM probe
# ---------------------------------------------------------------------------

def oom_probe(model: nn.Module, cfg: C.RunCfg, device: str) -> bool:
    try:
        x = torch.zeros(cfg.batch_size, 3, cfg.img_size, cfg.img_size, device=device)
        y = torch.zeros(cfg.batch_size, cfg.img_size, cfg.img_size,
                        device=device, dtype=torch.long)
        logits = model(x).logits
        logits = F.interpolate(logits, size=y.shape[-2:],
                               mode="bilinear", align_corners=False)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        del x, y, logits, loss
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

def train_one_epoch(model, loader, optimizer, criterion, device,
                    metrics: SegMetricsBF1, grad_accum: int,
                    epoch: int = 0, total_epochs: int = 0) -> Dict[str, float]:
    model.train()
    metrics.reset()
    loss_sum = 0.0
    n_batches = 0
    class_pixel_sum = np.zeros(C.NUM_CLASSES, dtype=np.int64)
    total_steps = len(loader)
    log_every = max(1, total_steps // 10)
    t_start = time.time()

    optimizer.zero_grad(set_to_none=True)
    for step, (imgs, masks, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).long()
        logits = model(imgs).logits
        logits = F.interpolate(logits, size=masks.shape[-2:],
                               mode="bilinear", align_corners=False)
        loss = criterion(logits, masks)
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_sum += float(loss.detach().cpu())
        n_batches += 1
        metrics.update(logits, masks)
        m_np = masks.detach().cpu().numpy()
        for c in range(C.NUM_CLASSES):
            class_pixel_sum[c] += int((m_np == c).sum())
        done = step + 1
        if done % log_every == 0 or done == total_steps:
            elapsed = time.time() - t_start
            eta = elapsed / done * (total_steps - done)
            print(f"  [epoch {epoch}/{total_epochs}] batch {done}/{total_steps}"
                  f" ({done*100//total_steps}%) loss={loss_sum/n_batches:.4f}"
                  f" elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    # Flush any tail micro-batches.
    if n_batches % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = loss_sum / max(n_batches, 1)
    m = metrics.compute()
    m["loss"] = avg_loss
    total = class_pixel_sum.sum()
    frac = class_pixel_sum / max(total, 1)
    print(f"  [train] label px frac: bg={frac[0]:.4f} crack={frac[1]:.4f} spalling={frac[2]:.4f}")
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, device,
             metrics: SegMetricsBF1) -> Dict[str, float]:
    model.eval()
    metrics.reset()
    loss_sum = 0.0
    n_batches = 0
    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).long()
        logits = model(imgs).logits
        logits = F.interpolate(logits, size=masks.shape[-2:],
                               mode="bilinear", align_corners=False)
        loss = criterion(logits, masks)
        loss_sum += float(loss.detach().cpu())
        n_batches += 1
        metrics.update(logits, masks)
    m = metrics.compute()
    m["loss"] = loss_sum / max(n_batches, 1)
    return m


METRIC_KEYS: List[str] = [
    "epoch", "split", "train_loss", "val_loss",
    "IoU_background", "IoU_crack", "IoU_spalling",
    "Dice_background", "Dice_crack", "Dice_spalling",
    "mIoU_fg", "mIoU_all", "pixel_acc",
    "BF1_crack", "BF1_spalling", "BF1_fg_mean",
]


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

    epochs: List[int] = []
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
            epochs.append(ep)
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
    axs[0].plot(epochs, train_loss, label="train"); axs[0].plot(epochs, val_loss, label="val")
    axs[0].set_title("loss"); axs[0].set_xlabel("epoch"); axs[0].legend()
    axs[1].plot(epochs, iou_crack, label="IoU_crack")
    axs[1].plot(epochs, iou_spalling, label="IoU_spalling")
    axs[1].plot(epochs, miou_fg, label="mIoU_fg")
    axs[1].set_title("IoU / mIoU_fg (val)"); axs[1].set_xlabel("epoch"); axs[1].legend()
    axs[2].plot(epochs, bf1_fg, label="BF1_fg_mean")
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
    parser.add_argument("--preset", choices=["A", "B"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-manual-weights", action="store_true")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override preset epochs (e.g. 10 for probe)")
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint path to resume from")
    parser.add_argument("--train-split", type=str, default=None,
                        help="custom train split file (e.g. splits/train_20.txt)")
    args = parser.parse_args()

    cfg = C.PRESETS[args.preset]
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
    print(f"[train] preset={args.preset} ({cfg.name})")
    print(f"[train] device={device}  torch={_t.__version__}  transformers={tf_v}")
    print(f"[train] cfg: pretrained={cfg.pretrained} img={cfg.img_size} bs={cfg.batch_size}"
          f" grad_accum={cfg.grad_accum} epochs={total_epochs} lr={cfg.lr}"
          f" warmup={cfg.warmup_epochs} curriculum={cfg.curriculum}")

    rdir = C.run_dir(cfg)
    samples_dir = rdir / "samples"

    # ----- read split files -----
    all_train_files = (read_split_file(Path(args.train_split))
                        if args.train_split else read_split_file(C.SPLIT_FILES["train"]))
    val_files = read_split_file(C.SPLIT_FILES["val"])
    test_files = read_split_file(C.SPLIT_FILES["test"])
    print(f"[train] sizes: train={len(all_train_files)} val={len(val_files)} test={len(test_files)}")

    # ----- class weights from ALL train files (shared across stages) -----
    raw_freq, smoothed, clipped = compute_class_weights(C.DATA_ROOT, all_train_files)
    print("[train] raw pixel frequency:", dict(zip(C.CLASS_NAMES, [float(x) for x in raw_freq])))
    print("[train] clipped weights    :", dict(zip(C.CLASS_NAMES, [float(x) for x in clipped])))
    if args.use_manual_weights:
        w = torch.tensor(C.CE_WEIGHTS, dtype=torch.float32)
        print("[train] using MANUAL CE weights")
    else:
        w = torch.tensor(clipped, dtype=torch.float32)
        print("[train] using AUTO CE weights")

    # ----- initial loaders -----
    if cfg.curriculum == "static":
        tiers = get_curriculum_tiers(1, total_epochs)
        cur_train_files = filter_by_tier(all_train_files, tiers)
        print(f"[train] curriculum tiers={tiers}  train_subset={len(cur_train_files)}")
    else:
        cur_train_files = all_train_files

    train_loader, val_loader, test_loader = build_loaders(
        cfg, device, cur_train_files, val_files, test_files)

    # ----- model -----
    model = build_model(cfg).to(device)
    preview_model = _PreviewWrapper(model)
    criterion = CEDiceLoss(ce_weight=w).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    # Warmup + cosine scheduler
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - cfg.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
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
            train_loader, val_loader, test_loader = build_loaders(
                cfg, device, cur_train_files, val_files, test_files)
            del model, optimizer, scheduler, preview_model
            if device == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            model = build_model(cfg).to(device)
            preview_model = _PreviewWrapper(model)
            criterion = CEDiceLoss(ce_weight=w).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)
            warmup = LinearLR(optimizer, start_factor=1e-2,
                              total_iters=cfg.warmup_epochs)
            cosine = CosineAnnealingLR(optimizer,
                                       T_max=total_epochs - cfg.warmup_epochs)
            scheduler = SequentialLR(optimizer, [warmup, cosine],
                                     milestones=[cfg.warmup_epochs])
        else:
            print("[train] OOM probe OK")

    train_metrics = SegMetricsBF1(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)
    eval_metrics = SegMetricsBF1(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)

    # ----- dry run -----
    if args.dry_run:
        print("[train] DRY RUN: 1 train step + 1 val batch (with BF1)")
        imgs, masks, _ = next(iter(train_loader))
        imgs = imgs.to(device).float(); masks = masks.to(device).long()
        logits = model(imgs).logits
        logits = F.interpolate(logits, size=masks.shape[-2:],
                               mode="bilinear", align_corners=False)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        print(f"[train] dry-run train loss = {float(loss):.4f}")

        model.eval()
        with torch.no_grad():
            v_imgs, v_masks, _ = next(iter(val_loader))
            v_imgs = v_imgs.to(device).float(); v_masks = v_masks.to(device).long()
            v_logits = model(v_imgs).logits
            v_logits = F.interpolate(v_logits, size=v_masks.shape[-2:],
                                     mode="bilinear", align_corners=False)
            eval_metrics.reset()
            eval_metrics.update(v_logits, v_masks)
            print(format_metrics(eval_metrics.compute()))
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
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception as e:
                print(f"[train] scheduler state ignored: {e}")
        start_epoch = int(state.get("epoch", 0)) + 1
        best_miou = float(state.get("best_miou_fg", state.get("mIoU_fg", -1.0) or -1.0))
        print(f"[train] resumed at epoch={start_epoch} best_miou_fg={best_miou:.4f}")

    # ----- file outputs -----
    csv_path = rdir / "metrics.csv"
    if args.resume is None and csv_path.exists():
        csv_path.unlink()
    if args.resume is None and accum_notice is not None:
        with open(csv_path, "w") as f:
            f.write(f"# {accum_notice}\n")

    last_pt = rdir / "last.pt"
    best_pt = rdir / "best.pt"

    if args.resume is None:
        save_preview(preview_model, viz_files, C.DATA_ROOT,
                     samples_dir / "epoch_000_init.png", device, cfg.img_size)

    # ----- track current curriculum tiers -----
    prev_tiers: List[str] | None = None

    # ----- training loop -----
    run_t0 = time.time()
    epochs_done = 0
    for epoch in range(start_epoch, total_epochs + 1):
        # Curriculum: rebuild train loader when tiers change
        if cfg.curriculum == "static":
            tiers = get_curriculum_tiers(epoch, total_epochs)
            if tiers != prev_tiers:
                cur_train_files = filter_by_tier(all_train_files, tiers)
                train_loader = build_loader(cur_train_files, cfg, device, train=True)
                prev_tiers = tiers
                print(f"[curriculum] epoch {epoch}: tiers={tiers}"
                      f"  train_subset={len(cur_train_files)}")

        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             train_metrics, cfg.grad_accum,
                             epoch=epoch, total_epochs=total_epochs)
        va = evaluate(model, val_loader, criterion, device, eval_metrics)
        scheduler.step()
        dt = time.time() - t0
        epochs_done += 1
        avg_ep = (time.time() - run_t0) / epochs_done
        remain = total_epochs - epoch
        run_eta = avg_ep * remain
        print(f"[epoch {epoch:03d}/{total_epochs}] remain={remain} epochs"
              f" ep_eta={run_eta/60:.1f}min (avg {avg_ep:.0f}s/ep)", flush=True)

        print(f"[epoch {epoch:03d}/{total_epochs}] lr={optimizer.param_groups[0]['lr']:.6f}"
              f"  dt={dt:.1f}s")
        print(f"  train loss={tr['loss']:.4f}")
        print("  val   loss={:.4f}".format(va['loss']))
        print("  " + format_metrics(va).replace("\n", "\n  "))

        row = {
            "epoch": epoch, "split": "val",
            "train_loss": tr["loss"], "val_loss": va["loss"],
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
        }
        write_metrics_row(csv_path, row)

        torch.save(
            {"model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(),
             "epoch": epoch,
             "best_miou_fg": best_miou},
            last_pt,
        )
        if va["mIoU_fg"] > best_miou:
            best_miou = va["mIoU_fg"]
            torch.save(
                {"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "epoch": epoch,
                 "mIoU_fg": best_miou,
                 "best_miou_fg": best_miou},
                best_pt,
            )
            print(f"  [best] mIoU_fg={best_miou:.4f}  saved -> {best_pt.name}")

        if epoch % 5 == 0 or epoch == 1 or epoch == total_epochs:
            save_preview(preview_model, viz_files, C.DATA_ROOT,
                         samples_dir / f"epoch_{epoch:03d}.png", device, cfg.img_size)

    # ----- final test eval using best -----
    print("\n[train] final test eval using best checkpoint")
    if not best_pt.exists():
        print("[train] WARNING: best.pt missing (no completed epoch?); using current model")
    else:
        state = torch.load(best_pt, map_location=device)
        model.load_state_dict(state["model"])
    test_m = evaluate(model, test_loader, criterion, device, eval_metrics)
    print(format_metrics(test_m))

    report = rdir / "test_report.txt"
    with open(report, "w") as f:
        f.write(f"run: {cfg.name}\n")
        f.write(f"preset: {args.preset}\n")
        f.write(f"pretrained: {cfg.pretrained}  img_size: {cfg.img_size}  "
                f"batch: {cfg.batch_size}  grad_accum: {cfg.grad_accum}  "
                f"epochs: {total_epochs}  lr: {cfg.lr}  "
                f"warmup: {cfg.warmup_epochs}  curriculum: {cfg.curriculum}\n")
        if accum_notice:
            f.write(f"note: {accum_notice}\n")
        if best_pt.exists():
            f.write(f"best epoch: {state.get('epoch')}\n")
            f.write(f"best val mIoU_fg: {state.get('mIoU_fg')}\n")
        f.write("test set metrics:\n")
        for k in ("IoU_background", "IoU_crack", "IoU_spalling",
                  "Dice_background", "Dice_crack", "Dice_spalling",
                  "mIoU_fg", "mIoU_all", "pixel_acc",
                  "BF1_crack", "BF1_spalling", "BF1_fg_mean"):
            f.write(f"  {k}: {test_m.get(k)}\n")
        f.write("confusion matrix (rows=gt, cols=pred):\n")
        f.write(str(eval_metrics.cm) + "\n")
    print(f"[train] wrote {report}")

    try:
        render_curves(csv_path, rdir / "curves.png")
        print(f"[train] wrote {rdir / 'curves.png'}")
    except Exception as e:
        print(f"[train] WARNING: curves render failed: {e}")


if __name__ == "__main__":
    main()

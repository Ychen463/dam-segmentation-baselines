"""Step-0 U-Net training script.

Usage:
    python -m baseline_unet.train            # full 30-epoch training
    python -m baseline_unet.train --dry-run  # 1 batch forward+backward sanity
    python -m baseline_unet.train --use-manual-weights
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Workaround: cuDNN may fail to initialize on some CUDA 12.x / driver combos.
torch.backends.cudnn.enabled = False

from . import config as C
from .dataset import (
    DamSegmentDataset,
    build_transforms,
    compute_class_weights,
    read_split_file,
)
from .losses import CEDiceLoss
from .metrics import SegMetrics, format_metrics
from .splits import SPLIT_FILES, ensure_splits
from .visualize import pick_viz_samples, save_preview


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(pref: str) -> str:
    # CUDA always wins if available (RunPod / any GPU box), regardless of pref.
    if torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(num_classes: int) -> nn.Module:
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )


def train_one_epoch(model, loader, optimizer, criterion, device, metrics: SegMetrics,
                    epoch: int = 0, total_epochs: int = 0) -> Dict[str, float]:
    model.train()
    metrics.reset()
    loss_sum = 0.0
    n_batches = 0
    class_pixel_sum = np.zeros(C.NUM_CLASSES, dtype=np.int64)
    total_steps = len(loader)
    log_every = max(1, total_steps // 10)
    t_start = time.time()
    for step, (imgs, masks, _) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).long()
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach().cpu())
        n_batches += 1
        metrics.update(logits, masks)
        m_np = masks.detach().cpu().numpy()
        for c in range(C.NUM_CLASSES):
            class_pixel_sum[c] += int((m_np == c).sum())
        if step % log_every == 0 or step == total_steps:
            elapsed = time.time() - t_start
            eta = elapsed / step * (total_steps - step)
            print(f"  [epoch {epoch}/{total_epochs}] batch {step}/{total_steps}"
                  f" ({step*100//total_steps}%) loss={loss_sum/n_batches:.4f}"
                  f" elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    avg_loss = loss_sum / max(n_batches, 1)
    m = metrics.compute()
    m["loss"] = avg_loss
    total = class_pixel_sum.sum()
    frac = class_pixel_sum / max(total, 1)
    print(
        f"  [train] label px frac: bg={frac[0]:.4f} crack={frac[1]:.4f} spalling={frac[2]:.4f}"
    )
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, device, metrics: SegMetrics) -> Dict[str, float]:
    model.eval()
    metrics.reset()
    loss_sum = 0.0
    n_batches = 0
    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).long()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss_sum += float(loss.detach().cpu())
        n_batches += 1
        metrics.update(logits, masks)
    m = metrics.compute()
    m["loss"] = loss_sum / max(n_batches, 1)
    return m


METRIC_KEYS: List[str] = [
    "epoch",
    "train_loss",
    "val_loss",
    "IoU_background",
    "IoU_crack",
    "IoU_spalling",
    "Dice_background",
    "Dice_crack",
    "Dice_spalling",
    "mIoU_fg",
    "mIoU_all",
    "pixel_acc",
]


def write_metrics_row(csv_path: Path, row: Dict[str, float]) -> None:
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_KEYS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in METRIC_KEYS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="1 batch forward+backward then exit")
    parser.add_argument("--use-manual-weights", action="store_true",
                        help="use config.CE_WEIGHTS instead of auto-computed class weights")
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--train-split", type=str, default=None,
                        help="custom train split file (e.g. splits/train_20.txt)")
    args = parser.parse_args()

    set_seed(C.SEED)
    device = pick_device(C.DEVICE)
    print(f"[train] device = {device}")

    splits = ensure_splits()
    train_files = (read_split_file(Path(args.train_split))
                   if args.train_split else splits["train"])
    val_files = splits["val"]
    test_files = splits["test"]
    print(f"[train] sizes: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    # ----- class weights -----
    raw_freq, smoothed, clipped = compute_class_weights(C.DATA_ROOT, train_files)
    print("[train] raw pixel frequency:", dict(zip(C.CLASS_NAMES, [float(x) for x in raw_freq])))
    print("[train] smoothed weights   :", dict(zip(C.CLASS_NAMES, [float(x) for x in smoothed])))
    print("[train] clipped weights    :", dict(zip(C.CLASS_NAMES, [float(x) for x in clipped])))
    print("[train] manual fallback    :", dict(zip(C.CLASS_NAMES, C.CE_WEIGHTS)))

    if args.use_manual_weights:
        w = torch.tensor(C.CE_WEIGHTS, dtype=torch.float32)
        print("[train] using MANUAL CE weights")
    else:
        w = torch.tensor(clipped, dtype=torch.float32)
        print("[train] using AUTO CE weights")

    # ----- data -----
    train_ds = DamSegmentDataset(C.DATA_ROOT, train_files, build_transforms(C.IMG_SIZE, train=True))
    val_ds = DamSegmentDataset(C.DATA_ROOT, val_files, build_transforms(C.IMG_SIZE, train=False))
    test_ds = DamSegmentDataset(C.DATA_ROOT, test_files, build_transforms(C.IMG_SIZE, train=False))

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=pin)

    # ----- model / loss / optim -----
    model = build_model(C.NUM_CLASSES).to(device)
    criterion = CEDiceLoss(ce_weight=w).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metrics = SegMetrics(C.NUM_CLASSES)

    # ----- dry run -----
    if args.dry_run:
        print("[train] DRY RUN: 1 batch forward + backward")
        imgs, masks, _ = next(iter(train_loader))
        imgs = imgs.to(device).float(); masks = masks.to(device).long()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        metrics.reset(); metrics.update(logits, masks)
        print(f"[train] dry-run loss = {float(loss):.4f}")
        print(format_metrics(metrics.compute()))
        print("[train] dry-run OK")
        return

    # ----- visualisation sample set -----
    viz_files = pick_viz_samples(val_files, C.DATA_ROOT, seed=C.SEED)
    print(f"[train] viz samples ({len(viz_files)}):")
    for r in viz_files:
        print(f"    {r}")
    save_preview(model, viz_files, C.DATA_ROOT,
                 C.SAMPLES_DIR / "epoch_000_init.png", device, C.IMG_SIZE)

    # ----- training loop -----
    csv_path = C.RUN_DIR / "metrics.csv"
    if csv_path.exists():
        csv_path.unlink()
    best_miou = -1.0
    last_pt = C.RUN_DIR / "last.pt"
    best_pt = C.RUN_DIR / "best.pt"

    run_t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, metrics,
                             epoch=epoch, total_epochs=args.epochs)
        va = evaluate(model, val_loader, criterion, device, metrics)
        scheduler.step()
        dt = time.time() - t0
        avg_ep = (time.time() - run_t0) / epoch
        remain = args.epochs - epoch
        run_eta = avg_ep * remain
        print(f"[epoch {epoch:03d}/{args.epochs}] remain={remain} epochs"
              f" ep_eta={run_eta/60:.1f}min (avg {avg_ep:.0f}s/ep)", flush=True)

        print(f"[epoch {epoch:03d}/{args.epochs}] lr={optimizer.param_groups[0]['lr']:.6f}  dt={dt:.1f}s")
        print(f"  train loss={tr['loss']:.4f}  " + format_metrics(tr).replace("\n", "\n  "))
        print(f"  val   loss={va['loss']:.4f}  " + format_metrics(va).replace("\n", "\n  "))

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "IoU_background": va["IoU_background"],
            "IoU_crack": va["IoU_crack"],
            "IoU_spalling": va["IoU_spalling"],
            "Dice_background": va["Dice_background"],
            "Dice_crack": va["Dice_crack"],
            "Dice_spalling": va["Dice_spalling"],
            "mIoU_fg": va["mIoU_fg"],
            "mIoU_all": va["mIoU_all"],
            "pixel_acc": va["pixel_acc"],
        }
        write_metrics_row(csv_path, row)

        torch.save({"model": model.state_dict(), "epoch": epoch}, last_pt)
        if va["mIoU_fg"] > best_miou:
            best_miou = va["mIoU_fg"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "mIoU_fg": best_miou}, best_pt)
            print(f"  [best] mIoU_fg={best_miou:.4f}  saved -> {best_pt.name}")

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            save_preview(model, viz_files, C.DATA_ROOT,
                         C.SAMPLES_DIR / f"epoch_{epoch:03d}.png", device, C.IMG_SIZE)

    # ----- final test eval (loads best) -----
    print("\n[train] final test eval using best checkpoint")
    state = torch.load(best_pt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_m = evaluate(model, test_loader, criterion, device, metrics)
    print(format_metrics(test_m))
    report = C.RUN_DIR / "test_report.txt"
    with open(report, "w") as f:
        f.write(f"best epoch: {state.get('epoch')}\n")
        f.write(f"best val mIoU_fg: {state.get('mIoU_fg')}\n")
        f.write("test set metrics:\n")
        for k in METRIC_KEYS[3:]:
            f.write(f"  {k}: {test_m.get(k)}\n")
        f.write("confusion matrix (rows=gt, cols=pred):\n")
        f.write(str(metrics.cm) + "\n")
    print(f"[train] wrote {report}")


if __name__ == "__main__":
    main()

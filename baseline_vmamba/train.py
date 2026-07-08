"""VMamba baseline training.

Usage:
    python -m baseline_vmamba.train --preset T --dry-run
    python -m baseline_vmamba.train --preset T
    python -m baseline_vmamba.train --preset T --pretrained-ckpt path/to/vmamba_tiny.pth
    python -m baseline_vmamba.train --preset T --seed 42 --name vmamba_tiny_s42
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False

from baseline_unet.dataset import (
    DamSegmentDataset, build_transforms, compute_class_weights, read_split_file,
)
from baseline_unet.losses import CEDiceLoss
from baseline_unet.metrics import SegMetrics
from baseline_unet.visualize import pick_viz_samples, save_preview
from baseline_deeplab.metrics import SegMetricsBF1, format_metrics
from baseline_deeplab.train import (
    train_one_epoch, evaluate, write_metrics_row, render_curves, METRIC_KEYS,
)

from baseline_vmamba import config as C
from baseline_vmamba.model import build_vmamba_seg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_loaders(cfg: C.RunCfg, device: str):
    train_files = read_split_file(C.SPLIT_FILES["train"])
    val_files = read_split_file(C.SPLIT_FILES["val"])
    test_files = read_split_file(C.SPLIT_FILES["test"])

    train_ds = DamSegmentDataset(C.DATA_ROOT, train_files,
                                 build_transforms(cfg.img_size, train=True))
    val_ds = DamSegmentDataset(C.DATA_ROOT, val_files,
                               build_transforms(cfg.img_size, train=False))
    test_ds = DamSegmentDataset(C.DATA_ROOT, test_files,
                                build_transforms(cfg.img_size, train=False))

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=2, pin_memory=pin)
    return train_files, val_files, test_files, train_loader, val_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="VMamba baseline training")
    parser.add_argument("--preset", choices=["T", "S"], default="T",
                        help="T=VMamba-Tiny, S=VMamba-Small")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained-ckpt", type=str, default=None,
                        help="Path to pretrained VMamba classification checkpoint")
    args = parser.parse_args()

    cfg = C.PRESETS[args.preset]
    if args.name:
        cfg.name = args.name
    if args.grad_accum:
        cfg.grad_accum = args.grad_accum
    if args.pretrained_ckpt:
        cfg.pretrained_ckpt = args.pretrained_ckpt
    total_epochs = args.epochs if args.epochs is not None else cfg.epochs
    seed = args.seed if args.seed is not None else C.SEED

    set_seed(seed)
    device = pick_device()
    print(f"[train] VMamba preset={args.preset} ({cfg.name}) seed={seed}")
    print(f"[train] device={device}  torch={torch.__version__}")
    print(f"[train] dims={cfg.dims} depths={cfg.depths} d_state={cfg.d_state}")
    print(f"[train] img={cfg.img_size} bs={cfg.batch_size} "
          f"grad_accum={cfg.grad_accum} epochs={total_epochs} lr={cfg.lr}")

    rdir = C.run_dir(cfg)
    samples_dir = rdir / "samples"

    train_files, val_files, test_files, train_loader, val_loader, test_loader = \
        build_loaders(cfg, device)
    print(f"[train] sizes: train={len(train_files)} val={len(val_files)} "
          f"test={len(test_files)}")

    # Class weights
    raw_freq, smoothed, clipped = compute_class_weights(C.DATA_ROOT, train_files)
    print("[train] clipped weights:", dict(zip(C.CLASS_NAMES, [float(x) for x in clipped])))
    w = torch.tensor(clipped, dtype=torch.float32)

    # Model
    model = build_vmamba_seg(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[train] params: {n_params:.1f}M total, {n_train:.1f}M trainable")

    criterion = CEDiceLoss(ce_weight=w).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    # Warmup + cosine schedule
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(1, total_epochs - cfg.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_metrics = SegMetrics(C.NUM_CLASSES)
    eval_metrics = SegMetricsBF1(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)

    # Dry run
    if args.dry_run:
        print("[train] DRY RUN: 1 forward + backward")
        model.train()
        imgs, masks, _ = next(iter(train_loader))
        imgs = imgs.to(device).float()
        masks = masks.to(device).long()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        print(f"[train] dry-run loss={float(loss):.4f} "
              f"logits shape={logits.shape}")
        print("[train] dry-run OK")
        return

    # Viz samples
    viz_files = pick_viz_samples(val_files, C.DATA_ROOT, seed=seed)

    # Resume
    start_epoch = 1
    best_miou = -1.0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            resume_path = rdir / resume_path
        print(f"[train] resuming from {resume_path}")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception:
                pass
        start_epoch = int(state.get("epoch", 0)) + 1
        best_miou = float(state.get("best_miou_fg", -1.0))
        print(f"[train] resumed at epoch={start_epoch} best_miou_fg={best_miou:.4f}")

    csv_path = rdir / "metrics.csv"
    if args.resume is None and csv_path.exists():
        csv_path.unlink()
    last_pt = rdir / "last.pt"
    best_pt = rdir / "best.pt"

    if args.resume is None:
        save_preview(model, viz_files, C.DATA_ROOT,
                     samples_dir / "epoch_000_init.png", device, cfg.img_size)

    # Training loop
    run_t0 = time.time()
    epochs_done = 0
    for epoch in range(start_epoch, total_epochs + 1):
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
        print(f"[epoch {epoch:03d}/{total_epochs}] lr={optimizer.param_groups[0]['lr']:.6f}"
              f"  dt={dt:.1f}s  eta={avg_ep * remain / 60:.1f}min")
        print(f"  train loss={tr['loss']:.4f}")
        print(f"  val   loss={va['loss']:.4f}")
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
            save_preview(model, viz_files, C.DATA_ROOT,
                         samples_dir / f"epoch_{epoch:03d}.png", device, cfg.img_size)

    # Final test eval
    print("\n[train] final test eval using best checkpoint")
    if best_pt.exists():
        state = torch.load(best_pt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    test_m = evaluate(model, test_loader, criterion, device, eval_metrics)
    print(format_metrics(test_m))

    report = rdir / "test_report.txt"
    with open(report, "w") as f:
        f.write(f"run: {cfg.name}\n")
        f.write(f"preset: {args.preset}\n")
        f.write(f"encoder: VMamba dims={cfg.dims} depths={cfg.depths}\n")
        f.write(f"img_size: {cfg.img_size}  batch: {cfg.batch_size}  "
                f"grad_accum: {cfg.grad_accum}  epochs: {total_epochs}  lr: {cfg.lr}\n")
        f.write(f"seed: {seed}\n")
        if best_pt.exists():
            f.write(f"best epoch: {state.get('epoch')}\n")
            f.write(f"best val mIoU_fg: {state.get('mIoU_fg')}\n")
        f.write("test set metrics:\n")
        for k in ("IoU_background", "IoU_crack", "IoU_spalling",
                  "Dice_background", "Dice_crack", "Dice_spalling",
                  "mIoU_fg", "mIoU_all", "pixel_acc",
                  "BF1_crack", "BF1_spalling", "BF1_fg_mean"):
            f.write(f"  {k}: {test_m.get(k)}\n")
        f.write(f"confusion matrix (rows=gt, cols=pred):\n")
        f.write(str(eval_metrics.cm) + "\n")
    print(f"[train] wrote {report}")

    try:
        render_curves(csv_path, rdir / "curves.png")
        print(f"[train] wrote {rdir / 'curves.png'}")
    except Exception as e:
        print(f"[train] WARNING: curves render failed: {e}")


if __name__ == "__main__":
    main()

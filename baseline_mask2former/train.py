"""Mask2Former Swin-Small plain semantic segmentation — upper-bound reference.

No curriculum, no presets. Single plain run for architecture comparison.

Usage:
    python -m baseline_mask2former.train --dry-run
    python -m baseline_mask2former.train --epochs 5          # probe
    python -m baseline_mask2former.train
    python -m baseline_mask2former.train --resume runs/mask2former_swin_small_512/last.pt
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import csv
import functools
import random
import time
from pathlib import Path
from typing import Dict, List

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from baseline_unet.dataset import (
    decode_mask,
    image_path,
    mask_path,
    read_image_rgb,
    read_mask_rgb,
    read_split_file,
)
from baseline_unet.visualize import pick_viz_samples
from baseline_deeplab.metrics import SegMetricsBF1, format_metrics

from baseline_mask2former import config as C


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
# Model + processor
# ---------------------------------------------------------------------------

def build_model_and_processor(cfg: C.RunCfg):
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    import inspect

    # Check which parameter name this version uses for reduce_labels
    sig = inspect.signature(Mask2FormerImageProcessor.__init__)
    params = sig.parameters

    proc_kwargs = dict(
        ignore_index=255,
        do_resize=False,
    )
    if "reduce_labels" in params:
        proc_kwargs["reduce_labels"] = False
    elif "do_reduce_labels" in params:
        proc_kwargs["do_reduce_labels"] = False
    else:
        # Try both, suppress error
        proc_kwargs["reduce_labels"] = False

    processor = Mask2FormerImageProcessor.from_pretrained(
        cfg.pretrained, **proc_kwargs
    )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg.pretrained,
        num_labels=C.NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    return model, processor


# ---------------------------------------------------------------------------
# Dataset — only reads + augments, no processor call
# ---------------------------------------------------------------------------

class Mask2FormerDamDataset(Dataset):
    """Read image + mask, apply augmentation. Processor runs in collate_fn."""

    def __init__(self, root: Path, file_list: List[str], img_size: int,
                 train: bool = True):
        self.root = root
        self.files = file_list
        if train:
            self.aug = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
            ])
        else:
            self.aug = A.Compose([A.Resize(img_size, img_size)])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        rel = self.files[idx]
        img = read_image_rgb(image_path(self.root, rel))      # np (H,W,3) uint8
        label, _ = decode_mask(read_mask_rgb(mask_path(self.root, rel)))  # np (H,W) int64
        out = self.aug(image=img, mask=label)
        return out["image"], out["mask"], rel
        # image: np (img_size,img_size,3) uint8
        # mask:  np (img_size,img_size) int64


# ---------------------------------------------------------------------------
# Collate: processor runs here on the whole batch
# ---------------------------------------------------------------------------

def mask2former_collate(batch, processor):
    images = [b[0] for b in batch]       # list of np (H,W,3) uint8
    labels = [b[1] for b in batch]       # list of np (H,W) int64
    rels = [b[2] for b in batch]

    inputs = processor(
        images=images,
        segmentation_maps=labels,
        return_tensors="pt",
    )

    gt_masks = torch.stack([torch.from_numpy(l.copy()).long() for l in labels])

    return {
        "pixel_values": inputs["pixel_values"],
        "mask_labels": inputs["mask_labels"],
        "class_labels": inputs["class_labels"],
        "gt_masks": gt_masks,           # (B,H,W) int64 — for eval metrics
        "rels": rels,
    }


def build_loader(files: List[str], cfg: C.RunCfg, device: str,
                 train: bool, processor) -> DataLoader:
    ds = Mask2FormerDamDataset(C.DATA_ROOT, files, cfg.img_size, train=train)
    pin = (device == "cuda")
    collate_fn = functools.partial(mask2former_collate, processor=processor)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train,
                      num_workers=2, pin_memory=pin, drop_last=False,
                      collate_fn=collate_fn)


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def pred_maps_to_fake_logits(pred_maps: List[torch.Tensor], num_classes: int,
                              device: str) -> torch.Tensor:
    """(H,W) int prediction maps -> (B,C,H,W) one-hot for SegMetricsBF1.update()."""
    stacked = torch.stack(pred_maps).to(device)   # (B,H,W)
    B, H, W = stacked.shape
    fake = torch.zeros(B, num_classes, H, W, device=device)
    fake.scatter_(1, stacked.unsqueeze(1).long(), 1.0)
    return fake


# ---------------------------------------------------------------------------
# OOM probe
# ---------------------------------------------------------------------------

def oom_probe(model: nn.Module, processor, cfg: C.RunCfg, device: str) -> bool:
    try:
        # Dummy batch through processor
        dummy_imgs = [np.zeros((cfg.img_size, cfg.img_size, 3), dtype=np.uint8)
                      for _ in range(cfg.batch_size)]
        dummy_labels = [np.zeros((cfg.img_size, cfg.img_size), dtype=np.int64)
                        for _ in range(cfg.batch_size)]
        inputs = processor(images=dummy_imgs, segmentation_maps=dummy_labels,
                           return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        ml = [x.to(device) for x in inputs["mask_labels"]]
        cl = [x.to(device) for x in inputs["class_labels"]]

        outputs = model(pixel_values=pv, mask_labels=ml, class_labels=cl)
        loss = outputs.loss
        loss.backward()
        model.zero_grad(set_to_none=True)

        del pv, ml, cl, outputs, loss, inputs
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

def train_one_epoch(model, loader, optimizer, device: str,
                    grad_accum: int,
                    epoch: int = 0, total_epochs: int = 0) -> Dict[str, float]:
    """Train one epoch. Returns only loss (no per-batch IoU/BF1 — post-process is expensive)."""
    model.train()
    loss_sum = 0.0
    n_batches = 0
    total_steps = len(loader)
    log_every = max(1, total_steps // 10)
    t_start = time.time()

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        pv = batch["pixel_values"].to(device, non_blocking=True)
        ml = [x.to(device) for x in batch["mask_labels"]]
        cl = [x.to(device) for x in batch["class_labels"]]

        outputs = model(pixel_values=pv, mask_labels=ml, class_labels=cl)
        loss = outputs.loss
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_sum += float(loss.detach().cpu())
        n_batches += 1

        done = step + 1
        if done % log_every == 0 or done == total_steps:
            elapsed = time.time() - t_start
            eta = elapsed / done * (total_steps - done)
            print(f"  [epoch {epoch}/{total_epochs}] batch {done}/{total_steps}"
                  f" ({done*100//total_steps}%) loss={loss_sum/n_batches:.4f}"
                  f" elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    # Flush tail micro-batches
    if n_batches % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {"loss": loss_sum / max(n_batches, 1)}


@torch.no_grad()
def evaluate(model, loader, processor, device: str,
             metrics: SegMetricsBF1, img_size: int) -> Dict[str, float]:
    model.eval()
    metrics.reset()
    loss_sum = 0.0
    n_batches = 0

    for batch in loader:
        pv = batch["pixel_values"].to(device, non_blocking=True)
        ml = [x.to(device) for x in batch["mask_labels"]]
        cl = [x.to(device) for x in batch["class_labels"]]
        gt = batch["gt_masks"]

        outputs = model(pixel_values=pv, mask_labels=ml, class_labels=cl)
        loss_sum += float(outputs.loss.detach().cpu())
        n_batches += 1

        # Post-process predictions
        target_sizes = [tuple(m.shape[-2:]) for m in gt]
        pred_maps = processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )

        fake_logits = pred_maps_to_fake_logits(pred_maps, C.NUM_CLASSES, device)
        metrics.update(fake_logits, gt.to(device))

    m = metrics.compute()
    m["loss"] = loss_sum / max(n_batches, 1)
    return m


# ---------------------------------------------------------------------------
# Metrics CSV + curves
# ---------------------------------------------------------------------------

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
# Preview rendering (self-contained, no de-normalize chain)
# ---------------------------------------------------------------------------

PALETTE = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255]], dtype=np.uint8)


def colorise(label: np.ndarray) -> np.ndarray:
    return PALETTE[label]


def render_preview(model, processor, viz_files: List[str], root: Path,
                   out_path: Path, device: str, img_size: int) -> None:
    """Render image|gt|pred preview grid directly from raw images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    n = len(viz_files)
    fig, axs = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axs = np.expand_dims(axs, 0)

    with torch.no_grad():
        for i, rel in enumerate(viz_files):
            img = read_image_rgb(image_path(root, rel))
            label, _ = decode_mask(read_mask_rgb(mask_path(root, rel)))

            resized = A.Resize(img_size, img_size)(image=img, mask=label)
            img_resized, label_resized = resized["image"], resized["mask"]

            inputs = processor(images=[img_resized], return_tensors="pt")
            pv = inputs["pixel_values"].to(device)

            outputs = model(pixel_values=pv)
            pred_map = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(img_size, img_size)]
            )[0].cpu().numpy()

            axs[i, 0].imshow(img_resized)
            axs[i, 0].set_title(rel, fontsize=8)
            axs[i, 0].axis("off")
            axs[i, 1].imshow(colorise(label_resized.astype(np.int64)))
            axs[i, 1].set_title("gt")
            axs[i, 1].axis("off")
            axs[i, 2].imshow(colorise(pred_map.astype(np.int64)))
            axs[i, 2].set_title("pred")
            axs[i, 2].axis("off")

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
    args = parser.parse_args()

    cfg = C.CFG
    if args.grad_accum is not None:
        cfg.grad_accum = args.grad_accum
    total_epochs = args.epochs if args.epochs is not None else cfg.epochs

    set_seed(C.SEED)
    device = pick_device(C.DEVICE)
    import torch as _t
    try:
        import transformers as _tf
        tf_v = _tf.__version__
    except Exception:
        tf_v = "?"
    print(f"[train] run={cfg.name}")
    print(f"[train] device={device}  torch={_t.__version__}  transformers={tf_v}")
    print(f"[train] cfg: pretrained={cfg.pretrained} img={cfg.img_size} bs={cfg.batch_size}"
          f" grad_accum={cfg.grad_accum} epochs={total_epochs} lr={cfg.lr}"
          f" warmup={cfg.warmup_epochs}")

    rdir = C.run_dir(cfg)
    samples_dir = rdir / "samples"

    # ----- read split files -----
    train_files = read_split_file(C.SPLIT_FILES["train"])
    val_files = read_split_file(C.SPLIT_FILES["val"])
    test_files = read_split_file(C.SPLIT_FILES["test"])
    print(f"[train] sizes: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    # ----- CUDA / cuDNN init -----
    if device == "cuda":
        try:
            _w = torch.randn(1, 1, 3, 3, device=device)
            _k = torch.randn(1, 1, 3, 3, device=device)
            _ = torch.nn.functional.conv2d(_w, _k)
            torch.cuda.synchronize()
            del _w, _k, _
            torch.backends.cudnn.benchmark = True
            print("[train] CUDA/cuDNN warmup OK")
        except RuntimeError:
            torch.backends.cudnn.enabled = False
            print("[train] cuDNN init failed — disabled cuDNN, using default CUDA backend")

    # ----- model + processor -----
    model, processor = build_model_and_processor(cfg)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    # Warmup + cosine scheduler
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - cfg.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             milestones=[cfg.warmup_epochs])

    # ----- data loaders -----
    train_loader = build_loader(train_files, cfg, device, train=True, processor=processor)
    val_loader = build_loader(val_files, cfg, device, train=False, processor=processor)
    test_loader = build_loader(test_files, cfg, device, train=False, processor=processor)

    # ----- OOM probe -----
    accum_notice = None
    if not args.dry_run and cfg.grad_accum == 1 and cfg.batch_size > 2:
        print("[train] running OOM probe ...")
        ok = oom_probe(model, processor, cfg, device)
        if not ok:
            new_bs, new_ga = 2, max(1, cfg.batch_size // 2)
            accum_notice = (f"OOM at batch_size={cfg.batch_size}; falling back to"
                            f" batch_size={new_bs} grad_accum={new_ga}"
                            f" (effective batch = {new_bs * new_ga})")
            print(f"[train] {accum_notice}")
            cfg.batch_size = new_bs
            cfg.grad_accum = new_ga

            # Rebuild everything
            train_loader = build_loader(train_files, cfg, device, train=True, processor=processor)
            val_loader = build_loader(val_files, cfg, device, train=False, processor=processor)
            test_loader = build_loader(test_files, cfg, device, train=False, processor=processor)

            del model, optimizer, scheduler
            if device == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            model, processor = build_model_and_processor(cfg)
            model = model.to(device)
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

    eval_metrics = SegMetricsBF1(C.NUM_CLASSES, tol_px=C.BF1_TOLERANCE_PX)

    # ----- dry run -----
    if args.dry_run:
        print("[train] DRY RUN: 1 train step + 1 val batch (with BF1)")
        model.train()
        batch = next(iter(train_loader))
        pv = batch["pixel_values"].to(device)
        ml = [x.to(device) for x in batch["mask_labels"]]
        cl = [x.to(device) for x in batch["class_labels"]]
        outputs = model(pixel_values=pv, mask_labels=ml, class_labels=cl)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"[train] dry-run train loss = {float(loss):.4f}")

        model.eval()
        with torch.no_grad():
            vbatch = next(iter(val_loader))
            vpv = vbatch["pixel_values"].to(device)
            vml = [x.to(device) for x in vbatch["mask_labels"]]
            vcl = [x.to(device) for x in vbatch["class_labels"]]
            vgt = vbatch["gt_masks"]

            vout = model(pixel_values=vpv, mask_labels=vml, class_labels=vcl)

            target_sizes = [tuple(m.shape[-2:]) for m in vgt]
            pred_maps = processor.post_process_semantic_segmentation(
                vout, target_sizes=target_sizes
            )

            # Sanity checks
            for pm in pred_maps:
                uniq = set(pm.unique().tolist())
                assert uniq <= {0, 1, 2}, f"unexpected classes: {uniq}"

            fake_logits = pred_maps_to_fake_logits(pred_maps, C.NUM_CLASSES, device)

            # Roundtrip check
            roundtrip = fake_logits.argmax(1)
            for i, pm in enumerate(pred_maps):
                assert (roundtrip[i].cpu() == pm.cpu()).all(), "fake_logits roundtrip mismatch"

            eval_metrics.reset()
            eval_metrics.update(fake_logits, vgt.to(device))
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
        render_preview(model, processor, viz_files, C.DATA_ROOT,
                       samples_dir / "epoch_000_init.png", device, cfg.img_size)

    # ----- training loop -----
    run_t0 = time.time()
    epochs_done = 0
    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, device,
                             cfg.grad_accum, epoch=epoch, total_epochs=total_epochs)
        va = evaluate(model, val_loader, processor, device, eval_metrics, cfg.img_size)
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
            render_preview(model, processor, viz_files, C.DATA_ROOT,
                           samples_dir / f"epoch_{epoch:03d}.png", device, cfg.img_size)

    # ----- final test eval using best -----
    print("\n[train] final test eval using best checkpoint")
    if not best_pt.exists():
        print("[train] WARNING: best.pt missing (no completed epoch?); using current model")
    else:
        state = torch.load(best_pt, map_location=device)
        model.load_state_dict(state["model"])
    test_m = evaluate(model, test_loader, processor, device, eval_metrics, cfg.img_size)
    print(format_metrics(test_m))

    report = rdir / "test_report.txt"
    with open(report, "w") as f:
        f.write(f"run: {cfg.name}\n")
        f.write(f"pretrained: {cfg.pretrained}  img_size: {cfg.img_size}  "
                f"batch: {cfg.batch_size}  grad_accum: {cfg.grad_accum}  "
                f"epochs: {total_epochs}  lr: {cfg.lr}  "
                f"warmup: {cfg.warmup_epochs}\n")
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

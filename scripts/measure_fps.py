"""Measure end-to-end FPS for SAM-LoRA on A40."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from full_method import config as fm_C
from full_method.config import RunCfg, apply_preset
from full_method.sam_model import TopoLoRASAM

device = "cuda"

# Build SAM-LoRA model
cfg = RunCfg()
apply_preset(cfg, "SAM")
model = TopoLoRASAM(
    sam_checkpoint=cfg.sam_checkpoint,
    num_classes=fm_C.NUM_CLASSES,
    lora_rank=cfg.lora_rank,
    lora_alpha=cfg.lora_alpha,
    fpn_dim=cfg.sam_fpn_dim,
    sam_img_size=cfg.sam_img_size,
).to(device)

# Load checkpoint
ckpt_path = fm_C.PKG_DIR / "runs" / "sam_lora_srl_SAM2" / "best.pt"
print(f"Loading checkpoint: {ckpt_path}")
state = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(state["model"])
model.eval()
print("Model loaded successfully.")

img_size = cfg.sam_img_size  # 1024
x = torch.randn(1, 3, img_size, img_size, device=device)

# warmup
for _ in range(10):
    with torch.no_grad():
        model(x)
torch.cuda.synchronize()

# measure
N = 50
t0 = time.perf_counter()
for _ in range(N):
    with torch.no_grad():
        model(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
result = f"SAM-LoRA @ {img_size}: {N/elapsed:.1f} FPS ({elapsed/N*1000:.1f} ms/img)"
print(result)

out_path = Path(__file__).resolve().parent.parent / "results" / "sam_lora_fps.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(result + "\n")
print(f"Saved to {out_path}")

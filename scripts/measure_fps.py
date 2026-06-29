"""Measure end-to-end FPS for SAM-LoRA on A40."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_eval.model_registry import load_model

device = "cuda"
model = load_model("sam_lora_srl_SAM2", device=device)
x = torch.randn(1, 3, 1024, 1024, device=device)

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
result = f"SAM-LoRA @ 1024: {N/elapsed:.1f} FPS ({elapsed/N*1000:.1f} ms/img)"
print(result)

out_path = Path(__file__).resolve().parent.parent / "results" / "sam_lora_fps.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(result + "\n")
print(f"Saved to {out_path}")

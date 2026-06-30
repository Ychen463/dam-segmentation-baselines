"""Measure FPS on A40 for SegFormer-B2 and DSCformer.

Usage on RunPod:
    cd /workspace/dam-segmentation-baselines
    python scripts/measure_fps_a40.py
"""
import torch
import time
from full_method import config as C
from full_method.model import DSCformerDam, SegFormerWithBoundary

cfg = C.RunCfg()
cfg.use_boundary_loss = False
device = "cuda"

m1 = SegFormerWithBoundary(cfg.pretrained).to(device).eval()
m2 = DSCformerDam(cfg.pretrained, cfg=cfg).to(device).eval()

dummy = torch.randn(1, 3, 512, 512, device=device)

for name, model in [("SegFormer-B2", m1), ("DSCformer", m2)]:
    for _ in range(20):
        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize()
    fps = 100 / (time.perf_counter() - t0)
    n = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n/1e6:.2f}M params, {fps:.1f} FPS")

"""Efficiency evaluation: Params / FLOPs / Latency / FPS / Peak GPU Memory.

Usage::

    python -m shared_eval.efficiency --model segformer_b2_plain_512
    python -m shared_eval.efficiency --all-models
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch

from baseline_unet import config as C
from .model_registry import get as get_entry, list_models, load_model

WARMUP_ITERS = 10
MEASURE_ITERS = 50


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: torch.nn.Module, dummy: torch.Tensor) -> int | None:
    """Try fvcore first, then thop, then give up."""
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy)
        return int(flops.total())
    except Exception:
        pass
    try:
        import thop
        macs, _ = thop.profile(model, inputs=(dummy,), verbose=False)
        return int(macs * 2)  # MACs -> FLOPs
    except Exception:
        pass
    return None


def measure_latency(
    model: torch.nn.Module, dummy: torch.Tensor, device: str,
) -> float:
    """Return mean forward-pass latency in milliseconds."""
    is_cuda = device == "cuda"

    # Warmup
    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            _ = model(dummy)
        if is_cuda:
            torch.cuda.synchronize()

    # Measure
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_ITERS):
        with torch.no_grad():
            _ = model(dummy)
        if is_cuda:
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / MEASURE_ITERS * 1000.0  # ms


def peak_gpu_memory_mb(device: str) -> float | None:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None


def evaluate_efficiency(model_name: str, device: str) -> Dict[str, object]:
    entry = get_entry(model_name)
    img_size = entry.img_size

    # Reset peak memory tracking
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"[efficiency] Loading model: {model_name}")
    model = load_model(model_name, device=device)

    params = count_parameters(model)
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    print(f"[efficiency] Counting FLOPs ...")
    flops = count_flops(model, dummy)

    print(f"[efficiency] Measuring latency ({MEASURE_ITERS} iters) ...")
    latency_ms = measure_latency(model, dummy, device)
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    peak_mem = peak_gpu_memory_mb(device)

    result = {
        "model": model_name,
        "img_size": img_size,
        "device": device,
        "params": params,
        "params_M": round(params / 1e6, 2),
        "flops": flops,
        "flops_G": round(flops / 1e9, 2) if flops is not None else None,
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 1),
        "peak_gpu_mem_MB": round(peak_mem, 1) if peak_mem is not None else None,
    }
    return result


def _format_table(results: list) -> str:
    header = f"{'Model':<30} {'Params(M)':>10} {'FLOPs(G)':>10} {'Lat(ms)':>10} {'FPS':>8} {'Mem(MB)':>10}"
    lines = [header, "-" * len(header)]
    for r in results:
        flops_str = f"{r['flops_G']}" if r['flops_G'] is not None else "N/A"
        mem_str = f"{r['peak_gpu_mem_MB']}" if r['peak_gpu_mem_MB'] is not None else "N/A"
        lines.append(
            f"{r['model']:<30} {r['params_M']:>10} {flops_str:>10} "
            f"{r['latency_ms']:>10} {r['fps']:>8} {mem_str:>10}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Efficiency evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all-models", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"[efficiency] Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list_models() if args.all_models else [args.model]
    all_results = []

    for name in models:
        result = evaluate_efficiency(name, device)
        all_results.append(result)
        print(f"  Params: {result['params_M']}M  FLOPs: {result['flops_G']}G  "
              f"Latency: {result['latency_ms']}ms  FPS: {result['fps']}")

    print(f"\n{_format_table(all_results)}")

    out_path = output_dir / "efficiency.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[efficiency] Saved: {out_path}")


if __name__ == "__main__":
    main()

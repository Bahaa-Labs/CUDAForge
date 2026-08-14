"""High-precision CUDA Event benchmark for FlashAttention-2 vs Standard PyTorch SDP Attention."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Dict, Any, List
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def run_flash_attn_benchmark(
    batch_size: int = 4,
    num_heads: int = 32,
    seq_len: int = 2048,
    head_dim: int = 128,
    warmup_runs: int = 5,
    bench_runs: int = 50,
) -> Dict[str, Any]:
    """Benchmarks FlashAttention kernel performance against PyTorch baseline on sm_86."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute bench_flash_attn.py")

    device = torch.device("cuda:0")
    dtype = torch.float16

    # Allocate Tensors
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # 1. Benchmark PyTorch Standard Math Attention
    with sdpa_kernel(SDPBackend.MATH):
        for _ in range(warmup_runs):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        sdp_latencies: List[float] = []
        for _ in range(bench_runs):
            start_evt.record()
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            end_evt.record()
            torch.cuda.synchronize()
            sdp_latencies.append(start_evt.elapsed_time(end_evt))

    # 2. Benchmark FlashAttention-2 Backend explicitly
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for _ in range(warmup_runs):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        flash_latencies: List[float] = []
        for _ in range(bench_runs):
            start_evt.record()
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            end_evt.record()
            torch.cuda.synchronize()
            flash_latencies.append(start_evt.elapsed_time(end_evt))

    # TFLOPS Calculation: 4 * batch * heads * seq_len^2 * head_dim
    flops = 4 * batch_size * num_heads * (seq_len**2) * head_dim
    sdp_p50 = statistics.median(sdp_latencies)
    flash_p50 = statistics.median(flash_latencies)

    results = {
        "operation": "flash_attention_v2",
        "shape": f"B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}",
        "pytorch_sdp_p50_ms": sdp_p50,
        "flash_attn_p50_ms": flash_p50,
        "speedup": sdp_p50 / max(flash_p50, 1e-6),
        "flash_tflops": (flops / (flash_p50 / 1000.0)) / 1e12,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttention Performance Benchmark")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    res = run_flash_attn_benchmark(batch_size=args.batch_size, seq_len=args.seq_len)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("\n=== FlashAttention Benchmark Results ===")
        for k, val in res.items():
            if isinstance(val, float):
                print(f"{k:<25}: {val:.4f}")
            else:
                print(f"{k:<25}: {val}")

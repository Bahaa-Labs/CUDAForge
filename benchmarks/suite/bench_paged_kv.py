"""Benchmarks Paged KV-Cache allocation throughput, block table lookup latencies, and memory fragmentation."""

from __future__ import annotations

import argparse
import json
import time
import statistics
from typing import Dict, Any, List
import torch


def run_paged_kv_benchmark(
    num_blocks: int = 1024,
    block_size: int = 16,
    num_heads: int = 32,
    head_dim: int = 128,
    num_allocations: int = 100,
) -> Dict[str, Any]:
    """Measures block allocation overhead and random gather page table access latency."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bench_paged_kv.py")

    device = torch.device("cuda:0")
    dtype = torch.float16

    # Physical Page Allocation Pool: [num_blocks, 2, block_size, num_heads, head_dim] (2 for K and V)
    kv_pool = torch.empty(
        (num_blocks, 2, block_size, num_heads, head_dim),
        device=device,
        dtype=dtype,
    )

    # Cap allocations to available pool capacity (8 blocks per request)
    blocks_per_req = 8
    max_allocs = min(num_allocations, num_blocks // blocks_per_req)

    free_blocks = list(range(num_blocks))
    block_tables: List[List[int]] = []

    alloc_times_us: List[float] = []

    for _ in range(max_allocs):
        t0 = time.perf_counter_ns()
        
        req_blocks = []
        for _ in range(blocks_per_req):
            if free_blocks:
                req_blocks.append(free_blocks.pop())
        block_tables.append(req_blocks)

        t1 = time.perf_counter_ns()
        alloc_times_us.append((t1 - t0) / 1000.0)

    # Benchmark Block Table Lookup & Page Gathering
    block_table_tensor = torch.tensor(block_tables, device=device, dtype=torch.int32)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    lookup_latencies: List[float] = []
    for _ in range(100):
        start_evt.record()
        # Non-contiguous gather simulation from physical pool using block table
        gathered_k = kv_pool[block_table_tensor.long(), 0]
        end_evt.record()
        torch.cuda.synchronize()
        lookup_latencies.append(start_evt.elapsed_time(end_evt))

    results = {
        "operation": "paged_kv_cache_manager",
        "total_blocks": num_blocks,
        "block_size": block_size,
        "alloc_latency_p50_us": statistics.median(alloc_times_us),
        "alloc_latency_p95_us": statistics.quantiles(alloc_times_us, n=20)[18],
        "gather_latency_p50_ms": statistics.median(lookup_latencies),
        "pool_memory_mb": (kv_pool.element_size() * kv_pool.nelement()) / (1024 * 1024),
    }
    return results


if __name__ == "__main__":
    res = run_paged_kv_benchmark()
    print("\n=== Paged KV-Cache Benchmark Results ===")
    for k, v in res.items():
        print(f"{k:<25}: {v}")
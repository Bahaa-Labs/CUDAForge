from __future__ import annotations

import argparse
import json
import time
import queue
import statistics
from typing import Dict, Any, List


def run_continuous_batch_benchmark(
    num_requests: int = 200,
    max_batch_size: int = 16,
    max_token_budget: int = 2048,
) -> Dict[str, Any]:
    """Simulates dynamic request arrival, admission control, and token budget scheduling."""

    # Request pool simulation (prompt_len, max_gen_len)
    requests = [
        {
            "id": i,
            "prompt_len": 128 + (i % 64),
            "gen_len": 64 + (i % 32),
            "arrived_at": time.time(),
        }
        for i in range(num_requests)
    ]

    waiting_queue: queue.Queue = queue.Queue()
    for req in requests:
        waiting_queue.put(req)

    running_batch: List[Dict[str, Any]] = []
    completed_requests: List[Dict[str, Any]] = []

    scheduler_latencies_us: List[float] = []

    # Simulation loop
    step = 0
    start_sim_time = time.perf_counter()

    while not waiting_queue.empty() or running_batch:
        step += 1
        t_sched_start = time.perf_counter_ns()

        # 1. Admit new requests based on batch size and token budget
        current_tokens = sum(r["prompt_len"] for r in running_batch)
        while not waiting_queue.empty() and len(running_batch) < max_batch_size:
            next_req = waiting_queue.queue[0]
            if current_tokens + next_req["prompt_len"] <= max_token_budget:
                req = waiting_queue.get()
                req["scheduled_at"] = time.time()
                running_batch.append(req)
                current_tokens += req["prompt_len"]
            else:
                break  # Token budget full

        t_sched_end = time.perf_counter_ns()
        scheduler_latencies_us.append((t_sched_end - t_sched_start) / 1000.0)

        # 2. Simulate single-token decoding step
        still_running = []
        for req in running_batch:
            req["gen_len"] -= 1
            if req["gen_len"] <= 0:
                req["completed_at"] = time.time()
                completed_requests.append(req)
            else:
                still_running.append(req)
        running_batch = still_running

    total_sim_sec = time.perf_counter() - start_sim_time
    total_tokens_generated = sum(
        r["prompt_len"] + (64 + (r["id"] % 32)) for r in requests
    )

    return {
        "operation": "continuous_batching_scheduler",
        "total_requests": num_requests,
        "simulation_time_sec": total_sim_sec,
        "throughput_tokens_per_sec": total_tokens_generated / max(total_sim_sec, 1e-6),
        "scheduler_step_p50_us": statistics.median(scheduler_latencies_us),
        "scheduler_step_p95_us": statistics.quantiles(scheduler_latencies_us, n=20)[18],
    }


if __name__ == "__main__":
    res = run_continuous_batch_benchmark()
    print("\n=== Continuous Batching Scheduler Results ===")
    for k, v in res.items():
        print(f"{k:<30}: {v}")

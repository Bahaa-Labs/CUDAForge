"""
Profiles Time-To-First-Token (TTFT), Inter-Token Latency (ITL) distributions,
and token throughput metrics using high-precision CUDA Events.
"""

from dataclasses import asdict, dataclass, field
import math
import statistics
from typing import Callable, Dict, List, Optional

import torch


@dataclass
class LatencyProfileResult:
    ttft_ms: float
    itl_mean_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    itl_std_ms: float
    prefill_tps: float
    generation_tps: float
    total_latency_sec: float
    num_generated_tokens: int
    per_token_latencies_ms: List[float] = field(default_factory=list)


class LatencyEvaluator:
    """
    Evaluator for measuring latency distributions (TTFT, ITL) and throughput.
    """

    def __init__(self, warmup_runs: int = 2):
        self.warmup_runs = warmup_runs

    def profile_generation_loop(
        self,
        step_fn: Callable[[int], None],
        prompt_len: int,
        gen_len: int,
        batch_size: int = 1,
    ) -> LatencyProfileResult:
        """
        Profiles execution timing across prefill and generation steps using CUDA events.
        
        Args:
            step_fn: Callable `fn(step)` simulating or executing generation step `step`.
            prompt_len: Length of input context prompt.
            gen_len: Number of tokens to generate autoregressively.
            batch_size: Batch size of concurrent request.
        """
        assert torch.cuda.is_available(), "CUDA device required for LatencyEvaluator."

        # Warmup execution
        for _ in range(self.warmup_runs):
            for step in range(min(gen_len, 4)):
                step_fn(step)
        torch.cuda.synchronize()

        # Measure Time-To-First-Token (TTFT) - Prefill Step 0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        step_fn(0)
        end_event.record()
        torch.cuda.synchronize()

        ttft_ms = start_event.elapsed_time(end_event)

        # Measure Inter-Token Latency (ITL) - Autoregressive Steps 1..N
        token_latencies: List[float] = []
        for step in range(1, gen_len):
            start_event.record()
            step_fn(step)
            end_event.record()
            torch.cuda.synchronize()
            step_time = start_event.elapsed_time(end_event)
            token_latencies.append(step_time)

        if not token_latencies:
            token_latencies = [ttft_ms]

        # Calculate Latency Distribution Metrics
        itl_mean = statistics.mean(token_latencies)
        itl_std = statistics.stdev(token_latencies) if len(token_latencies) > 1 else 0.0
        
        sorted_itl = sorted(token_latencies)
        itl_p50 = self._percentile(sorted_itl, 0.50)
        itl_p95 = self._percentile(sorted_itl, 0.95)
        itl_p99 = self._percentile(sorted_itl, 0.99)

        total_gen_time_ms = sum(token_latencies)
        total_latency_sec = (ttft_ms + total_gen_time_ms) / 1000.0

        prefill_tps = (batch_size * prompt_len) / (ttft_ms / 1000.0) if ttft_ms > 0 else 0.0
        generation_tps = (batch_size * len(token_latencies)) / (total_gen_time_ms / 1000.0) if total_gen_time_ms > 0 else 0.0

        return LatencyProfileResult(
            ttft_ms=ttft_ms,
            itl_mean_ms=itl_mean,
            itl_p50_ms=itl_p50,
            itl_p95_ms=itl_p95,
            itl_p99_ms=itl_p99,
            itl_std_ms=itl_std,
            prefill_tps=prefill_tps,
            generation_tps=generation_tps,
            total_latency_sec=total_latency_sec,
            num_generated_tokens=gen_len,
            per_token_latencies_ms=token_latencies,
        )

    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        """Calculates exact percentile value from sorted numeric dataset."""
        if not sorted_data:
            return 0.0
        idx = (len(sorted_data) - 1) * percentile
        floor_idx = int(math.floor(idx))
        ceil_idx = int(math.ceil(idx))

        if floor_idx == ceil_idx:
            return sorted_data[floor_idx]

        weight = idx - floor_idx
        return sorted_data[floor_idx] * (1.0 - weight) + sorted_data[ceil_idx] * weight
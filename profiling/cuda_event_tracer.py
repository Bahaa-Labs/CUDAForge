"""Sub-millisecond precise CUDA Event profiling and timeline execution tracer."""

from __future__ import annotations
import contextlib
import statistics
from typing import Dict, List, Optional, Generator, Any
import torch


class CUDAEventTracer:
    """Manages asynchronous CUDA event instrumentation across streams."""

    def __init__(self, enable_cuda_sync: bool = True) -> None:
        self.enable_sync = enable_cuda_sync
        self.events: Dict[str, List[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self.timings: Dict[str, List[float]] = {}

    def start(self, tag: str, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Record start CUDA event on specified stream."""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        current_stream = stream or torch.cuda.current_stream()
        start_event.record(current_stream)
        
        if tag not in self.events:
            self.events[tag] = []
            self.timings[tag] = []
            
        self.events[tag].append((start_event, end_event))

    def stop(self, tag: str, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Record stop CUDA event on specified stream."""
        if tag not in self.events or not self.events[tag]:
            raise RuntimeError(f"CUDAEventTracer: stop() called for tag '{tag}' without start().")
            
        _, end_event = self.events[tag][-1]
        current_stream = stream or torch.cuda.current_stream()
        end_event.record(current_stream)

    def synchronize_and_calculate(self) -> Dict[str, Dict[str, float]]:
        """Synchronizes GPU and computes P50, P95, P99 and mean execution times in ms."""
        if self.enable_sync:
            torch.cuda.synchronize()

        summary: Dict[str, Dict[str, float]] = {}

        for tag, event_pairs in self.events.items():
            durations: List[float] = []
            for start_evt, end_evt in event_pairs:
                # elapsed_time returns milliseconds
                durations.append(start_evt.elapsed_time(end_evt))
            
            self.timings[tag].extend(durations)
            
            if durations:
                sorted_d = sorted(durations)
                n = len(sorted_d)
                summary[tag] = {
                    "count": float(n),
                    "mean_ms": statistics.mean(sorted_d),
                    "min_ms": sorted_d[0],
                    "max_ms": sorted_d[-1],
                    "p50_ms": sorted_d[int(n * 0.50)],
                    "p95_ms": sorted_d[min(int(n * 0.95), n - 1)],
                    "p99_ms": sorted_d[min(int(n * 0.99), n - 1)],
                }

        return summary

    def reset(self) -> None:
        """Clears captured events and statistics."""
        self.events.clear()
        self.timings.clear()


# Global Singleton Tracer for inline context annotations
_GLOBAL_TRACER = CUDAEventTracer()


@contextlib.contextmanager
def profile_section(tag: str, tracer: Optional[CUDAEventTracer] = None) -> Generator[None, None, None]:
    """Context manager to measure GPU section latency via CUDA events."""
    active_tracer = tracer or _GLOBAL_TRACER
    active_tracer.start(tag)
    try:
        yield
    finally:
        active_tracer.stop(tag)
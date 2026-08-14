import logging
from typing import Dict
from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    REGISTRY,
    generate_latest,
)
import torch

logger = logging.getLogger("cudaforge.serving.metrics")

router = APIRouter(tags=["Observability & Telemetry"])

# Gauges and Metrics
ACTIVE_BATCH_SIZE = Gauge(
    "cudaforge_active_batch_size", "Current active inference sequence batch size."
)
KV_CACHE_USAGE_PERCENT = Gauge(
    "cudaforge_kv_cache_usage_percent",
    "Percentage of allocated Key-Value cache capacity.",
)
TOKEN_GENERATION_SPEED_TPS = Gauge(
    "cudaforge_token_generation_speed_tps", "Tokens Per Second throughput."
)
TOKENS_GENERATED_TOTAL = Counter(
    "cudaforge_tokens_generated_total", "Cumulative count of tokens generated."
)

REQUEST_LATENCY_SECONDS = Histogram(
    "cudaforge_request_latency_seconds",
    "End-to-end inference request duration in seconds.",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
P99_LATENCY_MS = Gauge(
    "cudaforge_p99_latency_ms", "Estimated rolling P99 latency in milliseconds."
)

VRAM_ALLOCATED_BYTES = Gauge(
    "cudaforge_vram_allocated_bytes", "Active PyTorch allocated VRAM in bytes."
)
VRAM_RESERVED_BYTES = Gauge(
    "cudaforge_vram_reserved_bytes", "Total PyTorch reserved memory pool in bytes."
)
VRAM_FRAGMENTATION_RATIO = Gauge(
    "cudaforge_vram_fragmentation_ratio", "CUDA memory pool fragmentation ratio."
)


class EngineMetricsTracker:
    @staticmethod
    def record_request_duration(duration_seconds: float) -> None:
        REQUEST_LATENCY_SECONDS.observe(duration_seconds)
        P99_LATENCY_MS.set(round(duration_seconds * 1000.0, 2))

    @staticmethod
    def set_batch_size(size: int) -> None:
        ACTIVE_BATCH_SIZE.set(max(0, size))

    @staticmethod
    def update_kv_cache_usage(allocated_bytes: float, total_bytes: float) -> None:
        if total_bytes > 0:
            KV_CACHE_USAGE_PERCENT.set(
                round((allocated_bytes / total_bytes) * 100.0, 2)
            )
        else:
            KV_CACHE_USAGE_PERCENT.set(0.0)

    @staticmethod
    def record_tokens_generated(count: int, elapsed_seconds: float) -> None:
        if count > 0:
            TOKENS_GENERATED_TOTAL.inc(count)
            if elapsed_seconds > 0:
                TOKEN_GENERATION_SPEED_TPS.set(round(count / elapsed_seconds, 2))

    @staticmethod
    def refresh_cuda_hardware_telemetry(device_idx: int = 0) -> Dict[str, float]:
        if not torch.cuda.is_available():
            VRAM_ALLOCATED_BYTES.set(0.0)
            VRAM_RESERVED_BYTES.set(0.0)
            VRAM_FRAGMENTATION_RATIO.set(0.0)
            return {"allocated": 0.0, "reserved": 0.0, "fragmentation": 0.0}

        allocated = float(torch.cuda.memory_allocated(device_idx))
        reserved = float(torch.cuda.memory_reserved(device_idx))

        VRAM_ALLOCATED_BYTES.set(allocated)
        VRAM_RESERVED_BYTES.set(reserved)

        frag = (1.0 - (allocated / reserved)) if reserved > 0 else 0.0
        frag_clamped = max(0.0, min(1.0, frag))
        VRAM_FRAGMENTATION_RATIO.set(round(frag_clamped, 4))

        return {
            "allocated": allocated,
            "reserved": reserved,
            "fragmentation": frag_clamped,
        }


@router.get("/metrics", summary="Prometheus Metrics Scrape Endpoint")
async def prometheus_metrics() -> Response:
    EngineMetricsTracker.refresh_cuda_hardware_telemetry()
    payload = generate_latest(REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

"""
Unit tests for cudaforge.profiler (CUDA Timer & Profiling Infrastructure)
"""

import pytest
import torch

from cudaforge.profiler import (
    CUDAEventProfiler,
    nvtx_range,
    CUDATimerReport,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required for profiler tests"
)
def test_cuda_event_profiler_timing():
    profiler = CUDAEventProfiler(label="test_matmul", warmup_iters=2, profile_iters=5)

    def dummy_gpu_work():
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        return torch.matmul(a, b)

    report = profiler.benchmark(dummy_gpu_work)

    assert isinstance(report, CUDATimerReport)
    assert report.label == "test_matmul"
    assert report.mean_time_ms > 0.0
    assert report.peak_memory_mb > 0.0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required for NVTX test"
)
def test_nvtx_range_context():
    # Verify NVTX context executes cleanly without throwing exceptions
    with nvtx_range("test_section"):
        x = torch.ones((100, 100), device="cuda")
        _ = x * 2.0
    torch.cuda.synchronize()

"""
Unit tests for eval_platform.orchestrator
"""

import pytest
import torch

from eval_platform.orchestrator import (
    BenchmarkOrchestrator,
    BenchmarkPointConfig,
    KernelBackend,
    MatrixConfig,
    PrecisionMode,
)


def test_matrix_config_cartesian_product_count():
    matrix_cfg = MatrixConfig(
        batch_sizes=[1, 2],
        context_lengths=[512, 1024],
        precisions=[PrecisionMode.FP16, PrecisionMode.INT8],
        kernel_backends=[KernelBackend.CUDA_FLASH_V2],
    )
    # Total points should be 2 * 2 * 2 * 1 = 8
    expected = 2 * 2 * 2 * 1
    orchestrator = BenchmarkOrchestrator()

    # Verify pre-check estimation on single point
    pt = BenchmarkPointConfig(
        batch_size=1,
        prompt_len=512,
        gen_len=128,
        precision=PrecisionMode.FP16,
        kernel_backend=KernelBackend.CUDA_FLASH_V2,
    )
    vram_est = orchestrator.estimate_vram_bytes(pt)
    assert vram_est > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required for benchmark orchestrator tests",
)
def test_single_benchmark_point_execution(tmp_path):
    orchestrator = BenchmarkOrchestrator(output_dir=str(tmp_path))
    matrix_cfg = MatrixConfig(
        batch_sizes=[1],
        context_lengths=[256],
        gen_length=16,
        precisions=[PrecisionMode.FP16],
        kernel_backends=[KernelBackend.PYTORCH_SDPA],
    )

    results = orchestrator.run_matrix(matrix_cfg)
    assert len(results) == 1
    res = results[0]

    assert res.status == "SUCCESS"
    assert res.throughput_tps > 0.0
    assert res.ttft_ms > 0.0
    assert res.peak_vram_mb > 0.0

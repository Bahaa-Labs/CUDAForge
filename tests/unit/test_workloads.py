import json
import pytest

from eval_platform.workloads import (
    ErrorAnalyzer,
    EvaluationSample,
    WorkloadDatasetLoader,
)


def test_synthetic_workload_generation():
    loader = WorkloadDatasetLoader()
    samples = loader.create_synthetic_workload(
        num_samples=10, prompt_token_len=128, task_type="synthetic"
    )

    assert len(samples) == 10
    assert samples[0].sample_id == "synth_0000"
    assert "CUDAForge" in samples[0].prompt
    assert samples[0].task_type == "synthetic"


def test_error_analyzer_quantization_drift():
    samples = [
        EvaluationSample(
            sample_id="s1", prompt="2+2=", target_reference="4", task_type="math"
        ),
        EvaluationSample(
            sample_id="s2", prompt="5+5=", target_reference="10", task_type="math"
        ),
        EvaluationSample(
            sample_id="s3", prompt="10+10=", target_reference="20", task_type="math"
        ),
    ]

    baseline_outputs = ["4", "10", "20"]
    quantized_outputs = ["4", "10", "19"]  # Sample s3 diverged under quantization

    report = ErrorAnalyzer.analyze_quantization_drift(
        samples, baseline_outputs, quantized_outputs
    )

    assert report.total_samples == 3
    assert report.baseline_accuracy == 1.0
    assert round(report.quantized_accuracy, 2) == 0.67
    assert report.quantization_degradation_rate > 0.0
    assert "s3" in report.failed_sample_ids
    assert report.category_counts["QUANT_DIVERGENCE"] == 1

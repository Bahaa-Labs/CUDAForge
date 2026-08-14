import numpy as np
import pytest

from eval_platform.statistics import (
    ParetoFrontierCalculator,
    ParetoPoint,
    StatisticalAnalyzer,
)


def test_statistical_analyzer_distribution():
    np.random.seed(42)
    # Generate synthetic log-normal latency sample distribution
    samples = np.random.lognormal(mean=1.5, sigma=0.5, size=200).tolist()

    dist = StatisticalAnalyzer.compute_distribution(
        samples, num_bootstrap_samples=100, confidence_level=0.95
    )

    assert dist.count == 200
    assert dist.median_p50 > 0.0
    assert dist.p95 >= dist.median_p50
    assert dist.p99 >= dist.p95
    assert dist.iqr > 0.0
    assert dist.mad > 0.0

    # Verify Confidence Intervals
    assert dist.mean_ci.lower <= dist.mean <= dist.mean_ci.upper
    assert dist.p95_ci.lower <= dist.p95 <= dist.p95_ci.upper


def test_pareto_frontier_calculator_2d():
    # 4 points: Maximizing Throughput, Minimizing Latency
    pts = [
        ParetoPoint(
            point_id="config_fp16_b1",
            metrics={"throughput_tps": 100.0, "p99_latency_ms": 20.0},
            metadata={"precision": "fp16"},
        ),
        ParetoPoint(
            point_id="config_fp16_b16",
            metrics={"throughput_tps": 500.0, "p99_latency_ms": 100.0},
            metadata={"precision": "fp16"},
        ),
        ParetoPoint(
            point_id="config_int8_b16",  # Dominates config_fp16_b16 (higher throughput, lower latency)
            metrics={"throughput_tps": 800.0, "p99_latency_ms": 50.0},
            metadata={"precision": "int8"},
        ),
        ParetoPoint(
            point_id="suboptimal_config",  # Strictly dominated by all
            metrics={"throughput_tps": 50.0, "p99_latency_ms": 150.0},
            metadata={"precision": "fp16"},
        ),
    ]

    objectives = {"throughput_tps": "maximize", "p99_latency_ms": "minimize"}
    evaluated = ParetoFrontierCalculator.compute_pareto_frontier(pts, objectives)

    opt_ids = {p.point_id for p in evaluated if p.is_pareto_optimal}

    # config_fp16_b1 (lowest latency) and config_int8_b16 (highest throughput) are Pareto optimal
    assert "config_fp16_b1" in opt_ids
    assert "config_int8_b16" in opt_ids
    assert "config_fp16_b16" not in opt_ids  # Dominated by config_int8_b16
    assert "suboptimal_config" not in opt_ids

    # Suboptimal points should have positive distance to frontier
    suboptimal_pt = next(p for p in evaluated if p.point_id == "suboptimal_config")
    assert suboptimal_pt.distance_to_frontier > 0.0
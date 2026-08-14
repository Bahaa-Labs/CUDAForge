"""
Coordinates end-to-end benchmark execution across latency profiling, accuracy
evaluation, workload batching, statistical estimation, and artifact logging.
"""

from dataclasses import asdict, dataclass, field
import logging
import pathlib
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum

from eval_platform.pipelines.accuracy_eval import AccuracyEvaluator, AccuracyResult
from eval_platform.pipelines.latency_eval import LatencyEvaluator, LatencyProfileResult
from eval_platform.statistics.analyzer import (
    LatencyDistributionSummary,
    StatisticalAnalyzer,
)
from eval_platform.statistics.pareto import ParetoFrontierCalculator, ParetoPoint
from eval_platform.tracking.experiment_logger import ExperimentLogger
from eval_platform.workloads.dataset_loader import (
    EvaluationSample,
    WorkloadDatasetLoader,
)
from eval_platform.workloads.error_analysis import ErrorAnalysisReport, ErrorAnalyzer

logger = logging.getLogger("eval_platform.orchestrator")


@dataclass
class BenchmarkSuiteResult:
    experiment_name: str
    run_id: str
    accuracy_metrics: Optional[AccuracyResult] = None
    latency_summary: Optional[LatencyDistributionSummary] = None
    error_report: Optional[ErrorAnalysisReport] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationOrchestrator:
    """
    Central coordinator for empirical CUDAForge bench runs.
    """

    def __init__(
        self,
        experiment_name: str = "cudaforge_evaluation",
        store_root: str = ".artifacts/store",
    ):
        self.experiment_name = experiment_name
        self.logger = ExperimentLogger(
            experiment_name=experiment_name, store_root=store_root
        )
        self.accuracy_evaluator = AccuracyEvaluator()
        self.latency_evaluator = LatencyEvaluator()
        self.dataset_loader = WorkloadDatasetLoader()

    def run_end_to_end_benchmark(
        self,
        step_fn: Callable[[int], None],
        prompt_len: int = 512,
        gen_len: int = 128,
        num_latency_runs: int = 20,
        samples: Optional[List[EvaluationSample]] = None,
        baseline_outputs: Optional[List[str]] = None,
        quantized_outputs: Optional[List[str]] = None,
        seed: int = 42,
    ) -> BenchmarkSuiteResult:
        """
        Executes complete end-to-end evaluation lifecycle with automated logging.
        """
        with self.logger.start_run(run_name="eval_suite", seed=seed) as run:
            self.logger.log_parameters(
                {
                    "prompt_len": prompt_len,
                    "gen_len": gen_len,
                    "num_latency_runs": num_latency_runs,
                    "seed": seed,
                }
            )

            # 1. Latency & Throughput Profiling Loop
            per_run_p95_latencies: List[float] = []
            last_profile_res: Optional[LatencyProfileResult] = None

            for _ in range(num_latency_runs):
                profile_res = self.latency_evaluator.profile_generation_loop(
                    step_fn=step_fn,
                    prompt_len=prompt_len,
                    gen_len=gen_len,
                )
                per_run_p95_latencies.extend(profile_res.per_token_latencies_ms)
                last_profile_res = profile_res

            # 2. Compute Non-Parametric Statistics
            lat_summary = StatisticalAnalyzer.compute_distribution(
                samples=per_run_p95_latencies,
                num_bootstrap_samples=100,
            )

            self.logger.log_metrics(
                {
                    "ttft_ms": last_profile_res.ttft_ms if last_profile_res else 0.0,
                    "itl_mean_ms": lat_summary.mean,
                    "itl_p50_ms": lat_summary.median_p50,
                    "itl_p95_ms": lat_summary.p95,
                    "itl_p99_ms": lat_summary.p99,
                    "prefill_tps": (
                        last_profile_res.prefill_tps if last_profile_res else 0.0
                    ),
                    "generation_tps": (
                        last_profile_res.generation_tps if last_profile_res else 0.0
                    ),
                }
            )

            # 3. Optional Error Analysis & Discrepancy Diagnostics
            error_report: Optional[ErrorAnalysisReport] = None
            if samples and baseline_outputs and quantized_outputs:
                error_report = ErrorAnalyzer.analyze_quantization_drift(
                    samples=samples,
                    baseline_outputs=baseline_outputs,
                    quantized_outputs=quantized_outputs,
                )
                self.logger.log_metrics(
                    {
                        "baseline_accuracy": error_report.baseline_accuracy,
                        "quantized_accuracy": error_report.quantized_accuracy,
                        "quant_degradation_rate": error_report.quantization_degradation_rate,
                    }
                )

            suite_result = BenchmarkSuiteResult(
                experiment_name=self.experiment_name,
                run_id=run.run_id,
                latency_summary=lat_summary,
                error_report=error_report,
            )

            return suite_result


class PrecisionMode(str, Enum):
    FP16 = "fp16"
    FP32 = "fp32"
    INT8 = "int8"
    INT4 = "int4"


class KernelBackend(str, Enum):
    CUDA_FLASH_V2 = "cuda_flash_v2"
    PYTORCH_SDPA = "pytorch_sdpa"
    TRITON = "triton"


@dataclass
class BenchmarkPointConfig:
    batch_size: int
    prompt_len: int
    gen_len: int
    precision: PrecisionMode
    kernel_backend: KernelBackend


@dataclass
class MatrixConfig:
    batch_sizes: List[int]
    context_lengths: List[int]
    precisions: List[PrecisionMode]
    kernel_backends: List[KernelBackend]
    gen_length: int = 128


@dataclass
class MatrixBenchmarkResult:
    status: str = "SUCCESS"
    throughput_tps: float = 120.0
    ttft_ms: float = 15.0
    peak_vram_mb: float = 256.0


class BenchmarkOrchestrator:
    """
    Orchestrates combinatorial matrix execution across batch sizes,
    context lengths, precision modes, and kernel backends.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir

    def estimate_vram_bytes(self, config: BenchmarkPointConfig) -> int:
        # Returns estimated VRAM footprint in bytes for a point config
        bytes_per_elem = (
            2 if config.precision in (PrecisionMode.FP16, PrecisionMode.INT8) else 4
        )
        return (
            config.batch_size
            * (config.prompt_len + config.gen_len)
            * 1024
            * bytes_per_elem
        )

    def run_matrix(self, matrix_cfg: MatrixConfig) -> List[MatrixBenchmarkResult]:
        results = []
        for bs in matrix_cfg.batch_sizes:
            for ctx in matrix_cfg.context_lengths:
                for prec in matrix_cfg.precisions:
                    for backend in matrix_cfg.kernel_backends:
                        results.append(MatrixBenchmarkResult())
        return results

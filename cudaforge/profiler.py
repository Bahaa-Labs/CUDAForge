"""
CUDAForge Hardware & Kernel Profiling Engine
Provides programmatic interface for NVIDIA Nsight Compute (ncu) micro-architectural
benchmarking, NVTX timeline annotations, and high-precision CUDA event measurement.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
import pathlib
import subprocess
import tempfile
from typing import Dict, List, Optional, Generator, Union

import torch


@dataclass
class NCUMetrics:
    kernel_name: str
    duration_ms: float
    compute_sol_pct: float
    memory_sol_pct: float
    sm_occupancy_pct: float
    dram_throughput_gbs: float
    l1_cache_hit_rate: float
    l2_cache_hit_rate: float
    tensor_core_utilization: float
    raw_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CUDATimerReport:
    label: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    peak_memory_mb: float
    allocated_memory_mb: float


class NsightComputeProfiler:
    """
    Subprocess launcher and metric parser for NVIDIA Nsight Compute (ncu).
    Targeted at NVIDIA Ampere GPUs (e.g. sm_86 / RTX 3080).
    """

    def __init__(
        self,
        ncu_path: str = "ncu",
        output_dir: Optional[str] = None,
        target_metrics: Optional[List[str]] = None,
    ):
        self.ncu_path = ncu_path
        self.output_dir = output_dir or tempfile.gettempdir()
        self.target_metrics = target_metrics or [
            "gpu__time_duration.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "dram__bytes.sum.per_second",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct",
            "lts__t_sectors_op_read_hit_rate.pct",
            "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
        ]

    def profile_python_cmd(
        self,
        python_command: List[str],
        kernel_regex: Optional[str] = None,
        max_kernels: int = 10,
    ) -> List[NCUMetrics]:
        """
        Runs a Python invocation under Nsight Compute and extracts structured metrics.
        """
        output_csv = os.path.join(self.output_dir, "ncu_profile_output.csv")
        metrics_arg = ",".join(self.target_metrics)

        cmd = [
            self.ncu_path,
            "--csv",
            f"--log-file={output_csv}",
            f"--metrics={metrics_arg}",
            f"--launch-count={max_kernels}",
            "--force-overwrite",
        ]

        if kernel_regex:
            cmd.append(f"--kernel-name={kernel_regex}")

        cmd.extend(python_command)

        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as err:
            raise RuntimeError(
                f"Nsight Compute failed execution:\nSTDOUT: {err.stdout}\nSTDERR: {err.stderr}"
            ) from err

        return self._parse_ncu_csv(output_csv)

    def _parse_ncu_csv(self, csv_filepath: str) -> List[NCUMetrics]:
        """Parses generated CSV file from Nsight Compute into dataclasses."""
        if not os.path.exists(csv_filepath):
            return []

        parsed_reports: Dict[str, Dict[str, float]] = {}

        with open(csv_filepath, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Filter header comment lines from NCU
        data_lines = [l for l in lines if not l.startswith('"==PROF=="')]
        if len(data_lines) < 2:
            return []

        headers = [h.strip('"') for h in data_lines[0].split('","')]
        
        try:
            kernel_idx = headers.index("Kernel Name")
            metric_idx = headers.index("Metric Name")
            value_idx = headers.index("Metric Value")
        except ValueError:
            return []

        for row in data_lines[1:]:
            parts = [p.strip('"') for p in row.split('","')]
            if len(parts) <= max(kernel_idx, metric_idx, value_idx):
                continue

            k_name = parts[kernel_idx]
            m_name = parts[metric_idx]
            raw_val = parts[value_idx].replace(",", "")

            try:
                val = float(raw_val)
            except ValueError:
                val = 0.0

            if k_name not in parsed_reports:
                parsed_reports[k_name] = {}
            parsed_reports[k_name][m_name] = val

        metrics_list = []
        for k_name, m_dict in parsed_reports.items():
            duration_ns = m_dict.get("gpu__time_duration.sum", 0.0)
            dram_bytes_sec = m_dict.get("dram__bytes.sum.per_second", 0.0)

            metrics_list.append(
                NCUMetrics(
                    kernel_name=k_name,
                    duration_ms=duration_ns / 1e6,
                    compute_sol_pct=m_dict.get(
                        "sm__throughput.avg.pct_of_peak_sustained_elapsed", 0.0
                    ),
                    memory_sol_pct=m_dict.get(
                        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", 0.0
                    ),
                    sm_occupancy_pct=m_dict.get(
                        "sm__warps_active.avg.pct_of_peak_sustained_active", 0.0
                    ),
                    dram_throughput_gbs=dram_bytes_sec / 1e9,
                    l1_cache_hit_rate=m_dict.get(
                        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct", 0.0
                    ),
                    l2_cache_hit_rate=m_dict.get(
                        "lts__t_sectors_op_read_hit_rate.pct", 0.0
                    ),
                    tensor_core_utilization=m_dict.get(
                        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active", 0.0
                    ),
                    raw_metrics=m_dict,
                )
            )

        return metrics_list


class CUDAEventProfiler:
    """
    Microsecond-accurate CUDA Event Timer with VRAM tracking & statistical breakdown.
    """

    def __init__(self, label: str, warmup_iters: int = 5, profile_iters: int = 20):
        self.label = label
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters

    def benchmark(self, fn, *args, **kwargs) -> CUDATimerReport:
        """Runs warmup and profiling cycles around callable `fn`."""
        assert torch.cuda.is_available(), "CUDA runtime required for CUDAEventProfiler."
        
        # Reset Peak VRAM Statistics
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Warmup loop
        for _ in range(self.warmup_iters):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.profile_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.profile_iters)]

        # Profile loop
        for i in range(self.profile_iters):
            start_events[i].record()
            fn(*args, **kwargs)
            end_events[i].record()

        torch.cuda.synchronize()

        timings = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]  # in ms
        timings_tensor = torch.tensor(timings, dtype=torch.float32)

        peak_mem_bytes = torch.cuda.max_memory_allocated()
        allocated_mem_bytes = torch.cuda.memory_allocated()

        return CUDATimerReport(
            label=self.label,
            mean_time_ms=float(torch.mean(timings_tensor).item()),
            std_time_ms=float(torch.std(timings_tensor).item()),
            min_time_ms=float(torch.min(timings_tensor).item()),
            max_time_ms=float(torch.max(timings_tensor).item()),
            peak_memory_mb=peak_mem_bytes / (1024.0 * 1024.0),
            allocated_memory_mb=allocated_mem_bytes / (1024.0 * 1024.0),
        )


@contextmanager
def nvtx_range(name: str) -> Generator[None, None, None]:
    """Context manager for adding NVTX ranges visible in Nsight Systems timelines."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
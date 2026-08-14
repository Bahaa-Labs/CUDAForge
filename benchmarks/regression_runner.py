from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

from benchmarks.suite.bench_flash_attn import run_flash_attn_benchmark
from benchmarks.suite.bench_paged_kv import run_paged_kv_benchmark
from benchmarks.suite.bench_continuous_batch import run_continuous_batch_benchmark


class RegressionRunner:
    """Detects performance degradation in latency, throughput, or kernel execution."""

    def __init__(self, baseline_path: Path, threshold_pct: float = 10.0) -> None:
        self.baseline_path = baseline_path
        self.threshold_pct = threshold_pct
        
        if not self.baseline_path.exists():
            raise FileNotFoundError(f"Baseline metrics missing at: {self.baseline_path}")

        with open(self.baseline_path, "r", encoding="utf-8") as f:
            self.baselines: Dict[str, Dict[str, float]] = json.load(f)

    def verify_metric(self, name: str, metric_key: str, actual: float, lower_is_better: bool = True) -> bool:
        """Compares current run metric against target baseline threshold."""
        if name not in self.baselines or metric_key not in self.baselines[name]:
            print(f"[Regression Skip] Baseline key '{name}.{metric_key}' not found.")
            return True

        target = self.baselines[name][metric_key]
        delta_pct = ((actual - target) / target) * 100.0

        if lower_is_better:
            # Degraded if latency increased beyond threshold %
            passed = delta_pct <= self.threshold_pct
            status = "PASSED" if passed else "FAILED (LATENCY REGRESSION)"
        else:
            # Degraded if throughput dropped beyond threshold %
            passed = delta_pct >= -self.threshold_pct
            status = "PASSED" if passed else "FAILED (THROUGHPUT REGRESSION)"

        print(
            f"[{status}] {name}.{metric_key:<25} | Target: {target:<8.2f} | Actual: {actual:<8.2f} | Change: {delta_pct:+.2f}%"
        )
        return passed

    def run_all(self) -> bool:
        """Runs entire benchmarking suite and evaluates regression passes."""
        all_passed = True

        # 1. FlashAttention
        flash_res = run_flash_attn_benchmark()
        pass_fa = self.verify_metric("flash_attention_v2", "flash_attn_p50_ms", flash_res["flash_attn_p50_ms"], lower_is_better=True)
        all_passed = all_passed and pass_fa

        # 2. Paged KV Cache
        paged_res = run_paged_kv_benchmark()
        pass_kv = self.verify_metric("paged_kv_cache_manager", "alloc_latency_p50_us", paged_res["alloc_latency_p50_us"], lower_is_better=True)
        all_passed = all_passed and pass_kv

        # 3. Continuous Batching
        cb_res = run_continuous_batch_benchmark()
        pass_cb = self.verify_metric("continuous_batching_scheduler", "throughput_tokens_per_sec", cb_res["throughput_tokens_per_sec"], lower_is_better=False)
        all_passed = all_passed and pass_cb

        if all_passed:
            print("SUCCESS: All performance metrics passed baseline regression checks!")
        else:
            print("FAILURE: Performance regression detected! CI check failed.")

        return all_passed


if __name__ == "__main__":
    base_file = Path(__file__).parent / "baselines" / "baseline_metrics.json"
    runner = RegressionRunner(baseline_path=base_file, threshold_pct=10.0)
    success = runner.run_all()
    sys.exit(0 if success else 1)
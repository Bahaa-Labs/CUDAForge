"""
Computes non-parametric latency distributions, bootstrapped confidence intervals,
interquartile ranges (IQR), and variance statistics for high-throughput LLM benchmarks.
"""

from dataclasses import asdict, dataclass, field
import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class ConfidenceInterval:
    lower: float
    upper: float
    confidence_level: float = 0.95


@dataclass
class LatencyDistributionSummary:
    count: int
    mean: float
    std_dev: float
    variance: float
    median_p50: float
    p90: float
    p95: float
    p99: float
    p99_9: float
    iqr: float
    mad: float  # Median Absolute Deviation
    mean_ci: ConfidenceInterval
    p95_ci: ConfidenceInterval
    p99_ci: ConfidenceInterval

    def to_dict(self) -> Dict[str, Union[int, float, Dict[str, float]]]:
        res = asdict(self)
        return res


class StatisticalAnalyzer:
    """
    Statistical engine for calculating non-parametric latency tails and
    bootstrapped confidence metrics.
    """

    @staticmethod
    def compute_distribution(
        samples: Union[List[float], np.ndarray],
        num_bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42,
    ) -> LatencyDistributionSummary:
        """
        Computes complete non-parametric latency distribution analysis.
        
        Args:
            samples: Raw latency observations in milliseconds.
            num_bootstrap_samples: Number of bootstrap resamples for CI estimation.
            confidence_level: Desired statistical confidence level (e.g., 0.95 for 95%).
            random_seed: Seed for reproducible bootstrap sampling.
        """
        data = np.asarray(samples, dtype=np.float64)
        data = data[~np.isnan(data)]  # Filter NaNs

        if len(data) == 0:
            raise ValueError("Cannot compute statistical distribution on empty or NaN dataset.")

        count = len(data)
        mean_val = float(np.mean(data))
        std_val = float(np.std(data, ddof=1)) if count > 1 else 0.0
        var_val = float(np.var(data, ddof=1)) if count > 1 else 0.0

        p25, p50, p75, p90, p95, p99, p99_9 = np.percentile(
            data, [25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9]
        )

        iqr = float(p75 - p25)
        mad = float(np.median(np.abs(data - p50)))

        # Compute Bootstrapped CIs
        mean_ci = StatisticalAnalyzer._bootstrap_ci(
            data, np.mean, num_bootstrap_samples, confidence_level, random_seed
        )
        p95_ci = StatisticalAnalyzer._bootstrap_ci(
            data, lambda x: np.percentile(x, 95.0), num_bootstrap_samples, confidence_level, random_seed
        )
        p99_ci = StatisticalAnalyzer._bootstrap_ci(
            data, lambda x: np.percentile(x, 99.0), num_bootstrap_samples, confidence_level, random_seed
        )

        return LatencyDistributionSummary(
            count=count,
            mean=mean_val,
            std_dev=std_val,
            variance=var_val,
            median_p50=float(p50),
            p90=float(p90),
            p95=float(p95),
            p99=float(p99),
            p99_9=float(p99_9),
            iqr=iqr,
            mad=mad,
            mean_ci=mean_ci,
            p95_ci=p95_ci,
            p99_ci=p99_ci,
        )

    @staticmethod
    def _bootstrap_ci(
        data: np.ndarray,
        stat_fn: callable,
        num_bootstrap: int,
        confidence_level: float,
        seed: int,
    ) -> ConfidenceInterval:
        """Computes non-parametric percentile bootstrap confidence interval."""
        if len(data) < 2 or num_bootstrap <= 0:
            val = float(stat_fn(data))
            return ConfidenceInterval(lower=val, upper=val, confidence_level=confidence_level)

        rng = np.random.default_rng(seed)
        n = len(data)
        bootstrap_stats = np.empty(num_bootstrap, dtype=np.float64)

        for i in range(num_bootstrap):
            resample = rng.choice(data, size=n, replace=True)
            bootstrap_stats[i] = stat_fn(resample)

        alpha = (1.0 - confidence_level) / 2.0
        lower_pct = alpha * 100.0
        upper_pct = (1.0 - alpha) * 100.0

        lower_bound = float(np.percentile(bootstrap_stats, lower_pct))
        upper_bound = float(np.percentile(bootstrap_stats, upper_pct))

        return ConfidenceInterval(
            lower=lower_bound, upper=upper_bound, confidence_level=confidence_level
        )
"""
Calculates Pareto-optimal frontiers and non-dominated decision surfaces across
multi-objective benchmark metrics (e.g., Throughput vs. Latency vs. Accuracy).
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class ParetoPoint:
    point_id: str
    metrics: Dict[str, float]
    metadata: Dict[str, Union[str, int, float]]
    is_pareto_optimal: bool = False
    distance_to_frontier: float = 0.0


class ParetoFrontierCalculator:
    """
    Multi-objective Pareto optimality solver for empirical system benchmarks.
    """

    @staticmethod
    def compute_pareto_frontier(
        points: List[ParetoPoint],
        objectives: Dict[str, str],  # Key: Metric Name, Value: "maximize" or "minimize"
    ) -> List[ParetoPoint]:
        """
        Computes the Pareto-optimal frontier set from benchmark observation points.

        Args:
            points: List of ParetoPoint evaluation entries.
            objectives: Dictionary specifying optimization direction per metric name.
                        Example: {"throughput_tps": "maximize", "p99_latency_ms": "minimize"}
        """
        if not points:
            return []

        metric_names = list(objectives.keys())
        num_points = len(points)
        num_metrics = len(metric_names)

        # Build normalized criteria matrix where higher is always better
        matrix = np.zeros((num_points, num_metrics), dtype=np.float64)

        for i, pt in enumerate(points):
            for j, metric in enumerate(metric_names):
                val = pt.metrics.get(metric, 0.0)
                direction = objectives[metric].lower()
                if direction == "maximize":
                    matrix[i, j] = val
                elif direction == "minimize":
                    matrix[i, j] = -val  # Invert so larger value is strictly better
                else:
                    raise ValueError(
                        f"Invalid objective direction '{direction}'. Must be 'maximize' or 'minimize'."
                    )

        # Find non-dominated points
        is_optimal = np.ones(num_points, dtype=bool)
        for i in range(num_points):
            if not is_optimal[i]:
                continue
            # Point j dominates point i if all metrics of j >= i and at least one metric of j > i
            for j in range(num_points):
                if i != j and is_optimal[j]:
                    if np.all(matrix[j] >= matrix[i]) and np.any(matrix[j] > matrix[i]):
                        is_optimal[i] = False
                        break

        # Calculate normalized Euclidean distance to closest Pareto frontier point
        frontier_matrix = matrix[is_optimal]

        # Normalize matrix for fair distance calculation
        min_vals = np.min(matrix, axis=0)
        max_vals = np.max(matrix, axis=0)
        range_vals = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)

        norm_matrix = (matrix - min_vals) / range_vals
        norm_frontier = (frontier_matrix - min_vals) / range_vals

        processed_points: List[ParetoPoint] = []
        for i, pt in enumerate(points):
            pt_is_opt = bool(is_optimal[i])

            if pt_is_opt:
                dist = 0.0
            else:
                # Euclidean distance to nearest frontier point
                diffs = norm_frontier - norm_matrix[i]
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                dist = float(np.min(dists))

            processed_pt = ParetoPoint(
                point_id=pt.point_id,
                metrics=pt.metrics,
                metadata=pt.metadata,
                is_pareto_optimal=pt_is_opt,
                distance_to_frontier=dist,
            )
            processed_points.append(processed_pt)

        return processed_points

"""Process behaviour classification interfaces and implementations.

Follows Workdoc Section 6.3:
Rule-based classifier:
- Needs minimum of 3 observed bursts, otherwise returns "unknown".
- Rules:
  if mean_burst < 5 and io_ratio > 0.6:             "interactive"
  elif cv_burst > 1.0:                              "bursty"
  elif io_ratio > 0.5:                              "io_bound"
  elif io_ratio < 0.1 and full_quantum_frac > 0.8:  "cpu_bound"
  else: fallback centroid / default class
- Charges classification cost (cost_classify = 0.01 ms)
"""

from __future__ import annotations

from typing import Any
import numpy as np

from sabre.intel.features import Features


class BehaviourClassifier:
    """Classifies process behaviour into 4 canonical classes:
    'interactive', 'io_bound', 'cpu_bound', 'bursty', or 'unknown'.
    """

    def __init__(
        self,
        min_bursts: int = 3,
        cost_ms: float = 0.01,
        centroids: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.min_bursts = min_bursts
        self.cost_ms = float(cost_ms)
        self.centroids = centroids
        self._accrued_cost_ms: float = 0.0

        # Per-pid burst observation count tracking
        self.observed_burst_counts: dict[int, int] = {}

    def record_burst(self, pid: int) -> None:
        """Increment count of observed bursts for process `pid`."""
        self.observed_burst_counts[pid] = self.observed_burst_counts.get(pid, 0) + 1

    def classify(self, f: Features, pid: int | None = None, observed_bursts: int | None = None) -> str:
        """Classify a process given its extracted features.

        Returns 'unknown' if fewer than min_bursts observed.
        """
        self._accrued_cost_ms += self.cost_ms

        bursts_count = observed_bursts
        if bursts_count is None and pid is not None:
            bursts_count = self.observed_burst_counts.get(pid, 0)

        # If burst count is tracked and < min_bursts, return 'unknown'
        if bursts_count is not None and bursts_count < self.min_bursts:
            return "unknown"

        # Rule-based tree
        if f.mean_burst < 5.0 and f.io_ratio > 0.6:
            return "interactive"
        elif f.cv_burst > 1.0:
            return "bursty"
        elif f.io_ratio > 0.5:
            return "io_bound"
        elif f.io_ratio < 0.1 and f.full_quantum_frac > 0.8:
            return "cpu_bound"

        # Centroid distance fallback if centroids provided
        if self.centroids:
            best_cls = "unknown"
            min_dist = float("inf")
            for cls_name, c_dict in self.centroids.items():
                # Normalized distance on [mean_burst, io_ratio]
                dist = 0.0
                if "mean_burst" in c_dict:
                    dist += ((f.mean_burst - c_dict["mean_burst"]) / max(1.0, c_dict["mean_burst"])) ** 2
                if "io_ratio" in c_dict:
                    dist += (f.io_ratio - c_dict["io_ratio"]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_cls = cls_name
            return best_cls

        # Default fallback
        if f.io_ratio > 0.4:
            return "io_bound"
        elif f.mean_burst > 50.0:
            return "cpu_bound"
        return "unknown"

    def overhead_ms(self) -> float:
        """Return monitoring cost accrued since last call, then reset."""
        cost = self._accrued_cost_ms
        self._accrued_cost_ms = 0.0
        return cost

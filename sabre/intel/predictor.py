"""Burst prediction interfaces and implementations.

Follows Workdoc Section 3 interface contracts exactly:
class BurstPredictor: predict(pid)->float ; update(pid, observed_burst)
"""

from __future__ import annotations


class BurstPredictor:
    """Predicts CPU burst length for processes."""

    def predict(self, pid: int) -> float:
        """Predict the next CPU burst length for process `pid`."""
        raise NotImplementedError

    def update(self, pid: int, observed_burst: float) -> None:
        """Update prediction state for process `pid` with newly observed burst."""
        raise NotImplementedError

"""Burst prediction interfaces and implementations.

Follows Workdoc Section 6.2 and Section 3:
- EMA predictor: τ_{n+1} = α · t_n + (1 - α) · τ_n, default α = 0.5
- Initial τ given by class prior (default: interactive 2, io 8, cpu 150, bursty 10)
- Charges prediction cost (cost_ema = 0.01 ms)
"""

from __future__ import annotations

from typing import Any


DEFAULT_PRIORS: dict[str, float] = {
    "interactive": 2.0,
    "io_bound": 8.0,
    "cpu_bound": 150.0,
    "bursty": 10.0,
    "unknown": 10.0,
}


class BurstPredictor:
    """Exponential Moving Average (EMA) and baseline burst predictor."""

    def __init__(
        self,
        alpha: float = 0.5,
        priors: dict[str, float] | None = None,
        cost_ms: float = 0.01,
        default_prior: float = 10.0,
    ) -> None:
        self.alpha = float(alpha)
        self.priors = dict(priors) if priors is not None else dict(DEFAULT_PRIORS)
        self.cost_ms = float(cost_ms)
        self.default_prior = float(default_prior)

        # per-process state: pid -> predicted burst
        self.estimates: dict[int, float] = {}
        self._accrued_cost_ms: float = 0.0

    def init_process(self, pid: int, kind_or_class: str = "unknown") -> None:
        """Initialize prediction estimate using class prior."""
        prior = self.priors.get(kind_or_class, self.default_prior)
        self.estimates[pid] = float(prior)

    def predict(self, pid: int) -> float:
        """Predict the next CPU burst length for process `pid`."""
        self._accrued_cost_ms += self.cost_ms
        if pid not in self.estimates:
            self.init_process(pid)
        return self.estimates[pid]

    def update(self, pid: int, observed_burst: float) -> None:
        """Update prediction state for process `pid` with newly observed burst."""
        if pid not in self.estimates:
            self.init_process(pid)
        old_est = self.estimates[pid]
        # τ_{n+1} = α · t_n + (1 - α) · τ_n
        new_est = self.alpha * float(observed_burst) + (1.0 - self.alpha) * old_est
        self.estimates[pid] = new_est

    def overhead_ms(self) -> float:
        """Return monitoring cost accrued since last call, then reset."""
        cost = self._accrued_cost_ms
        self._accrued_cost_ms = 0.0
        return cost

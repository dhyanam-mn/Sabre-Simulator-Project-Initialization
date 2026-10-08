"""Scheduler base abstract class.

Follows Workdoc Section 3 interface contracts.
"""

from __future__ import annotations

from typing import Any
from sabre.core.process import Process


class Scheduler:
    """Base class for all discrete-event schedulers."""

    name: str = "base"

    def on_arrival(self, p: Process, now: float) -> None:
        """Handle new process arrival."""
        pass

    def on_ready(self, p: Process, now: float) -> None:
        """Handle process entering ready queue after IO or preemption."""
        pass

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        """Handle completion of a time slice."""
        pass

    def on_exit(self, p: Process, now: float) -> None:
        """Handle process termination."""
        pass

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        """Pick next process to run and quantum length (process, quantum)."""
        raise NotImplementedError

    def overhead_ms(self) -> float:
        """Monitoring cost accrued since last call, then reset."""
        return 0.0

    def flagged(self) -> dict[int, float]:
        """Map pid -> first time level >= 2 (for detection metrics)."""
        return {}

    def decision_log(self) -> list[dict[str, Any]]:
        """Optional explainability log of decisions."""
        return []

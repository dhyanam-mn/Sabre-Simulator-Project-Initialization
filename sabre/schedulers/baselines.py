"""Baseline schedulers: FCFS and Round Robin (RR).

Follows Workdoc Section 5.4.
"""

from __future__ import annotations

from collections import deque
from typing import Any
from sabre.core.process import Process
from sabre.schedulers.base import Scheduler


class FCFS(Scheduler):
    """First-Come, First-Served non-preemptive scheduler."""

    name = "FCFS"

    def __init__(self) -> None:
        self.queue: deque[Process] = deque()

    def on_arrival(self, p: Process, now: float) -> None:
        self.queue.append(p)

    def on_ready(self, p: Process, now: float) -> None:
        self.queue.append(p)

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        # If preemption happened (should not in pure FCFS unless quantum was limited), requeue
        if not burst_done and p.remaining_burst > 0:
            self.queue.appendleft(p)

    def on_exit(self, p: Process, now: float) -> None:
        pass

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        if not self.queue:
            return None
        p = self.queue.popleft()
        # Non-preemptive: grant entire remaining burst (or huge quantum)
        quantum = float("inf")
        return p, quantum


class RoundRobin(Scheduler):
    """Round Robin (RR) preemptive scheduler with configurable time quantum."""

    name = "RR"

    def __init__(self, quantum_ms: float = 10.0) -> None:
        self.quantum_ms = float(quantum_ms)
        self.queue: deque[Process] = deque()

    def on_arrival(self, p: Process, now: float) -> None:
        self.queue.append(p)

    def on_ready(self, p: Process, now: float) -> None:
        self.queue.append(p)

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        if not burst_done and p.remaining_burst > 0:
            self.queue.append(p)

    def on_exit(self, p: Process, now: float) -> None:
        pass

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        if not self.queue:
            return None
        p = self.queue.popleft()
        return p, self.quantum_ms

"""Process and ProcState definitions.

Follows Workdoc Section 3 interface contracts exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProcState(Enum):
    """Process execution states in discrete-event simulation."""

    NEW = 0
    READY = 1
    RUNNING = 2
    WAITING = 3
    DONE = 4


@dataclass
class Process:
    """Represents a simulated process."""

    pid: int
    ppid: int | None
    arrival: float
    kind: str  # benign kinds: "interactive","io_bound","cpu_bound","bursty"
    # malicious: "forkbomb","hog","lowslow","adaptive","forkchild"
    label: int  # ground truth 0 benign / 1 malicious (forkchild = 1)
    total_cpu: float  # total CPU demand; inf allowed for hog
    state: ProcState = ProcState.NEW
    remaining_burst: float = 0.0  # remaining time in current CPU burst
    cpu_used: float = 0.0
    first_run: float | None = None
    finish: float | None = None
    wait_time: float = 0.0
    last_ready_at: float = 0.0

    # hooks: engine calls these to get the next phase
    def next_cpu_burst(self, now: float, rng: Any) -> float:
        """Return length of next CPU burst in ms."""
        raise NotImplementedError

    def next_io_wait(self, now: float, rng: Any) -> float:
        """Return length of next I/O wait in ms."""
        raise NotImplementedError

    def children_to_spawn(self, cpu_delta: float, now: float, rng: Any) -> list[Process]:
        """Return new children processes spawned during this slice."""
        raise NotImplementedError

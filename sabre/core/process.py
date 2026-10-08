"""Process and ProcState definitions.

Follows Workdoc Section 3 interface contracts and engine requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


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

    # Custom burst/io/spawn generators (optional, for custom or test workloads)
    burst_generator: Callable[[float, Any], float] | None = None
    io_generator: Callable[[float, Any], float] | None = None
    spawn_generator: Callable[[float, float, Any], list[Process]] | None = None

    # Tracking per-burst interactive response samples
    # Timestamp when process entered ready state from I/O completion
    last_io_complete_at: float | None = None
    response_samples: list[float] = field(default_factory=list)

    # hooks: engine calls these to get the next phase
    def next_cpu_burst(self, now: float, rng: Any) -> float:
        """Return length of next CPU burst in ms."""
        if self.burst_generator is not None:
            return float(self.burst_generator(now, rng))
        # Default: if no generator, remaining CPU work needed
        rem = self.total_cpu - self.cpu_used
        return float(rem) if rem > 0 else 0.0

    def next_io_wait(self, now: float, rng: Any) -> float:
        """Return length of next I/O wait in ms."""
        if self.io_generator is not None:
            return float(self.io_generator(now, rng))
        return 0.0

    def children_to_spawn(self, cpu_delta: float, now: float, rng: Any) -> list[Process]:
        """Return new children processes spawned during this slice."""
        if self.spawn_generator is not None:
            return self.spawn_generator(cpu_delta, now, rng)
        return []

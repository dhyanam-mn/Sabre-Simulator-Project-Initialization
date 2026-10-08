"""Feature representations and FeatureExtractor.

Follows Workdoc Section 6.1 and Section 3:
- Ring buffers of last W=8 bursts, IO waits, and slices per PID
- Sliding window T_share=500 ms for cpu_share
- F1 additions: burst_age_ms (CPU run since last IO/burst end) and slices_since_io
- fork_rate (children per 100 ms over last 500 ms) and n_descendants (via parent links)
- Charges feature cost (cost_feature = 0.02 ms)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class Features:
    """Statistical features extracted from process behavior."""

    mean_burst: float
    cv_burst: float
    io_ratio: float
    cpu_share: float  # share of wall time on CPU over last T_share ms
    full_quantum_frac: float  # fraction of recent slices that used the whole quantum
    fork_rate: float  # children per 100 ms
    n_descendants: int
    burst_age_ms: float  # F1: CPU run time since last IO / burst end
    slices_since_io: int  # F1: number of execution slices without IO


@dataclass
class _ProcRecord:
    pid: int
    ppid: int | None
    bursts: deque[float]
    io_waits: deque[float]
    # Slices history: list of (time, ran_ms, was_full_quantum)
    slices: deque[tuple[float, float, bool]]
    # Children spawn times: (timestamp, count)
    spawn_events: deque[tuple[float, int]]
    # F1 tracking
    burst_age_ms: float = 0.0
    slices_since_io: int = 0
    # Burst accumulation
    current_burst_acc: float = 0.0


class FeatureExtractor:
    """Extracts behavioral and risk features per process."""

    def __init__(
        self,
        W: int = 8,
        T_share_ms: float = 500.0,
        cost_feature_ms: float = 0.02,
    ) -> None:
        self.W = W
        self.T_share_ms = float(T_share_ms)
        self.cost_feature_ms = float(cost_feature_ms)

        # per-process state: pid -> _ProcRecord
        self.records: dict[int, _ProcRecord] = {}
        # Parent-child tracking: ppid -> set of child pids
        self.children_map: dict[int, set[int]] = {}
        # Parent pointer: child_pid -> ppid
        self.parent_map: dict[int, int | None] = {}

        # Accrued overhead
        self._accrued_cost_ms: float = 0.0

    def register_process(self, pid: int, ppid: int | None) -> None:
        """Register a new process for feature tracking."""
        if pid not in self.records:
            self.records[pid] = _ProcRecord(
                pid=pid,
                ppid=ppid,
                bursts=deque(maxlen=self.W),
                io_waits=deque(maxlen=self.W),
                slices=deque(),
                spawn_events=deque(),
            )
            self.parent_map[pid] = ppid
            if ppid is not None:
                if ppid not in self.children_map:
                    self.children_map[ppid] = set()
                self.children_map[ppid].add(pid)

    def on_spawn(self, pid: int, ppid: int | None, count: int, now: float) -> None:
        """Record process spawning event."""
        if pid not in self.records:
            self.register_process(pid, ppid)
        rec = self.records[pid]
        rec.spawn_events.append((now, count))

    def on_io(self, pid: int, io_wait_ms: float) -> None:
        """Record I/O wait occurrence."""
        if pid not in self.records:
            self.register_process(pid, None)
        rec = self.records[pid]
        rec.io_waits.append(io_wait_ms)
        # IO completes burst: reset burst_age_ms and slices_since_io
        rec.burst_age_ms = 0.0
        rec.slices_since_io = 0

    def on_slice_end(
        self,
        pid: int,
        ran: float,
        burst_done: bool,
        quantum: float,
        now: float,
        ppid: int | None = None,
    ) -> None:
        """Update metrics upon slice completion."""
        if pid not in self.records:
            self.register_process(pid, ppid)

        self._accrued_cost_ms += self.cost_feature_ms
        rec = self.records[pid]

        was_full = (ran >= quantum - 1e-9) and (not burst_done)
        rec.slices.append((now, ran, was_full))

        # Accumulate burst
        rec.current_burst_acc += ran
        rec.burst_age_ms += ran
        rec.slices_since_io += 1

        if burst_done:
            rec.bursts.append(rec.current_burst_acc)
            rec.current_burst_acc = 0.0
            # Burst ended (e.g. going to IO or done)
            # If going to IO, burst_age_ms will be reset in on_io, but even if immediately next burst:
            rec.burst_age_ms = 0.0

        # Purge slices older than T_share_ms
        cutoff = now - self.T_share_ms
        while rec.slices and rec.slices[0][0] < cutoff:
            rec.slices.popleft()

        # Purge spawn events older than 500 ms
        while rec.spawn_events and rec.spawn_events[0][0] < cutoff:
            rec.spawn_events.popleft()

    def count_descendants(self, root_pid: int) -> int:
        """Count total alive/registered descendants under root_pid."""
        visited: set[int] = set()
        stack = list(self.children_map.get(root_pid, []))
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                stack.extend(self.children_map.get(curr, []))
        return len(visited)

    def extract(self, pid: int, now: float) -> Features:
        """Compute Features dataclass for process pid at timestamp now."""
        if pid not in self.records:
            self.register_process(pid, None)
        rec = self.records[pid]

        # 1. Bursts statistics
        if rec.bursts:
            b_arr = np.array(list(rec.bursts), dtype=float)
            m_burst = float(np.mean(b_arr))
            std_burst = float(np.std(b_arr))
            cv_burst = float(std_burst / m_burst) if m_burst > 1e-9 else 0.0
        else:
            m_burst = rec.current_burst_acc
            cv_burst = 0.0

        # 2. IO ratio: io / (io + cpu) over the window
        sum_io = float(sum(rec.io_waits)) if rec.io_waits else 0.0
        sum_cpu = float(sum(rec.bursts)) if rec.bursts else rec.current_burst_acc
        total_time = sum_io + sum_cpu
        io_ratio = float(sum_io / total_time) if total_time > 1e-9 else 0.0

        # 3. CPU share over T_share_ms
        cutoff = now - self.T_share_ms
        cpu_in_window = sum(ran for (t, ran, _) in rec.slices if t >= cutoff)
        cpu_share = float(min(1.0, max(0.0, cpu_in_window / self.T_share_ms)))

        # 4. Full quantum fraction over recent slices in window (or last W slices)
        recent_slices = [was_full for (t, _, was_full) in rec.slices if t >= cutoff]
        if recent_slices:
            full_quantum_frac = float(sum(recent_slices) / len(recent_slices))
        else:
            full_quantum_frac = 0.0

        # 5. Fork rate (children per 100 ms over last 500 ms)
        cutoff_fork = now - 500.0
        total_spawned = sum(cnt for (t, cnt) in rec.spawn_events if t >= cutoff_fork)
        # fork_rate = total_spawned per 100 ms = (total_spawned / 500.0) * 100.0
        fork_rate = float(total_spawned / 5.0)

        # 6. Descendants count
        n_desc = self.count_descendants(pid)

        return Features(
            mean_burst=m_burst,
            cv_burst=cv_burst,
            io_ratio=io_ratio,
            cpu_share=cpu_share,
            full_quantum_frac=full_quantum_frac,
            fork_rate=fork_rate,
            n_descendants=n_desc,
            burst_age_ms=rec.burst_age_ms,
            slices_since_io=rec.slices_since_io,
        )

    def overhead_ms(self) -> float:
        """Return accrued monitoring cost since last call, then reset."""
        cost = self._accrued_cost_ms
        self._accrued_cost_ms = 0.0
        return cost

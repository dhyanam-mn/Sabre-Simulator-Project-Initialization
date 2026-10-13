"""Baseline schedulers for SABRE comparisons.

Follows Workdoc Section 5.4 and Research Report Section A4:
- FCFS: First-Come, First-Served non-preemptive
- SJF_Oracle: Shortest Job First non-preemptive with oracle next-burst knowledge
- SJF_EMA: Shortest Job First non-preemptive with EMA-predicted next burst
- RoundRobin: Standard Round Robin with configurable quantum
- PriorityScheduler: Static priority by kind (interactive > bursty > io_bound > cpu_bound)
- MLFQ: Multi-Level Feedback Queue (3 levels, 10ms / 20ms / 40ms) with periodic priority boost
- RR_StaticThreshold: RR plus sliding 500ms CPU-share monitor that throttles if share > tau_static
- CFS_Like: Virtual runtime-based fair share scheduler (Linux CFS analogue)
- RR_Cgroup: RR plus hard CPU bandwidth cap per process and max-children limit

Note on queue management:
Simulator engine invokes `on_ready(p, now)` whenever a process becomes ready
(upon arrival, after IO completion, or upon preemption at slice end).
Schedulers enqueue in `on_ready(p, now)` and `on_arrival(p, now)`, while
`on_slice_end` updates internal accounting (e.g. predictor, MLFQ levels, monitors).
"""

from __future__ import annotations

from collections import deque
import heapq
from typing import Any
import numpy as np

from sabre.core.process import Process
from sabre.schedulers.base import Scheduler
from sabre.intel.predictor import BurstPredictor


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
        pass

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        if not self.queue:
            return None
        p = self.queue.popleft()
        return p, float("inf")


class SJF_Oracle(Scheduler):
    """Shortest Job First non-preemptive with oracle next burst length."""

    name = "SJF-oracle"

    def __init__(self) -> None:
        self.ready_list: list[Process] = []

    def on_arrival(self, p: Process, now: float) -> None:
        self.ready_list.append(p)

    def on_ready(self, p: Process, now: float) -> None:
        self.ready_list.append(p)

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        pass

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        if not self.ready_list:
            return None

        def get_oracle_burst(p: Process) -> float:
            if p.remaining_burst > 0:
                return p.remaining_burst
            if hasattr(p, "_raw_burst_pool"):
                pool = getattr(p, "_raw_burst_pool")
                return pool[0] if pool else 10.0
            return 10.0

        best_idx = 0
        best_burst = get_oracle_burst(self.ready_list[0])
        for i in range(1, len(self.ready_list)):
            b = get_oracle_burst(self.ready_list[i])
            if b < best_burst:
                best_burst = b
                best_idx = i

        p = self.ready_list.pop(best_idx)
        return p, float("inf")


class SJF_EMA(Scheduler):
    """Shortest Job First non-preemptive using EMA burst predictor."""

    name = "SJF-EMA"

    def __init__(self, alpha: float = 0.5) -> None:
        self.ready_list: list[Process] = []
        self.predictor = BurstPredictor(alpha=alpha)

    def on_arrival(self, p: Process, now: float) -> None:
        self.predictor.init_process(p.pid, p.kind)
        self.ready_list.append(p)

    def on_ready(self, p: Process, now: float) -> None:
        self.ready_list.append(p)

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        if burst_done:
            self.predictor.update(p.pid, ran)

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        if not self.ready_list:
            return None
        best_idx = 0
        best_pred = self.predictor.predict(self.ready_list[0].pid)
        for i in range(1, len(self.ready_list)):
            pred = self.predictor.predict(self.ready_list[i].pid)
            if pred < best_pred:
                best_pred = pred
                best_idx = i
        p = self.ready_list.pop(best_idx)
        return p, float("inf")

    def overhead_ms(self) -> float:
        return self.predictor.overhead_ms()


class RoundRobin(Scheduler):
    """Round Robin (RR) preemptive scheduler."""

    name = "RR"

    def __init__(self, quantum_ms: float = 10.0) -> None:
        self.quantum_ms = float(quantum_ms)
        self.queue: deque[Process] = deque()

    def on_arrival(self, p: Process, now: float) -> None:
        self.queue.append(p)

    def on_ready(self, p: Process, now: float) -> None:
        self.queue.append(p)

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        pass

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        if not self.queue:
            return None
        p = self.queue.popleft()
        return p, self.quantum_ms


class PriorityScheduler(Scheduler):
    """Static priority scheduler by kind (interactive > bursty > io_bound > cpu_bound)."""

    name = "Priority"

    PRIORITY_MAP = {
        "interactive": 0,
        "bursty": 1,
        "io_bound": 2,
        "cpu_bound": 3,
        "unknown": 2,
    }

    def __init__(self, quantum_ms: float = 10.0) -> None:
        self.quantum_ms = float(quantum_ms)
        self.queues: list[deque[Process]] = [deque() for _ in range(4)]

    def _get_prio(self, p: Process) -> int:
        return self.PRIORITY_MAP.get(p.kind, 2)

    def on_arrival(self, p: Process, now: float) -> None:
        self.queues[self._get_prio(p)].append(p)

    def on_ready(self, p: Process, now: float) -> None:
        self.queues[self._get_prio(p)].append(p)

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        pass

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        for q in self.queues:
            if q:
                p = q.popleft()
                return p, self.quantum_ms
        return None


class MLFQ(Scheduler):
    """Multi-Level Feedback Queue (3 levels) with periodic priority boost."""

    name = "MLFQ"

    def __init__(
        self,
        quanta: tuple[float, float, float] = (10.0, 20.0, 40.0),
        boost_interval_ms: float = 200.0,
    ) -> None:
        self.quanta = tuple(quanta)
        self.boost_interval_ms = float(boost_interval_ms)
        self.queues: list[deque[Process]] = [deque(), deque(), deque()]
        self.proc_level: dict[int, int] = {}
        self.last_boost: float = 0.0

    def _boost_all(self, now: float) -> None:
        """Boost all processes back to top queue (Q0)."""
        combined: list[Process] = []
        for q in self.queues:
            while q:
                combined.append(q.popleft())
        for p in combined:
            self.proc_level[p.pid] = 0
            self.queues[0].append(p)
        self.last_boost = now

    def on_arrival(self, p: Process, now: float) -> None:
        self.proc_level[p.pid] = 0
        self.queues[0].append(p)

    def on_ready(self, p: Process, now: float) -> None:
        lvl = self.proc_level.get(p.pid, 0)
        self.queues[lvl].append(p)

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        lvl = self.proc_level.get(p.pid, 0)
        if not burst_done and p.remaining_burst > 0:
            # Demote if full quantum used without finishing burst
            if lvl < 2:
                self.proc_level[p.pid] = lvl + 1

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        if now - self.last_boost >= self.boost_interval_ms:
            self._boost_all(now)

        for lvl, q in enumerate(self.queues):
            if q:
                p = q.popleft()
                return p, self.quanta[lvl]
        return None


class RR_StaticThreshold(Scheduler):
    """Round Robin with a naive 500ms CPU share monitor that throttles processes."""

    name = "RR+StaticThreshold"

    def __init__(
        self,
        quantum_ms: float = 10.0,
        static_threshold: float = 0.60,
        window_ms: float = 500.0,
        throttled_quantum_ms: float = 2.0,
    ) -> None:
        self.quantum_ms = float(quantum_ms)
        self.static_threshold = float(static_threshold)
        self.window_ms = float(window_ms)
        self.throttled_quantum_ms = float(throttled_quantum_ms)

        self.queue: deque[Process] = deque()
        self.slice_history: dict[int, deque[tuple[float, float]]] = {}
        self._flagged: dict[int, float] = {}

    def _cpu_share(self, pid: int, now: float) -> float:
        history = self.slice_history.get(pid)
        if not history:
            return 0.0
        cutoff = now - self.window_ms
        while history and history[0][0] < cutoff:
            history.popleft()
        cpu_ran = sum(r for (t, r) in history if t >= cutoff)
        return float(min(1.0, max(0.0, cpu_ran / self.window_ms)))

    def on_arrival(self, p: Process, now: float) -> None:
        self.queue.append(p)
        if p.pid not in self.slice_history:
            self.slice_history[p.pid] = deque()

    def on_ready(self, p: Process, now: float) -> None:
        self.queue.append(p)

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        if p.pid not in self.slice_history:
            self.slice_history[p.pid] = deque()
        self.slice_history[p.pid].append((now, ran))

        share = self._cpu_share(p.pid, now)
        if share >= self.static_threshold:
            if p.pid not in self._flagged:
                self._flagged[p.pid] = now

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        if not self.queue:
            return None
        p = self.queue.popleft()
        share = self._cpu_share(p.pid, now)
        q = self.throttled_quantum_ms if share >= self.static_threshold else self.quantum_ms
        return p, q

    def flagged(self) -> dict[int, float]:
        return dict(self._flagged)


class CFS_Like(Scheduler):
    """Completely Fair Scheduler (CFS) analogue using virtual runtime (vruntime)."""

    name = "CFS-like"

    WEIGHT_MAP = {
        "interactive": 1280.0,
        "bursty": 1024.0,
        "io_bound": 1024.0,
        "cpu_bound": 820.0,
        "unknown": 1024.0,
    }

    def __init__(self, latency_target_ms: float = 10.0, min_granularity_ms: float = 2.0) -> None:
        self.latency_target_ms = float(latency_target_ms)
        self.min_granularity_ms = float(min_granularity_ms)
        self.heap: list[tuple[float, int, Process]] = []
        self.vruntime: dict[int, float] = {}
        self.min_vruntime: float = 0.0
        self.seq: int = 0

    def _get_weight(self, p: Process) -> float:
        return self.WEIGHT_MAP.get(p.kind, 1024.0)

    def on_arrival(self, p: Process, now: float) -> None:
        self.vruntime[p.pid] = self.min_vruntime
        self.seq += 1
        heapq.heappush(self.heap, (self.min_vruntime, self.seq, p))

    def on_ready(self, p: Process, now: float) -> None:
        vr = max(self.vruntime.get(p.pid, self.min_vruntime), self.min_vruntime)
        self.vruntime[p.pid] = vr
        self.seq += 1
        heapq.heappush(self.heap, (vr, self.seq, p))

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        weight = self._get_weight(p)
        delta_vruntime = ran * (1024.0 / weight)
        new_vr = self.vruntime.get(p.pid, self.min_vruntime) + delta_vruntime
        self.vruntime[p.pid] = new_vr

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        if not self.heap:
            return None
        vr, _, p = heapq.heappop(self.heap)
        self.min_vruntime = max(self.min_vruntime, vr)
        n_runnable = len(self.heap) + 1
        time_slice = max(self.min_granularity_ms, self.latency_target_ms / max(1, n_runnable))
        return p, time_slice


class RR_Cgroup(Scheduler):
    """Round Robin with cgroup-style hard per-process CPU bandwidth cap and child budget."""

    name = "RR+cgroup"

    def __init__(
        self,
        quantum_ms: float = 10.0,
        quota_period_ms: float = 100.0,
        max_quota_share: float = 0.50,
        max_children: int = 30,
    ) -> None:
        self.quantum_ms = float(quantum_ms)
        self.quota_period_ms = float(quota_period_ms)
        self.max_quota_ms = float(quota_period_ms * max_quota_share)
        self.max_children = max_children

        self.queue: deque[Process] = deque()
        self.period_used: dict[int, float] = {}
        self.last_period_reset: float = 0.0
        self.child_count: dict[int, int] = {}
        self._flagged: dict[int, float] = {}

    def _reset_period_if_needed(self, now: float) -> None:
        if now - self.last_period_reset >= self.quota_period_ms:
            self.period_used.clear()
            self.last_period_reset = now

    def on_arrival(self, p: Process, now: float) -> None:
        if p.ppid is not None:
            self.child_count[p.ppid] = self.child_count.get(p.ppid, 0) + 1
            if self.child_count[p.ppid] > self.max_children:
                if p.ppid not in self._flagged:
                    self._flagged[p.ppid] = now
        self.queue.append(p)

    def on_ready(self, p: Process, now: float) -> None:
        self.queue.append(p)

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        self._reset_period_if_needed(now)
        used = self.period_used.get(p.pid, 0.0) + ran
        self.period_used[p.pid] = used

        if used >= self.max_quota_ms:
            if p.pid not in self._flagged:
                self._flagged[p.pid] = now

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        self._reset_period_if_needed(now)
        if not self.queue:
            return None
        for _ in range(len(self.queue)):
            p = self.queue.popleft()
            used = self.period_used.get(p.pid, 0.0)
            if used < self.max_quota_ms:
                rem_quota = self.max_quota_ms - used
                q = min(self.quantum_ms, rem_quota)
                return p, max(1.0, q)
            self.queue.append(p)

        # Work-conserving fallback: run first available with minimal quantum
        p = self.queue.popleft()
        return p, 2.0

    def flagged(self) -> dict[int, float]:
        return dict(self._flagged)


class RR_LearnedDetector(Scheduler):
    """Round Robin with an offline/online trained ML anomaly detector (IsolationForest) on features.

    Serves as the ML upper-bound baseline per Research Report A4.
    """

    name = "RR+LearnedDetector"

    def __init__(
        self,
        quantum_ms: float = 10.0,
        throttled_quantum_ms: float = 2.0,
        window_ms: float = 500.0,
    ) -> None:
        self.quantum_ms = float(quantum_ms)
        self.throttled_quantum_ms = float(throttled_quantum_ms)
        self.window_ms = float(window_ms)

        self.queue: deque[Process] = deque()
        self.slice_history: dict[int, deque[tuple[float, float]]] = {}
        self._flagged: dict[int, float] = {}
        self.children_count: dict[int, int] = {}

        # Lightweight IsolationForest model from scikit-learn
        from sklearn.ensemble import IsolationForest

        self.detector = IsolationForest(n_estimators=25, contamination=0.05, random_state=42)
        # Seed the model with baseline benign operational feature vectors:
        # [cpu_share, fork_rate, burst_len]
        benign_calibration_data = np.array([
            [0.02, 0.0, 5.0],
            [0.05, 0.0, 10.0],
            [0.10, 0.0, 15.0],
            [0.15, 0.0, 30.0],
            [0.25, 0.0, 50.0],
            [0.40, 0.0, 80.0],
            [0.60, 0.0, 120.0],
            [0.85, 0.0, 150.0],
            [0.01, 0.0, 2.0],
            [0.08, 0.0, 20.0],
        ])
        self.detector.fit(benign_calibration_data)

    def _cpu_share(self, pid: int, now: float) -> float:
        history = self.slice_history.get(pid)
        if not history:
            return 0.0
        cutoff = now - self.window_ms
        while history and history[0][0] < cutoff:
            history.popleft()
        cpu_ran = sum(r for (t, r) in history if t >= cutoff)
        return float(min(1.0, max(0.0, cpu_ran / self.window_ms)))

    def on_arrival(self, p: Process, now: float) -> None:
        if p.ppid is not None:
            self.children_count[p.ppid] = self.children_count.get(p.ppid, 0) + 1
        self.queue.append(p)
        if p.pid not in self.slice_history:
            self.slice_history[p.pid] = deque()

    def on_ready(self, p: Process, now: float) -> None:
        self.queue.append(p)

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        if p.pid not in self.slice_history:
            self.slice_history[p.pid] = deque()
        self.slice_history[p.pid].append((now, ran))

        share = self._cpu_share(p.pid, now)
        fork_cnt = self.children_count.get(p.pid, 0)

        # Anomaly vector: [share, fork_rate_proxy, burst_used]
        vec = np.array([[share, float(fork_cnt), ran]])
        pred = self.detector.predict(vec)[0]  # -1 for anomaly, 1 for normal

        # Outlier detected and sustained share or excessive forks
        if pred == -1 and (share >= 0.50 or fork_cnt > 15):
            if p.pid not in self._flagged:
                self._flagged[p.pid] = now

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        if not self.queue:
            return None
        p = self.queue.popleft()
        is_anom = p.pid in self._flagged
        q = self.throttled_quantum_ms if is_anom else self.quantum_ms
        return p, q

    def flagged(self) -> dict[int, float]:
        return dict(self._flagged)

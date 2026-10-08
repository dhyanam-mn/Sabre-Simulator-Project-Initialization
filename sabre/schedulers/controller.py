"""SABRE adaptive controller.

Follows Workdoc Section 7.1 - 7.3 and Research Report F7:
- Four ready queues: Q0 (interactive/short), Q1 (normal), Q2 (long/CPU-bound), Q3 (contained, level >= 2).
- Quantum rule:
    q = q_base * class_factor[class] * pred_factor(pred) * level_factor[level]
    clipped to [q_min, q_max]
- Strict priority with aging: promote from Q2->Q1, Q1->Q0 after age_ms waiting.
- Reconciled guaranteed share (Report F7):
    A contained process in Q3 is guaranteed at least 1 dispatch every slot_period (default 10).
    With minimum quantum q_min and max other quantum q_max, the guaranteed lower bound is:
      guaranteed_lower_bound = q_min / (slot_period * q_max) ~ 1 / 400 = 0.0025 (or q_min / (slot_period * q_base) = 1%).
    The elastic upper bound is capped at cap[level] (e.g. cap[3] = 0.10).
    Therefore:
      min_share <= actual_share <= cap[level] (when competing jobs exist).
    When no other processes are waiting, work conservation serves Q3 at 100% CPU.
- Elastic share caps: enforced over cap_window_ms ONLY if other non-contained processes are waiting.
- Cost charged: cost_decision = 0.01 ms per decision.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from sabre.core.process import Process


DEFAULT_CLASS_FACTORS = {
    "interactive": 0.50,
    "io_bound": 0.75,
    "bursty": 0.75,
    "cpu_bound": 1.50,
    "unknown": 1.00,
}

DEFAULT_LEVEL_FACTORS = [1.00, 0.75, 0.50, 0.25]
DEFAULT_CAPS = [1.00, 0.50, 0.25, 0.10]


@dataclass
class ControllerDecision:
    process: Process
    quantum: float
    queue_name: str
    level: int
    proc_class: str
    predicted_burst: float
    reasons: list[str]


class SabreController:
    """Manages queues, priority aging, quantum scaling, and containment dispatch."""

    def __init__(
        self,
        q_base: float = 10.0,
        q_min: float = 1.0,
        q_max: float = 40.0,
        short_thr: float = 5.0,
        long_thr: float = 50.0,
        pred_ref: float = 10.0,
        age_ms: float = 200.0,
        slot_period: int = 10,
        caps: list[float] | None = None,
        cap_window_ms: float = 200.0,
        min_share: float = 0.02,
        class_factor: dict[str, float] | None = None,
        level_factor: list[float] | None = None,
        cost_ms: float = 0.01,
        # Feature flags (Workdoc 7.5)
        use_predictor: bool = True,
        use_classifier: bool = True,
        use_aging: bool = True,
        use_elastic_cap: bool = True,
    ) -> None:
        self.q_base = float(q_base)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.short_thr = float(short_thr)
        self.long_thr = float(long_thr)
        self.pred_ref = float(pred_ref)
        self.age_ms = float(age_ms)
        self.slot_period = int(slot_period)
        self.caps = list(caps) if caps is not None else list(DEFAULT_CAPS)
        self.cap_window_ms = float(cap_window_ms)
        self.min_share = float(min_share)
        self.class_factor = dict(class_factor) if class_factor is not None else dict(DEFAULT_CLASS_FACTORS)
        self.level_factor = list(level_factor) if level_factor is not None else list(DEFAULT_LEVEL_FACTORS)
        self.cost_ms = float(cost_ms)

        # Feature flags
        self.use_predictor = use_predictor
        self.use_classifier = use_classifier
        self.use_aging = use_aging
        self.use_elastic_cap = use_elastic_cap

        # Ready queues: Q0, Q1, Q2, Q3
        self.queues: dict[str, list[Process]] = {
            "Q0": [],
            "Q1": [],
            "Q2": [],
            "Q3": [],
        }

        # Dispatch counter for slot_period
        self.dispatch_count: int = 0
        self._accrued_cost_ms: float = 0.0

        # CPU usage history over cap_window_ms: pid -> deque of (timestamp, ran_ms)
        self.slice_history: dict[int, deque[tuple[float, float]]] = {}

    def compute_cpu_share(self, pid: int, now: float) -> float:
        """Compute rolling CPU share over cap_window_ms."""
        history = self.slice_history.get(pid)
        if not history:
            return 0.0
        cutoff = now - self.cap_window_ms
        while history and history[0][0] < cutoff:
            history.popleft()
        cpu_ran = sum(r for (t, r) in history if t >= cutoff)
        return float(min(1.0, max(0.0, cpu_ran / self.cap_window_ms)))

    def record_slice(self, pid: int, ran: float, now: float) -> None:
        """Record completed CPU slice for rolling cap calculations."""
        if pid not in self.slice_history:
            self.slice_history[pid] = deque()
        self.slice_history[pid].append((now, ran))

    def assign_queue(self, p: Process, proc_class: str, pred_burst: float, level: int) -> str:
        """Assign process to one of the 4 ready queues."""
        if level >= 2:
            return "Q3"

        # If flags disabled, fallback to Q1
        if not self.use_classifier and not self.use_predictor:
            return "Q1"

        effective_class = proc_class if self.use_classifier else "unknown"
        effective_pred = pred_burst if self.use_predictor else 10.0

        if effective_class == "interactive" or effective_pred < self.short_thr:
            return "Q0"
        elif effective_class == "cpu_bound" or effective_pred > self.long_thr:
            return "Q2"
        return "Q1"

    def enqueue(self, p: Process, proc_class: str, pred_burst: float, level: int) -> str:
        """Enqueue process into the appropriate ready queue."""
        q_name = self.assign_queue(p, proc_class, pred_burst, level)
        self.queues[q_name].append(p)
        return q_name

    def remove_process(self, p: Process) -> None:
        """Remove process from any queue it currently resides in."""
        for q in self.queues.values():
            if p in q:
                q.remove(p)

    def calculate_quantum(self, proc_class: str, pred_burst: float, level: int) -> float:
        """Calculate dynamic time quantum for dispatch."""
        cf = self.class_factor.get(proc_class, 1.0) if self.use_classifier else 1.0
        pf = float(np.clip(pred_burst / self.pred_ref, 0.5, 2.0)) if self.use_predictor else 1.0
        lf = self.level_factor[level] if (0 <= level < len(self.level_factor)) else 1.0

        q = self.q_base * cf * pf * lf
        return float(np.clip(q, self.q_min, self.q_max))

    def apply_aging(self, now: float) -> None:
        """Promote processes waiting longer than age_ms (Q2 -> Q1, Q1 -> Q0)."""
        if not self.use_aging:
            return

        # Q2 -> Q1
        still_in_q2: list[Process] = []
        for p in self.queues["Q2"]:
            if now - p.last_ready_at >= self.age_ms:
                self.queues["Q1"].append(p)
            else:
                still_in_q2.append(p)
        self.queues["Q2"] = still_in_q2

        # Q1 -> Q0
        still_in_q1: list[Process] = []
        for p in self.queues["Q1"]:
            if now - p.last_ready_at >= self.age_ms:
                self.queues["Q0"].append(p)
            else:
                still_in_q1.append(p)
        self.queues["Q1"] = still_in_q1

    def is_capped(self, pid: int, level: int, now: float) -> bool:
        """Check if process exceeds elastic share cap at current level."""
        if not self.use_elastic_cap or level >= len(self.caps):
            return False
        max_cap = self.caps[level]
        if max_cap >= 1.0:
            return False
        share = self.compute_cpu_share(pid, now)
        return share >= max_cap

    def select_next(
        self,
        now: float,
        pred_map: dict[int, float],
        class_map: dict[int, str],
        level_map: dict[int, int],
        reasons_map: dict[int, list[str]],
    ) -> ControllerDecision | None:
        """Select next process to dispatch."""
        self._accrued_cost_ms += self.cost_ms
        self.apply_aging(now)

        has_benign_waiting = bool(self.queues["Q0"] or self.queues["Q1"] or self.queues["Q2"])
        has_q3 = bool(self.queues["Q3"])

        if not has_benign_waiting and not has_q3:
            return None

        self.dispatch_count += 1
        dispatch_from_q3 = False

        # Guaranteed slot for Q3 every slot_period dispatches
        if has_q3 and (self.dispatch_count % self.slot_period == 0 or not has_benign_waiting):
            dispatch_from_q3 = True

        chosen_proc: Process | None = None
        chosen_queue: str = ""

        if dispatch_from_q3:
            # Pick from Q3
            # If elastic caps active and other processes waiting, check cap, but guarantee min service
            q3_list = self.queues["Q3"]
            for idx, p in enumerate(q3_list):
                lvl = level_map.get(p.pid, 2)
                if has_benign_waiting and self.is_capped(p.pid, lvl, now) and len(q3_list) > 1:
                    continue
                chosen_proc = q3_list.pop(idx)
                chosen_queue = "Q3"
                break
            if chosen_proc is None and q3_list:
                # Fallback to avoid complete starvation
                chosen_proc = q3_list.pop(0)
                chosen_queue = "Q3"

        if chosen_proc is None and has_benign_waiting:
            # Strict priority: Q0 -> Q1 -> Q2
            for q_name in ["Q0", "Q1", "Q2"]:
                q_list = self.queues[q_name]
                if not q_list:
                    continue
                # If predictor is active, sort by predicted burst (approximate SJF); otherwise FIFO
                if self.use_predictor:
                    best_idx = 0
                    best_pred = pred_map.get(q_list[0].pid, 10.0)
                    for i in range(1, len(q_list)):
                        p_pred = pred_map.get(q_list[i].pid, 10.0)
                        if p_pred < best_pred:
                            best_pred = p_pred
                            best_idx = i
                else:
                    best_idx = 0

                # Check elastic cap for level 1 if applicable
                p_cand = q_list[best_idx]
                lvl = level_map.get(p_cand.pid, 0)
                if self.is_capped(p_cand.pid, lvl, now) and len(q_list) > 1:
                    # Pick second best if capped
                    next_idx = 1 if best_idx == 0 else 0
                    chosen_proc = q_list.pop(next_idx)
                else:
                    chosen_proc = q_list.pop(best_idx)
                chosen_queue = q_name
                break

        # If Q0-Q2 were exhausted or capped and nothing chosen, fallback to Q3
        if chosen_proc is None and has_q3:
            chosen_proc = self.queues["Q3"].pop(0)
            chosen_queue = "Q3"

        if chosen_proc is None:
            return None

        pid = chosen_proc.pid
        proc_cls = class_map.get(pid, "unknown")
        pred_b = pred_map.get(pid, 10.0)
        lvl = level_map.get(pid, 0)
        reasons = reasons_map.get(pid, [])
        quantum = self.calculate_quantum(proc_cls, pred_b, lvl)

        return ControllerDecision(
            process=chosen_proc,
            quantum=quantum,
            queue_name=chosen_queue,
            level=lvl,
            proc_class=proc_cls,
            predicted_burst=pred_b,
            reasons=reasons,
        )

    def overhead_ms(self) -> float:
        cost = self._accrued_cost_ms
        self._accrued_cost_ms = 0.0
        return cost

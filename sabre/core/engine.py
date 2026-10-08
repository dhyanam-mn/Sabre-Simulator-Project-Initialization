"""Discrete-event simulation engine for SABRE.

Follows Workdoc Section 5.1:
- Event-driven min-heap loop: ARRIVAL, IO_COMPLETE, SLICE_END, SPAWN
- Context switch cost ctx_switch_ms charged when running pid changes
- Idle jump: advances time to the next scheduled event when CPU is idle
- Scheduler overhead_ms() charged to the clock
- Hard stop at sim_end_ms or when all benign processes are DONE
- Survivors get finish=None
- Per-burst response samples tracked
"""

from __future__ import annotations

import heapq
from enum import IntEnum
from typing import Any
import numpy as np

from sabre.core.process import Process, ProcState
from sabre.schedulers.base import Scheduler


class EventType(IntEnum):
    ARRIVAL = 1
    IO_COMPLETE = 2
    SLICE_END = 3
    SPAWN = 4


class Simulator:
    """Deterministic discrete-event simulator."""

    def __init__(
        self,
        scheduler: Scheduler,
        processes: list[Process],
        sim_end_ms: float = 60000.0,
        ctx_switch_ms: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.scheduler = scheduler
        self.initial_processes = processes
        self.sim_end_ms = float(sim_end_ms)
        self.ctx_switch_ms = float(ctx_switch_ms)
        self.rng = np.random.Generator(np.random.PCG64(seed))

        # Simulation state
        self.now: float = 0.0
        self.current_proc: Process | None = None
        self.current_quantum: float = 0.0
        self.event_heap: list[tuple[float, int, int, Any]] = []  # (time, event_type, seq, payload)
        self.event_seq: int = 0
        self.all_processes: list[Process] = []

        # Tracking metrics
        self.ctx_switches: int = 0
        self.total_overhead_ms: float = 0.0

    def _push_event(self, event_time: float, event_type: EventType, payload: Any) -> None:
        self.event_seq += 1
        heapq.heappush(self.event_heap, (event_time, int(event_type), self.event_seq, payload))

    def run(self) -> dict[str, Any]:
        """Run the simulation until sim_end_ms or all benign processes finish."""
        # 1. Schedule initial arrivals
        for p in self.initial_processes:
            self.all_processes.append(p)
            self._push_event(p.arrival, EventType.ARRIVAL, p)

        # Main event loop
        while self.now < self.sim_end_ms:
            # Check termination: all benign processes done
            benign_procs = [p for p in self.all_processes if p.label == 0]
            if benign_procs and all(p.state == ProcState.DONE for p in benign_procs):
                break

            # If no CPU job is running: try to dispatch one
            if self.current_proc is None:
                # Charge scheduler overhead before dispatching
                oh = self.scheduler.overhead_ms()
                if oh > 0:
                    self.now += oh
                    self.total_overhead_ms += oh
                    if self.now >= self.sim_end_ms:
                        break

                dispatch = self.scheduler.pick_next(self.now)
                if dispatch is not None:
                    proc, quantum = dispatch
                    # Context switch cost if switching to a new process
                    if self.ctx_switch_ms > 0:
                        self.now += self.ctx_switch_ms
                        self.ctx_switches += 1
                        if self.now >= self.sim_end_ms:
                            break

                    self.current_proc = proc
                    self.current_quantum = quantum

                    # Transition from READY to RUNNING
                    proc.wait_time += self.now - proc.last_ready_at
                    proc.state = ProcState.RUNNING

                    # Track first run
                    if proc.first_run is None:
                        proc.first_run = self.now

                    # Track per-burst response time for interactive processes (from IO completion to dispatch)
                    if proc.last_io_complete_at is not None:
                        sample = self.now - proc.last_io_complete_at
                        proc.response_samples.append(sample)
                        proc.last_io_complete_at = None
                    elif len(proc.response_samples) == 0:
                        # Initial response time (from arrival to first dispatch)
                        proc.response_samples.append(self.now - proc.arrival)

                    # Determine CPU burst
                    if proc.remaining_burst <= 0:
                        next_b = proc.next_cpu_burst(self.now, self.rng)
                        # Clamp by remaining total_cpu if total_cpu is finite
                        if proc.total_cpu != float("inf"):
                            rem_total = proc.total_cpu - proc.cpu_used
                            next_b = min(next_b, max(0.0, rem_total))
                        proc.remaining_burst = next_b

                    ran = min(quantum, proc.remaining_burst)
                    # Schedule SLICE_END event
                    slice_end_time = self.now + ran
                    self._push_event(slice_end_time, EventType.SLICE_END, (proc, ran))
                    continue

                else:
                    # CPU is idle: jump to next event
                    if not self.event_heap:
                        # Nothing scheduled and nothing ready
                        break
                    next_time, _, _, _ = self.event_heap[0]
                    if next_time > self.now:
                        self.now = next_time
                    if self.now >= self.sim_end_ms:
                        break

            # Handle next event from the min-heap
            if not self.event_heap:
                continue

            event_time, event_type, _, payload = heapq.heappop(self.event_heap)
            if event_time > self.now:
                self.now = event_time

            if self.now >= self.sim_end_ms:
                break

            if event_type == EventType.ARRIVAL:
                proc = payload
                proc.state = ProcState.READY
                proc.last_ready_at = self.now
                self.scheduler.on_arrival(proc, self.now)

            elif event_type == EventType.IO_COMPLETE:
                proc = payload
                proc.state = ProcState.READY
                proc.last_ready_at = self.now
                proc.last_io_complete_at = self.now
                self.scheduler.on_ready(proc, self.now)

            elif event_type == EventType.SLICE_END:
                proc, ran = payload
                proc.cpu_used += ran
                proc.remaining_burst -= ran

                # Check spawned children
                children = proc.children_to_spawn(ran, self.now, self.rng)
                for child in children:
                    self.all_processes.append(child)
                    self._push_event(self.now, EventType.SPAWN, child)

                burst_done = proc.remaining_burst <= 1e-9
                if burst_done:
                    proc.remaining_burst = 0.0

                # Notify scheduler of slice completion
                self.scheduler.on_slice_end(proc, ran, burst_done, self.now)

                # Check if process has finished total_cpu
                is_done = (proc.total_cpu != float("inf")) and (proc.cpu_used >= proc.total_cpu - 1e-9)

                if is_done:
                    proc.state = ProcState.DONE
                    proc.finish = self.now
                    self.scheduler.on_exit(proc, self.now)
                elif burst_done:
                    # IO wait
                    io_wait = proc.next_io_wait(self.now, self.rng)
                    if io_wait > 0:
                        proc.state = ProcState.WAITING
                        self._push_event(self.now + io_wait, EventType.IO_COMPLETE, proc)
                    else:
                        # Immediately ready for next burst
                        proc.state = ProcState.READY
                        proc.last_ready_at = self.now
                        self.scheduler.on_ready(proc, self.now)
                else:
                    # Preempted (slice ended before burst completed)
                    proc.state = ProcState.READY
                    proc.last_ready_at = self.now
                    self.scheduler.on_ready(proc, self.now)

                self.current_proc = None

            elif event_type == EventType.SPAWN:
                child = payload
                child.state = ProcState.READY
                child.last_ready_at = self.now
                self.scheduler.on_arrival(child, self.now)

        # End of simulation cleanup
        # Any process not DONE at sim_end gets finish = None
        for p in self.all_processes:
            if p.state != ProcState.DONE:
                p.finish = None

        return {
            "sim_time": min(self.now, self.sim_end_ms),
            "ctx_switches": self.ctx_switches,
            "overhead_ms": self.total_overhead_ms,
            "processes": self.all_processes,
        }

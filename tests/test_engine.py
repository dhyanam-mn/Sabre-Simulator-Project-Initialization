"""Tests for simulator core engine, FCFS and Round Robin.

Covers Workdoc 5.1 requirements and acceptance checks:
1. FCFS on a hand-built 3-process example matches hand-computed waits.
2. RR conserves total CPU time.
3. Same seed gives identical metrics.
4. Process with total_cpu=inf cannot hang the loop (terminates at sim_end_ms).
5. No process finishes before arrival + total_cpu.
"""

from __future__ import annotations

import copy
import pytest
from sabre.core.process import Process, ProcState
from sabre.core.engine import Simulator
from sabre.schedulers.baselines import FCFS, RoundRobin


def test_fcfs_hand_computed_3_process_waits():
    """Verify FCFS waiting times against exact manual hand calculations.

    Hand calculation details:
    -------------------------
    Settings:
      - ctx_switch_ms = 0.0 (for pure theoretical FCFS verification)
      - Overhead = 0.0

    Processes:
      - P1: arrival = 0.0, total_cpu = 10.0 (one burst = 10.0, no I/O)
      - P2: arrival = 2.0, total_cpu = 5.0 (one burst = 5.0, no I/O)
      - P3: arrival = 4.0, total_cpu = 8.0 (one burst = 8.0, no I/O)

    Timeline:
      t = 0.0: P1 arrives -> READY -> Dispatched immediately.
               P1 wait_time = 0.0 - 0.0 = 0.0 ms.
      t = 2.0: P2 arrives -> READY.
      t = 4.0: P3 arrives -> READY.
      t = 10.0: P1 finishes execution (ran 10 ms).
                P1 state = DONE, finish = 10.0.
                Next in queue is P2.
                P2 is dispatched at t = 10.0.
                P2 wait_time = 10.0 - 2.0 = 8.0 ms.
      t = 15.0: P2 finishes execution (ran 5 ms).
                P2 state = DONE, finish = 15.0.
                Next in queue is P3.
                P3 is dispatched at t = 15.0.
                P3 wait_time = 15.0 - 4.0 = 11.0 ms.
      t = 23.0: P3 finishes execution (ran 8 ms).
                P3 state = DONE, finish = 23.0.

    Expected Waiting Times:
      P1: 0.0 ms
      P2: 8.0 ms
      P3: 11.0 ms
      Average wait time = (0.0 + 8.0 + 11.0) / 3 = 19.0 / 3 ≈ 6.3333 ms.
    """
    p1 = Process(pid=1, ppid=None, arrival=0.0, kind="cpu_bound", label=0, total_cpu=10.0)
    p2 = Process(pid=2, ppid=None, arrival=2.0, kind="cpu_bound", label=0, total_cpu=5.0)
    p3 = Process(pid=3, ppid=None, arrival=4.0, kind="cpu_bound", label=0, total_cpu=8.0)

    sched = FCFS()
    sim = Simulator(scheduler=sched, processes=[p1, p2, p3], ctx_switch_ms=0.0)
    res = sim.run()

    procs = {p.pid: p for p in res["processes"]}

    # Compare exact hand calculations
    assert pytest.approx(procs[1].wait_time) == 0.0
    assert pytest.approx(procs[2].wait_time) == 8.0
    assert pytest.approx(procs[3].wait_time) == 11.0

    # Finish times
    assert pytest.approx(procs[1].finish) == 10.0
    assert pytest.approx(procs[2].finish) == 15.0
    assert pytest.approx(procs[3].finish) == 23.0


def test_rr_conserves_total_cpu_time():
    """Verify that Round Robin conserves total CPU time across all processes.

    Total CPU time executed across all processes must equal the sum of their total_cpu demands.
    """
    p1 = Process(pid=1, ppid=None, arrival=0.0, kind="interactive", label=0, total_cpu=25.0)
    p2 = Process(pid=2, ppid=None, arrival=5.0, kind="cpu_bound", label=0, total_cpu=40.0)
    p3 = Process(pid=3, ppid=None, arrival=10.0, kind="bursty", label=0, total_cpu=15.0)

    total_demanded = p1.total_cpu + p2.total_cpu + p3.total_cpu  # 80.0 ms

    sched = RoundRobin(quantum_ms=10.0)
    sim = Simulator(scheduler=sched, processes=[p1, p2, p3], ctx_switch_ms=0.05)
    res = sim.run()

    total_used = sum(p.cpu_used for p in res["processes"])
    assert pytest.approx(total_used, rel=1e-5) == total_demanded
    for p in res["processes"]:
        assert p.state == ProcState.DONE
        assert pytest.approx(p.cpu_used) == p.total_cpu


def test_no_process_finishes_before_arrival_plus_total_cpu():
    """Physical causality check: no process can finish earlier than arrival + total_cpu."""
    procs = [
        Process(pid=1, ppid=None, arrival=0.0, kind="interactive", label=0, total_cpu=20.0),
        Process(pid=2, ppid=None, arrival=15.0, kind="cpu_bound", label=0, total_cpu=35.0),
        Process(pid=3, ppid=None, arrival=25.0, kind="io_bound", label=0, total_cpu=10.0),
    ]

    sched = RoundRobin(quantum_ms=5.0)
    sim = Simulator(scheduler=sched, processes=procs, ctx_switch_ms=0.1)
    res = sim.run()

    for p in res["processes"]:
        assert p.finish is not None
        assert p.finish >= p.arrival + p.total_cpu - 1e-9


def test_total_cpu_inf_does_not_hang():
    """Verify that an infinite CPU hog process does not hang the loop and stops at sim_end_ms."""
    hog = Process(pid=99, ppid=None, arrival=0.0, kind="hog", label=1, total_cpu=float("inf"))
    benign = Process(pid=1, ppid=None, arrival=0.0, kind="cpu_bound", label=0, total_cpu=50.0)

    sched = RoundRobin(quantum_ms=10.0)
    sim = Simulator(scheduler=sched, processes=[hog, benign], sim_end_ms=500.0, ctx_switch_ms=0.05)
    res = sim.run()

    # Simulation finishes within sim_end_ms
    assert res["sim_time"] <= 500.1
    p_hog = next(p for p in res["processes"] if p.pid == 99)
    assert p_hog.finish is None  # Survivor recorded with finish=None
    assert p_hog.cpu_used > 0.0

    p_benign = next(p for p in res["processes"] if p.pid == 1)
    assert p_benign.state == ProcState.DONE
    assert p_benign.finish is not None


def test_deterministic_repeat():
    """Verify that identical seed and initial processes produce bit-for-bit identical results."""
    def run_sim(seed: int):
        procs = [
            Process(pid=1, ppid=None, arrival=0.0, kind="interactive", label=0, total_cpu=30.0),
            Process(pid=2, ppid=None, arrival=5.0, kind="cpu_bound", label=0, total_cpu=60.0),
            Process(pid=3, ppid=None, arrival=10.0, kind="io_bound", label=0, total_cpu=20.0),
        ]
        sched = RoundRobin(quantum_ms=10.0)
        sim = Simulator(scheduler=sched, processes=procs, seed=seed, ctx_switch_ms=0.05)
        res = sim.run()
        return [(p.pid, p.wait_time, p.cpu_used, p.finish) for p in res["processes"]]

    res1 = run_sim(seed=12345)
    res2 = run_sim(seed=12345)
    assert res1 == res2

"""Tests for workload generator and baseline schedulers.

Requirements:
- Workload generator produces consistent deterministic processes for identical seed
- All schedulers run the seeded workload to completion
- Preserves pre-sampled burst sequences across cloned processes
- Verifies relative performance ordering on non-preemptive workloads
"""

from __future__ import annotations

import pytest

from sabre.core.workload import WorkloadGenerator, clone_processes
from sabre.core.engine import Simulator
from sabre.core.process import ProcState
from sabre.schedulers.baselines import (
    FCFS,
    SJF_Oracle,
    SJF_EMA,
    RoundRobin,
    PriorityScheduler,
    MLFQ,
    RR_StaticThreshold,
    CFS_Like,
    RR_Cgroup,
)


def test_workload_generator_determinism():
    """Verify workload generator produces identical processes on same seed."""
    gen = WorkloadGenerator(n_benign=20, arrival_rate_per_s=4.0)
    w1 = gen.generate(seed=42)
    w2 = gen.generate(seed=42)

    assert len(w1) == len(w2)
    for p1, p2 in zip(w1, w2):
        assert p1.pid == p2.pid
        assert p1.arrival == p2.arrival
        assert p1.kind == p2.kind
        assert p1.total_cpu == p2.total_cpu
        assert getattr(p1, "_raw_burst_pool") == getattr(p2, "_raw_burst_pool")
        assert getattr(p1, "_raw_io_pool") == getattr(p2, "_raw_io_pool")


def test_clone_processes_isolation():
    """Verify cloned processes do not share execution state."""
    gen = WorkloadGenerator(n_benign=5, arrival_rate_per_s=4.0)
    base = gen.generate(seed=10)
    clones = clone_processes(base)

    assert len(clones) == len(base)
    # Mutate clone
    clones[0].cpu_used = 100.0
    clones[0].state = ProcState.RUNNING

    assert base[0].cpu_used == 0.0
    assert base[0].state == ProcState.NEW


def test_all_baselines_run_to_completion():
    """Run all baseline schedulers on a seeded workload and verify all finish cleanly."""
    gen = WorkloadGenerator(n_benign=15, arrival_rate_per_s=4.0)
    base = gen.generate(seed=7)

    schedulers = [
        FCFS(),
        SJF_Oracle(),
        SJF_EMA(),
        RoundRobin(quantum_ms=10.0),
        PriorityScheduler(quantum_ms=10.0),
        MLFQ(),
        RR_StaticThreshold(),
        CFS_Like(),
        RR_Cgroup(),
    ]

    for sched in schedulers:
        procs = clone_processes(base)
        sim = Simulator(scheduler=sched, processes=procs, sim_end_ms=120000.0)
        res = sim.run()
        for p in res["processes"]:
            assert p.state == ProcState.DONE
            assert p.finish is not None
            assert p.finish >= p.arrival + p.total_cpu - 1e-9


def test_sjf_oracle_vs_fcfs_on_seed_0():
    """Verify SJF-oracle achieves lower average wait and turnaround than FCFS."""
    gen = WorkloadGenerator(n_benign=30, arrival_rate_per_s=4.0)
    base = gen.generate(seed=0)

    # SJF-oracle
    procs_sjf = clone_processes(base)
    sim_sjf = Simulator(scheduler=SJF_Oracle(), processes=procs_sjf, sim_end_ms=250000.0)
    res_sjf = sim_sjf.run()
    avg_wait_sjf = sum(p.wait_time for p in res_sjf["processes"]) / len(res_sjf["processes"])
    avg_ta_sjf = sum(p.finish - p.arrival for p in res_sjf["processes"]) / len(res_sjf["processes"])

    # FCFS
    procs_fcfs = clone_processes(base)
    sim_fcfs = Simulator(scheduler=FCFS(), processes=procs_fcfs, sim_end_ms=250000.0)
    res_fcfs = sim_fcfs.run()
    avg_wait_fcfs = sum(p.wait_time for p in res_fcfs["processes"]) / len(res_fcfs["processes"])
    avg_ta_fcfs = sum(p.finish - p.arrival for p in res_fcfs["processes"]) / len(res_fcfs["processes"])

    assert avg_wait_sjf < avg_wait_fcfs
    assert avg_ta_sjf < avg_ta_fcfs

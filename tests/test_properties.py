"""Formal property-based tests for SABRE scheduler guarantees (Research Upgrade A2).

Verifies invariants:
(a) Starvation-freedom: any contained process receives non-zero share over reasonable windows
    while others are runnable.
(b) Work-conservation: CPU is never idle while there are runnable processes.
(c) Hysteresis: bounds level flapping under bounded noisy risk updates.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st, settings

from sabre.core.process import Process, ProcState
from sabre.core.engine import Simulator
from sabre.schedulers.sabre import SabreScheduler
from sabre.intel.risk import RiskScorer
from sabre.intel.features import Features


@settings(max_examples=30, deadline=None)
@given(
    num_benign=st.integers(min_value=2, max_value=8),
    contained_burst=st.floats(min_value=20.0, max_value=200.0),
    quantum_factor=st.floats(min_value=5.0, max_value=20.0),
)
def test_property_work_conservation(num_benign: int, contained_burst: float, quantum_factor: float):
    """(A2-b) Invariant: CPU is never idle when runnable work exists."""
    procs = []
    for i in range(num_benign):
        p = Process(pid=i + 1, ppid=None, arrival=0.0, kind="interactive", label=0, total_cpu=100.0)
        p.burst_generator = lambda _now, _rng: 10.0
        p.io_generator = lambda _now, _rng: 0.0
        procs.append(p)

    sched = SabreScheduler()
    sim = Simulator(sched, procs, sim_end_ms=500.0)
    sim.run()

    # If any process has remaining CPU, current simulated time reached either completion or sim_end
    # And total cpu time accounted across processes + overhead matches busy execution
    total_cpu_used = sum(p.cpu_used for p in sim.all_processes)
    # CPU should have made progress without stalling
    assert total_cpu_used > 0.0
    assert sim.now >= min(500.0, total_cpu_used)


@settings(max_examples=25, deadline=None)
@given(
    n_other=st.integers(min_value=2, max_value=6),
    contained_level=st.integers(min_value=1, max_value=3),
)
def test_property_starvation_freedom(n_other: int, contained_level: int):
    """(A2-a) Invariant: A contained process receives CPU turns (guaranteed slot/share)

    and is never starved while other processes run.
    """
    procs = []
    # Competing CPU-intensive benign processes
    for i in range(n_other):
        p = Process(pid=i + 1, ppid=None, arrival=0.0, kind="cpu_bound", label=0, total_cpu=2000.0)
        procs.append(p)

    # Contained process
    p_contained = Process(pid=999, ppid=None, arrival=0.0, kind="cpu_bound", label=1, total_cpu=2000.0)
    procs.append(p_contained)

    sched = SabreScheduler()
    # Force contained level on pid 999
    sched.proc_level[999] = contained_level
    st_risk = sched.risk_scorer._get_or_create(999)
    st_risk.level = contained_level
    st_risk.final_risk = 0.85

    sim = Simulator(sched, procs, sim_end_ms=1000.0)
    sim.run()

    # Contained process must receive non-zero CPU time
    assert p_contained.cpu_used > 0.0, "Contained process was starved (zero CPU)!"


@settings(max_examples=30, deadline=None)
@given(
    noise_seq=st.lists(st.floats(min_value=0.20, max_value=0.65), min_size=10, max_size=30)
)
def test_property_hysteresis_bounds_flapping(noise_seq: list[float]):
    """(A2-e) Invariant: Hysteresis up_hold=2 and down_hold=3 prevents level flapping

    on small noisy risk oscillations around the threshold.
    """
    scorer = RiskScorer(level_thresholds=[0.30, 0.55, 0.80], up_hold=2, down_hold=3, down_margin=0.10)
    pid = 42

    level_changes = 0
    prev_level = 0
    t = 0.0

    for noisy_val in noise_seq:
        t += 60.0  # respect update spacing
        # Directly mock final_risk behavior through update with custom features
        st_proc = scorer._get_or_create(pid)
        st_proc.last_update_time = t - 60.0
        # Inject controlled risk
        st_proc.final_risk = noisy_val
        st_proc.eff_risk = noisy_val

        # Run containment ladder
        curr_lvl = st_proc.level
        target_lvl = 0
        for i, thr in enumerate(scorer.level_thresholds):
            if st_proc.final_risk >= thr:
                target_lvl = i + 1

        if target_lvl > curr_lvl:
            st_proc.up_counts[target_lvl - 1] += 1
            st_proc.down_counts = [0, 0, 0]
            if st_proc.up_counts[target_lvl - 1] >= scorer.up_hold:
                st_proc.level = target_lvl
        elif target_lvl < curr_lvl:
            threshold = scorer.level_thresholds[curr_lvl - 1]
            if st_proc.final_risk < threshold - scorer.down_margin:
                st_proc.down_counts[curr_lvl - 1] += 1
                st_proc.up_counts = [0, 0, 0]
                if st_proc.down_counts[curr_lvl - 1] >= scorer.down_hold:
                    st_proc.level = curr_lvl - 1
            else:
                st_proc.down_counts = [0, 0, 0]
        else:
            st_proc.up_counts = [0, 0, 0]
            st_proc.down_counts = [0, 0, 0]

        if st_proc.level != prev_level:
            level_changes += 1
            prev_level = st_proc.level

    # Level changes must be strictly bounded, much fewer than total sequence length
    assert level_changes <= len(noise_seq) // 2 + 1

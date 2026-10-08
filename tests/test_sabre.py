"""Property tests for SABRE scheduler and controller.

Requirements:
- Contained process never below min_share while others are busy
- CPU never idle with runnable work (work-conserving)
- Hysteresis bounds level flapping on a noisy benign series
- Lone contained process gets 100% CPU
- All flags off is within 5% of RR
"""

from __future__ import annotations

import numpy as np
import pytest

from sabre.core.process import Process, ProcState
from sabre.core.engine import Simulator
from sabre.core.workload import WorkloadGenerator, clone_processes
from sabre.schedulers.baselines import RoundRobin
from sabre.schedulers.sabre import SabreScheduler
from sabre.intel.risk import RiskScorer, DEFAULT_BASELINES
from sabre.intel.features import Features


def test_lone_contained_process_gets_100_percent_cpu():
    """Verify work-conservation: a single contained process (level 3) gets 100% CPU when alone."""
    contained = Process(pid=100, ppid=None, arrival=0.0, kind="hog", label=1, total_cpu=500.0)
    contained.burst_generator = lambda _now, _rng: 50.0

    sabre = SabreScheduler()
    # Force process into Q3 / level 3
    sabre.proc_level[100] = 3

    sim = Simulator(scheduler=sabre, processes=[contained], sim_end_ms=1000.0, ctx_switch_ms=0.0)
    res = sim.run()

    p = res["processes"][0]
    assert p.state == ProcState.DONE
    assert pytest.approx(p.cpu_used) == 500.0
    assert pytest.approx(p.finish, abs=0.2) == 500.0 + res["overhead_ms"]


def test_contained_process_never_below_min_share_while_others_busy():
    """Verify starvation-freedom: contained process receives >= min_share (~2%) while others are runnable."""
    contained = Process(pid=100, ppid=None, arrival=0.0, kind="hog", label=1, total_cpu=float("inf"))
    contained.burst_generator = lambda _now, _rng: 100.0

    # Competing busy benign processes
    competing = [
        Process(pid=i, ppid=None, arrival=0.0, kind="cpu_bound", label=0, total_cpu=float("inf"))
        for i in range(1, 5)
    ]
    for c in competing:
        c.burst_generator = lambda _now, _rng: 100.0

    sabre = SabreScheduler()
    sabre.proc_level[100] = 3  # Level 3 contained

    sim = Simulator(scheduler=sabre, processes=[contained] + competing, sim_end_ms=5000.0, ctx_switch_ms=0.05)
    res = sim.run()

    p_cont = next(p for p in res["processes"] if p.pid == 100)
    total_time = res["sim_time"]
    actual_share = p_cont.cpu_used / total_time
    assert actual_share >= 0.015, f"Contained process starved! Share was only {actual_share:.4f}"


def test_cpu_never_idle_with_runnable_work():
    """Verify CPU never goes idle when runnable processes are present."""
    procs = [
        Process(pid=1, ppid=None, arrival=0.0, kind="cpu_bound", label=0, total_cpu=500.0),
        Process(pid=2, ppid=None, arrival=50.0, kind="cpu_bound", label=0, total_cpu=500.0),
    ]
    for p in procs:
        p.burst_generator = lambda _now, _rng: 100.0

    sabre = SabreScheduler()
    sim = Simulator(scheduler=sabre, processes=procs, sim_end_ms=1000.0, ctx_switch_ms=0.0)
    res = sim.run()

    # Total CPU used plus overhead charged to clock should equal total sim_time (1000 ms)
    total_cpu_used = sum(p.cpu_used for p in res["processes"])
    assert pytest.approx(total_cpu_used + res["overhead_ms"]) == 1000.0


def test_hysteresis_bounds_level_flapping_on_noisy_benign_series():
    """Verify hysteresis limits the number of level transitions under noise."""
    scorer = RiskScorer(baselines=DEFAULT_BASELINES, up_hold=2, down_hold=3, down_margin=0.10)
    pid = 42

    rng = np.random.Generator(np.random.PCG64(12345))
    level_changes = 0
    curr_lvl = 0

    # Noisy series oscillating near threshold (mean 0.30 +/- 0.15)
    for step in range(500):
        noisy_share = float(np.clip(rng.normal(0.30, 0.15), 0.0, 1.0))
        f = Features(
            mean_burst=5.0,
            cv_burst=0.2,
            io_ratio=0.5,
            cpu_share=noisy_share,
            full_quantum_frac=0.0,
            fork_rate=0.0,
            n_descendants=0,
            burst_age_ms=5.0,
            slices_since_io=0,
        )
        scorer.update(pid, f, cls="interactive", now=step * 60.0)
        new_lvl = scorer.level(pid)
        if new_lvl != curr_lvl:
            level_changes += 1
            curr_lvl = new_lvl

    # Across 500 steps of noise, level transitions must be bounded (e.g. < 25 changes)
    assert level_changes < 25, f"Excessive level flapping: {level_changes} changes"


def test_all_flags_off_within_5_percent_of_rr():
    """Verify SABRE with all feature flags off matches RoundRobin within 5%."""
    all_false_flags = {
        "use_predictor": False,
        "use_classifier": False,
        "use_risk": False,
        "use_lineage": False,
        "use_cusum": False,
        "use_hysteresis": False,
        "use_trust": False,
        "use_elastic_cap": False,
        "use_aging": False,
    }

    gen = WorkloadGenerator(n_benign=40, arrival_rate_per_s=4.0)
    base = gen.generate(seed=42)

    procs_rr = clone_processes(base)
    sim_rr = Simulator(RoundRobin(10.0), procs_rr, sim_end_ms=60000.0)
    res_rr = sim_rr.run()
    avg_w_rr = sum(p.wait_time for p in res_rr["processes"]) / len(res_rr["processes"])

    procs_sabre = clone_processes(base)
    sim_sabre = Simulator(SabreScheduler(flags=all_false_flags), procs_sabre, sim_end_ms=60000.0)
    res_sabre = sim_sabre.run()
    avg_w_sabre = sum(p.wait_time for p in res_sabre["processes"]) / len(res_sabre["processes"])

    diff_pct = abs(avg_w_sabre - avg_w_rr) / avg_w_rr
    assert diff_pct < 0.05, f"Flags-off SABRE differed by {diff_pct*100:.2f}% from RR (expected < 5%)"

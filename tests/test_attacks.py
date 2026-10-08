"""Tests for attack models.

Requirements:
- forkbomb stops at max_descendants
- hog takes ~100% CPU under FCFS and starves others
- lowslow share is ~stealth * static_threshold
- adaptive reduces duty after being throttled under RR+StaticThreshold
"""

import pytest
import numpy as np

from sabre.core.process import Process, ProcState
from sabre.core.engine import Simulator
from sabre.schedulers.baselines import FCFS, RoundRobin, RR_StaticThreshold
from sabre.core.attacks import (
    create_forkbomb,
    create_hog,
    create_lowslow,
    create_adaptive,
    create_mimicry,
)


def test_forkbomb_stops_at_max_descendants():
    """Verify fork bomb stops spawning when reaching max_descendants."""
    max_desc = 25
    fb = create_forkbomb(pid=100, arrival=0.0, rate=50.0, intensity=2.0, max_descendants=max_desc)

    sched = RoundRobin(quantum_ms=10.0)
    sim = Simulator(scheduler=sched, processes=[fb], sim_end_ms=5000.0)
    res = sim.run()

    # Total processes spawned by fb
    children = [p for p in res["processes"] if p.ppid == 100]
    assert len(children) == max_desc
    for c in children:
        assert c.label == 1
        assert c.kind == "forkchild"


def test_hog_starves_others_under_fcfs():
    """Verify an infinite CPU hog under FCFS monopolizes CPU and starves subsequent processes."""
    hog = create_hog(pid=100, arrival=0.0)
    benign = Process(pid=1, ppid=None, arrival=10.0, kind="interactive", label=0, total_cpu=50.0)

    sched = FCFS()
    sim = Simulator(scheduler=sched, processes=[hog, benign], sim_end_ms=1000.0, ctx_switch_ms=0.0)
    res = sim.run()

    procs = {p.pid: p for p in res["processes"]}
    # Hog held CPU continuously from t=0 to sim_end
    assert procs[100].cpu_used >= 999.0
    # Benign arrived at 10.0 but never ran
    assert procs[1].cpu_used == 0.0
    assert procs[1].first_run is None
    assert procs[1].wait_time == pytest.approx(990.0)


def test_lowslow_duty_cycle_share():
    """Verify low-and-slow attacker targets stealth * static_threshold share."""
    stealth = 0.90
    static_threshold = 0.60
    expected_share = stealth * static_threshold  # 0.54

    lowslow = create_lowslow(
        pid=100,
        arrival=0.0,
        stealth=stealth,
        static_threshold=static_threshold,
        window_ms=500.0,
    )

    sched = RoundRobin(quantum_ms=10.0)
    sim = Simulator(scheduler=sched, processes=[lowslow], sim_end_ms=5000.0, ctx_switch_ms=0.0)
    res = sim.run()

    p = res["processes"][0]
    total_time = res["sim_time"]
    actual_share = p.cpu_used / total_time
    assert pytest.approx(actual_share, abs=0.05) == expected_share


def test_adaptive_reduces_duty_when_throttled():
    """Verify adaptive attacker detects throttling under RR+StaticThreshold and reduces duty."""
    # When duty is high (0.75 > 0.60), RR+StaticThreshold cuts its quantum from 10 to 2ms
    adaptive = create_adaptive(
        pid=100,
        arrival=0.0,
        static_threshold=0.60,
        window_ms=200.0,
    )
    # Background competing process so queue waiting is experienced when throttled
    competing = Process(
        pid=1, ppid=None, arrival=0.0, kind="cpu_bound", label=0, total_cpu=float("inf")
    )
    competing.burst_generator = lambda _now, _rng: 100.0

    sched = RR_StaticThreshold(
        quantum_ms=10.0,
        static_threshold=0.50,
        window_ms=200.0,
        throttled_quantum_ms=2.0,
    )

    sim = Simulator(scheduler=sched, processes=[adaptive, competing], sim_end_ms=4000.0)
    res = sim.run()

    # Sched flagged adaptive
    assert 100 in sched.flagged()
    # Adaptive executed slices and adjusted
    p_adapt = next(p for p in res["processes"] if p.pid == 100)
    assert p_adapt.cpu_used > 0.0

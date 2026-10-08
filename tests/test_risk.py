"""Tests for intel/risk.py and intel/calibrate.py.

Requirements per Workdoc 6.5 and Research Report F1:
1. False alarms on stationary benign series < 1%
2. Step increase in cpu_share triggers level >= 2 within 1 s
3. Slow ramp that never crosses tau_static triggers via CUSUM
4. Benign cpu_bound at its own normal share stays at level 0 (even with high CPU share)
5. Child of level-3 parent starts at level >= 2 via inherit()
6. Risk decays back to level 0 after behavior stops
7. Synthetic hog (never-ending burst) reaches level >= 2 via burst_age signal (F1)
"""

from __future__ import annotations

import pytest

from sabre.intel.features import Features
from sabre.intel.risk import RiskScorer, DEFAULT_BASELINES
from sabre.intel.calibrate import calibrate


def test_benign_cpu_bound_at_normal_share_stays_level_0():
    """Verify that a legitimate cpu_bound process running at its normal high share stays level 0."""
    scorer = RiskScorer(baselines=DEFAULT_BASELINES)
    pid = 1

    # cpu_bound process with share 0.90, no forks, normal bursts within p99
    f = Features(
        mean_burst=150.0,
        cv_burst=0.3,
        io_ratio=0.02,
        cpu_share=0.90,  # Matches mu_cpu_share = 0.90 for cpu_bound
        full_quantum_frac=0.9,
        fork_rate=0.0,
        n_descendants=0,
        burst_age_ms=100.0,  # Below p99 = 500.0
        slices_since_io=5,
    )

    for step in range(50):
        t = step * 60.0
        r = scorer.update(pid, f, cls="cpu_bound", now=t)
        lvl = scorer.level(pid)
        assert lvl == 0, f"False positive: cpu_bound flagged as level {lvl} at step {step}"
        assert r < 0.30


def test_synthetic_hog_reaches_level_2_or_higher():
    """Verify that a hog with a never-ending burst (burst_age_ms >> p99) reaches level >= 2."""
    scorer = RiskScorer(baselines=DEFAULT_BASELINES)
    pid = 100

    # Hog running continuously: burst_age climbs well above p99_burst_age (500ms)
    for step in range(25):
        t = step * 60.0
        age = 100.0 + step * 50.0  # Climbs from 100 up to 1350 ms
        f = Features(
            mean_burst=500.0,
            cv_burst=0.1,
            io_ratio=0.0,
            cpu_share=0.95,
            full_quantum_frac=1.0,
            fork_rate=0.0,
            n_descendants=0,
            burst_age_ms=age,
            slices_since_io=step + 1,
        )
        r = scorer.update(pid, f, cls="cpu_bound", now=t)

    assert scorer.level(pid) >= 2, f"Hog was not contained! Level: {scorer.level(pid)}, risk: {r}"
    assert "BURST_AGE" in scorer.reasons(pid)


def test_step_increase_triggers_level_2_within_1_second():
    """Verify an abrupt step increase in CPU share for an interactive process raises level >= 2 rapidly."""
    scorer = RiskScorer(baselines=DEFAULT_BASELINES)
    pid = 2

    # Sudden jump from 0.05 to 0.95 share on interactive class
    f_abuse = Features(
        mean_burst=2.0,
        cv_burst=0.1,
        io_ratio=0.1,
        cpu_share=0.95,  # mu for interactive is 0.05, jump creates large z_cpu
        full_quantum_frac=0.9,
        fork_rate=0.0,
        n_descendants=0,
        burst_age_ms=10.0,
        slices_since_io=1,
    )

    triggered_time = None
    for step in range(20):
        t = step * 60.0  # 60 ms intervals
        r = scorer.update(pid, f_abuse, cls="interactive", now=t)
        if scorer.level(pid) >= 2 and triggered_time is None:
            triggered_time = t

    assert triggered_time is not None
    assert triggered_time <= 1000.0, f"Triggered too slowly: {triggered_time} ms"


def test_slow_ramp_triggers_via_cusum():
    """Verify that a slow creeping increase in share triggers level >= 2 via CUSUM."""
    scorer = RiskScorer(baselines=DEFAULT_BASELINES)
    pid = 3

    # Gradually ramp from 0.05 to 0.25 (never crossing any global static threshold like 0.6)
    flagged = False
    for step in range(40):
        t = step * 60.0
        share = min(0.30, 0.05 + step * 0.007)
        f = Features(
            mean_burst=2.0,
            cv_burst=0.1,
            io_ratio=0.5,
            cpu_share=share,
            full_quantum_frac=0.0,
            fork_rate=0.0,
            n_descendants=0,
            burst_age_ms=5.0,
            slices_since_io=0,
        )
        scorer.update(pid, f, cls="interactive", now=t)
        if scorer.level(pid) >= 2:
            flagged = True
            break

    assert flagged, "CUSUM failed to detect slow drift!"
    assert "CUSUM" in scorer.reasons(pid)


def test_lineage_inheritance():
    """Verify child of a level-3 parent starts at level >= 2 upon inherit()."""
    scorer = RiskScorer(baselines=DEFAULT_BASELINES)
    parent_pid = 10
    child_pid = 20

    # Bring parent to level 3 with high fork rate and descendants
    f_parent = Features(
        mean_burst=10.0,
        cv_burst=0.1,
        io_ratio=0.0,
        cpu_share=0.5,
        full_quantum_frac=0.0,
        fork_rate=15.0,
        n_descendants=40,  # exceeds D_max = 30
        burst_age_ms=10.0,
        slices_since_io=0,
    )

    for step in range(10):
        scorer.update(parent_pid, f_parent, cls="interactive", now=step * 60.0)

    assert scorer.level(parent_pid) == 3

    # Child inherits from parent
    scorer.inherit(child_pid=child_pid, parent_pid=parent_pid)
    assert scorer.level(child_pid) >= 2
    assert "LINEAGE" in scorer.reasons(child_pid)


def test_risk_decays_back_to_level_0():
    """Verify that after abusive behavior stops, risk decays and level recovers to 0."""
    scorer = RiskScorer(baselines=DEFAULT_BASELINES)
    pid = 5

    # Trigger level 2
    f_abuse = Features(
        mean_burst=10.0,
        cv_burst=0.1,
        io_ratio=0.0,
        cpu_share=0.9,
        full_quantum_frac=0.9,
        fork_rate=0.0,
        n_descendants=0,
        burst_age_ms=10.0,
        slices_since_io=0,
    )
    for step in range(5):
        scorer.update(pid, f_abuse, cls="interactive", now=step * 60.0)

    assert scorer.level(pid) >= 2

    # Behavior ceases: normal interactive features
    f_normal = Features(
        mean_burst=2.0,
        cv_burst=0.1,
        io_ratio=0.8,
        cpu_share=0.05,
        full_quantum_frac=0.0,
        fork_rate=0.0,
        n_descendants=0,
        burst_age_ms=2.0,
        slices_since_io=0,
    )
    now = 500.0
    for _ in range(220):
        now += 60.0
        scorer.update(pid, f_normal, cls="interactive", now=now)

    assert scorer.level(pid) == 0


def test_stationary_benign_false_alarm_rate():
    """Verify false alarms (level >= 2) per 1000 updates < 1% on stationary benign series."""
    scorer = RiskScorer(baselines=DEFAULT_BASELINES)
    pid = 7

    false_alarms = 0
    total_updates = 1000

    import numpy as np
    rng = np.random.Generator(np.random.PCG64(42))

    for step in range(total_updates):
        # Sample normal interactive distribution
        share = float(np.clip(rng.normal(0.05, 0.02), 0.0, 1.0))
        f = Features(
            mean_burst=2.0,
            cv_burst=0.1,
            io_ratio=0.8,
            cpu_share=share,
            full_quantum_frac=0.0,
            fork_rate=0.0,
            n_descendants=0,
            burst_age_ms=2.0,
            slices_since_io=0,
        )
        scorer.update(pid, f, cls="interactive", now=step * 60.0)
        if scorer.level(pid) >= 2:
            false_alarms += 1

    fa_rate = false_alarms / total_updates
    assert fa_rate < 0.01, f"False alarm rate too high: {fa_rate:.4f}"

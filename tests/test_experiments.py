"""Tests for metrics, stats, and experiment execution (Step 8)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from sabre.core.process import Process, ProcState
from sabre.core.metrics import (
    compute_performance_metrics,
    compute_security_metrics,
    compute_comparative_metrics,
    compute_kaplan_meier,
)
from sabre.experiments.stats import (
    mean_ci_t,
    mean_ci_bootstrap,
    paired_wilcoxon,
    cliffs_delta,
    holm_bonferroni,
)


def test_performance_metrics():
    """Verify performance metrics calculation and Jain fairness."""
    p1 = Process(pid=1, ppid=None, arrival=0.0, kind="interactive", label=0, total_cpu=100.0)
    p1.state = ProcState.DONE
    p1.finish = 120.0
    p1.wait_time = 20.0
    p1.cpu_used = 100.0
    p1.response_samples = [2.0, 5.0, 10.0]

    p2 = Process(pid=2, ppid=None, arrival=10.0, kind="cpu_bound", label=0, total_cpu=200.0)
    p2.state = ProcState.DONE
    p2.finish = 300.0
    p2.wait_time = 90.0
    p2.cpu_used = 200.0

    sim_res = {"ctx_switches": 15, "overhead_ms": 1.5}
    perf = compute_performance_metrics([p1, p2], sim_res, sim_end_ms=1000.0)

    assert pytest.approx(perf["avg_wait"]) == (20.0 + 90.0) / 2.0
    assert pytest.approx(perf["avg_turnaround"]) == (120.0 + 290.0) / 2.0
    assert pytest.approx(perf["throughput"]) == 2.0  # 2 processes / 1.0 s
    assert perf["resp_p50"] == 5.0
    assert pytest.approx(perf["jain_fairness"]) == 1.0  # both got 100% of demand
    assert perf["ctx_switches"] == 15.0
    assert perf["overhead_ms"] == 1.5
    assert pytest.approx(perf["overhead_pct"]) == 0.15


def test_security_metrics_instance_grouping():
    """Verify Research Report Fix F3: fork bomb children do not inflate instance-level metrics."""
    # 10 benign processes
    procs: list[Process] = []
    for i in range(1, 11):
        p = Process(pid=i, ppid=None, arrival=0.0, kind="cpu_bound", label=0, total_cpu=100.0)
        p.cpu_used = 100.0
        procs.append(p)

    # 1 fork bomb root + 50 fork children (single attack instance)
    fb_root = Process(pid=100, ppid=None, arrival=1000.0, kind="forkbomb", label=1, total_cpu=500.0)
    fb_root.cpu_used = 50.0
    procs.append(fb_root)

    for c in range(1, 51):
        child = Process(pid=1000 + c, ppid=100, arrival=1010.0, kind="forkchild", label=1, total_cpu=20.0)
        child.cpu_used = 20.0
        procs.append(child)

    # Scenario: only 5 fork children are flagged at t=1050 (root not directly flagged)
    flagged = {1001: 1050.0, 1002: 1052.0}

    sec = compute_security_metrics(procs, flagged, attack_start_ms=1000.0, sim_end_ms=5000.0)

    # Process-level recall: 2 / 51 ~ 0.0392 (diluted by children)
    assert pytest.approx(sec["recall"]) == 2.0 / 51.0
    # Instance-level recall: 1 attack instance detected -> 1.0!
    assert sec["instance_recall"] == 1.0
    assert sec["instance_precision"] == 1.0
    assert sec["instance_fpr"] == 0.0
    assert sec["detection_rate"] == 1.0
    # TTD: earliest flag in tree (1050) - attack start (1000) = 50 ms
    assert pytest.approx(sec["time_to_detect"]) == 50.0


def test_kaplan_meier_censoring():
    """Verify Kaplan-Meier computation for censored detection."""
    events = [10.0, 20.0]
    censored = [30.0, 40.0]
    km = compute_kaplan_meier(events, censored)
    # Total subjects = 4
    # At t=10: 1 event out of 4 -> S = 1 * (1 - 1/4) = 0.75 -> F = 0.25
    # At t=20: 1 event out of 3 -> S = 0.75 * (1 - 1/3) = 0.50 -> F = 0.50
    # At t=30: censored -> F stays 0.50
    # At t=40: censored -> F stays 0.50
    t_vals = [t for (t, p) in km]
    p_vals = [p for (t, p) in km]
    assert 10.0 in t_vals
    assert 20.0 in t_vals
    assert pytest.approx(p_vals[1]) == 0.25
    assert pytest.approx(p_vals[2]) == 0.50


def test_comparative_metrics_lsi_adi():
    """Verify LSI and ADI definitions with epsilon guards (Fix F4)."""
    p_benign_att = Process(pid=1, ppid=None, arrival=0.0, kind="cpu_bound", label=0, total_cpu=100.0)
    p_benign_att.finish = 200.0  # turnaround = 200

    p_benign_rr = Process(pid=1, ppid=None, arrival=0.0, kind="cpu_bound", label=0, total_cpu=100.0)
    p_benign_rr.finish = 100.0  # turnaround = 100

    sched_attack_perf = {"resp_p95": 20.0, "avg_wait": 50.0}
    sched_no_attack_perf = {"resp_p95": 10.0, "avg_wait": 30.0}
    rr_no_attack_perf = {"resp_p95": 10.0, "avg_wait": 30.0}

    comp = compute_comparative_metrics(
        sched_attack_perf,
        sched_no_attack_perf,
        rr_no_attack_perf,
        [p_benign_att],
        [p_benign_rr],
    )

    # LSI = 200.0 / 100.0 = 2.0
    assert pytest.approx(comp["lsi"], rel=1e-2) == 2.0
    # ADI = 20.0 / 10.0 = 2.0
    assert pytest.approx(comp["adi"], rel=1e-2) == 2.0


def test_stats_ci_and_wilcoxon():
    """Verify t and bootstrap CIs and paired Wilcoxon."""
    data = [10.0, 12.0, 9.0, 11.0, 10.5, 11.5]
    m_t, l_t, u_t = mean_ci_t(data, confidence=0.95)
    assert l_t < m_t < u_t
    assert pytest.approx(m_t) == np.mean(data)

    m_b, l_b, u_b = mean_ci_bootstrap(data, confidence=0.95, seed=42)
    assert l_b < m_b < u_b
    assert pytest.approx(m_b) == np.mean(data)

    # Wilcoxon paired test (needs at least 6 pairs for p < 0.05 with two-sided test)
    x = [20, 22, 24, 25, 28, 30]
    y = [5, 6, 8, 9, 10, 11]
    stat, pval = paired_wilcoxon(x, y)
    assert pval < 0.05

    # Cliff's delta
    delta = cliffs_delta(x, y)
    assert delta == 1.0  # all x_i > y_j


def test_holm_bonferroni():
    """Verify Holm-Bonferroni step-down correction."""
    # 4 tests: raw p-values [0.01, 0.04, 0.03, 0.005]
    # Sorted: p(1)=0.005 (x4 = 0.02), p(2)=0.01 (x3 = 0.03), p(3)=0.03 (x2 = 0.06), p(4)=0.04 (x1 = 0.04 -> max=0.06)
    p_vals = [0.01, 0.04, 0.03, 0.005]
    results = holm_bonferroni(p_vals, alpha=0.05)
    
    # Check that lowest raw p-value has lowest adjusted p-value
    adj_p = [r[0] for r in results]
    sig = [r[1] for r in results]
    assert adj_p[3] == pytest.approx(0.02)  # 0.005 * 4
    assert sig[3] is True
    assert adj_p[0] == pytest.approx(0.03)  # 0.01 * 3
    assert sig[0] is True
    assert sig[1] is False


def test_sabre_flags_hog_detected():
    """Verify that a hog run under SABRE reaches level >= 2, is recorded in flagged(),

    and compute_security_metrics counts it as detected (recall == 1.0).
    """
    from sabre.core.attacks import create_hog
    from sabre.schedulers.sabre import SabreScheduler
    from sabre.core.engine import Simulator

    hog = create_hog(pid=501, arrival=100.0)
    sched = SabreScheduler()
    sim = Simulator(sched, [hog], sim_end_ms=10000.0, seed=42)
    sim.run()

    flagged = sched.flagged()
    assert 501 in flagged
    assert flagged[501] >= 100.0

    sec = compute_security_metrics(sim.all_processes, flagged, attack_start_ms=100.0, sim_end_ms=10000.0)
    assert sec["recall"] == 1.0
    assert sec["instance_recall"] == 1.0
    assert sec["fpr"] == 0.0


def test_adi_metric_under_attack():
    """Verify that ADI >= 1.0 when an attack hurts interactive response latency."""
    # Under attack, interactive 95th percentile response is degraded
    sched_no_attack = {"resp_p95": 20.0, "avg_wait": 100.0}
    sched_attack = {"resp_p95": 40.0, "avg_wait": 250.0}
    rr_no_attack = {"resp_p95": 25.0, "avg_wait": 110.0}

    comp = compute_comparative_metrics(
        sched_attack_metrics=sched_attack,
        sched_no_attack_metrics=sched_no_attack,
        rr_no_attack_metrics=rr_no_attack,
        sched_attack_procs=[],
        rr_no_attack_procs=[],
    )
    assert comp["adi"] >= 1.0
    assert pytest.approx(comp["adi"], rel=1e-3) == 2.0


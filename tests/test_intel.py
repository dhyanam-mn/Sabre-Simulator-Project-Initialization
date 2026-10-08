"""Tests for intel module (features, predictor, classifier).

Requirements:
- EMA converges to a constant burst
- Classifier returns "unknown" with fewer than 3 observed bursts
- Classifier labels all 4 kinds correctly on generated series
- fork_rate and n_descendants computed correctly
- F1 fields burst_age_ms and slices_since_io tracked correctly
- Costs are charged
"""

import math
import numpy as np
import pytest

from sabre.intel.features import Features, FeatureExtractor
from sabre.intel.predictor import BurstPredictor
from sabre.intel.classifier import BehaviourClassifier


def test_ema_converges_to_constant_burst():
    """Verify that EMA predictor converges to a steady constant burst value."""
    predictor = BurstPredictor(alpha=0.5, default_prior=10.0)
    pid = 1

    # Starting estimate is default prior
    initial = predictor.predict(pid)
    assert initial == 10.0
    assert predictor.overhead_ms() == 0.01

    target_burst = 50.0
    # Feed constant burst of 50.0
    for _ in range(30):
        predictor.update(pid, target_burst)

    pred = predictor.predict(pid)
    assert pytest.approx(pred, abs=1e-4) == target_burst


def test_classifier_unknown_with_fewer_than_3_bursts():
    """Verify classifier returns 'unknown' when fewer than 3 bursts have been observed."""
    classifier = BehaviourClassifier(min_bursts=3)
    f = Features(
        mean_burst=2.0,
        cv_burst=0.1,
        io_ratio=0.8,
        cpu_share=0.1,
        full_quantum_frac=0.0,
        fork_rate=0.0,
        n_descendants=0,
        burst_age_ms=0.0,
        slices_since_io=0,
    )

    # 0, 1, 2 bursts -> "unknown"
    assert classifier.classify(f, observed_bursts=0) == "unknown"
    assert classifier.classify(f, observed_bursts=1) == "unknown"
    assert classifier.classify(f, observed_bursts=2) == "unknown"

    # 3 bursts -> classified as "interactive"
    assert classifier.classify(f, observed_bursts=3) == "interactive"
    assert classifier.overhead_ms() > 0.0


def test_classifier_labels_all_4_kinds_correctly():
    """Verify rule-based classifier assigns all 4 canonical kinds correctly."""
    classifier = BehaviourClassifier(min_bursts=3)

    # 1. interactive: mean_burst < 5 and io_ratio > 0.6
    f_inter = Features(
        mean_burst=2.5,
        cv_burst=0.2,
        io_ratio=0.75,
        cpu_share=0.1,
        full_quantum_frac=0.0,
        fork_rate=0.0,
        n_descendants=0,
        burst_age_ms=0.0,
        slices_since_io=0,
    )
    assert classifier.classify(f_inter, observed_bursts=5) == "interactive"

    # 2. bursty: cv_burst > 1.0
    f_bursty = Features(
        mean_burst=25.0,
        cv_burst=1.4,
        io_ratio=0.2,
        cpu_share=0.3,
        full_quantum_frac=0.4,
        fork_rate=0.0,
        n_descendants=0,
        burst_age_ms=0.0,
        slices_since_io=0,
    )
    assert classifier.classify(f_bursty, observed_bursts=5) == "bursty"

    # 3. io_bound: io_ratio > 0.5 (and not interactive / bursty)
    f_io = Features(
        mean_burst=12.0,
        cv_burst=0.3,
        io_ratio=0.55,
        cpu_share=0.2,
        full_quantum_frac=0.1,
        fork_rate=0.0,
        n_descendants=0,
        burst_age_ms=0.0,
        slices_since_io=0,
    )
    assert classifier.classify(f_io, observed_bursts=5) == "io_bound"

    # 4. cpu_bound: io_ratio < 0.1 and full_quantum_frac > 0.8
    f_cpu = Features(
        mean_burst=120.0,
        cv_burst=0.2,
        io_ratio=0.02,
        cpu_share=0.95,
        full_quantum_frac=0.9,
        fork_rate=0.0,
        n_descendants=0,
        burst_age_ms=100.0,
        slices_since_io=10,
    )
    assert classifier.classify(f_cpu, observed_bursts=5) == "cpu_bound"


def test_fork_rate_and_n_descendants():
    """Verify FeatureExtractor computes fork_rate and n_descendants across process trees."""
    fe = FeatureExtractor(W=8, T_share_ms=500.0, cost_feature_ms=0.02)

    # Process 1 (root)
    fe.register_process(pid=1, ppid=None)

    # Root spawns 10 children over 500 ms
    for i in range(2, 12):
        fe.register_process(pid=i, ppid=1)
    fe.on_spawn(pid=1, ppid=None, count=10, now=100.0)

    # Child 2 spawns 5 grandchildren
    for j in range(12, 17):
        fe.register_process(pid=j, ppid=2)
    fe.on_spawn(pid=2, ppid=1, count=5, now=150.0)

    f1 = fe.extract(pid=1, now=200.0)
    # Total descendants under pid 1: 10 children + 5 grandchildren = 15
    assert f1.n_descendants == 15
    # fork_rate for root: 10 children spawned in window / 5.0 = 2.0 (children per 100 ms)
    assert pytest.approx(f1.fork_rate) == 2.0

    f2 = fe.extract(pid=2, now=200.0)
    # Descendants under child 2: 5 grandchildren
    assert f2.n_descendants == 5
    # fork_rate for child 2: 5 spawned / 5.0 = 1.0
    assert pytest.approx(f2.fork_rate) == 1.0


def test_burst_age_and_slices_since_io():
    """Verify F1 additions: burst_age_ms and slices_since_io update and reset appropriately."""
    fe = FeatureExtractor(W=8, T_share_ms=500.0, cost_feature_ms=0.02)
    pid = 42
    fe.register_process(pid=pid, ppid=None)

    # 3 slices of 10 ms with no I/O, burst not finished
    fe.on_slice_end(pid=pid, ran=10.0, burst_done=False, quantum=10.0, now=10.0)
    fe.on_slice_end(pid=pid, ran=10.0, burst_done=False, quantum=10.0, now=20.0)
    fe.on_slice_end(pid=pid, ran=10.0, burst_done=False, quantum=10.0, now=30.0)

    f = fe.extract(pid=pid, now=30.0)
    assert pytest.approx(f.burst_age_ms) == 30.0
    assert f.slices_since_io == 3
    assert f.full_quantum_frac == 1.0

    # Process completes burst and does I/O
    fe.on_slice_end(pid=pid, ran=5.0, burst_done=True, quantum=10.0, now=35.0)
    fe.on_io(pid=pid, io_wait_ms=25.0)

    f_after_io = fe.extract(pid=pid, now=60.0)
    assert pytest.approx(f_after_io.burst_age_ms) == 0.0
    assert f_after_io.slices_since_io == 0
    assert f_after_io.io_ratio > 0.0

    # Verify overhead charging
    assert fe.overhead_ms() > 0.0
    assert fe.overhead_ms() == 0.0  # reset after call

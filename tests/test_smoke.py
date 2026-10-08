"""Smoke tests importing all modules and validating dataclass fields."""

import dataclasses
import pytest

import sabre
import sabre.config
import sabre.core
import sabre.core.process
import sabre.schedulers
import sabre.schedulers.base
import sabre.intel
import sabre.intel.features
import sabre.intel.predictor
import sabre.intel.classifier
import sabre.intel.risk
import sabre.experiments
import sabre.dashboard


def test_imports():
    """Verify all skeleton modules are importable."""
    assert sabre is not None
    assert sabre.config is not None
    assert sabre.core.process.Process is not None
    assert sabre.schedulers.base.Scheduler is not None
    assert sabre.intel.features.Features is not None
    assert sabre.intel.predictor.BurstPredictor is not None
    assert sabre.intel.classifier.BehaviourClassifier is not None
    assert sabre.intel.risk.RiskScorer is not None


def test_features_has_f1_fields():
    """Verify Features dataclass has the required F1 additions: burst_age_ms and slices_since_io."""
    fields = {f.name: f.type for f in dataclasses.fields(sabre.intel.features.Features)}

    # Section 3 fields
    assert "mean_burst" in fields
    assert "cv_burst" in fields
    assert "io_ratio" in fields
    assert "cpu_share" in fields
    assert "full_quantum_frac" in fields
    assert "fork_rate" in fields
    assert "n_descendants" in fields

    # F1 additions
    assert "burst_age_ms" in fields, "burst_age_ms missing from Features"
    assert "slices_since_io" in fields, "slices_since_io missing from Features"


def test_config_loader():
    """Verify loading default config and seeds."""
    cfg = sabre.config.load_config()
    assert "sim" in cfg
    assert "sabre" in cfg
    assert "costs" in cfg

    seeds = sabre.config.load_seeds()
    assert "tuning" in seeds
    assert "calibration" in seeds
    assert "eval" in seeds

    tuning_set = set(seeds["tuning"])
    calib_set = set(seeds["calibration"])
    eval_set = set(seeds["eval"])

    # Disjointness checks
    assert len(tuning_set & calib_set) == 0
    assert len(tuning_set & eval_set) == 0
    assert len(calib_set & eval_set) == 0
    assert len(eval_set) == 30
    assert len(tuning_set) == 20
    assert len(calib_set) == 20

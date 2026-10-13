"""SABRE: Security- And Behaviour-aware, Risk-Elastic CPU Scheduler.

Follows Workdoc Section 7.4, 7.5, Section 3, and Research Report additions:
- Integrates FeatureExtractor, BurstPredictor, BehaviourClassifier, RiskScorer, and SabreController.
- Supports all feature flags in `flags` for ablation studies.
- Maintains explainability `decision_log()` with reason codes (N7).
- Tracks `flagged()` timestamps for level >= 2 processes.
- Charges real monitoring overhead into `overhead_ms()`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from sabre.core.process import Process
from sabre.schedulers.base import Scheduler
from sabre.schedulers.controller import SabreController, ControllerDecision
from sabre.intel.features import FeatureExtractor, Features
from sabre.intel.predictor import BurstPredictor
from sabre.intel.classifier import BehaviourClassifier
from sabre.intel.risk import RiskScorer, DEFAULT_BASELINES


DEFAULT_FLAGS = {
    "use_predictor": True,
    "use_classifier": True,
    "use_risk": True,
    "use_lineage": True,
    "use_cusum": True,
    "use_hysteresis": True,
    "use_trust": True,
    "use_elastic_cap": True,
    "use_aging": True,
    "use_burst_age": True,
}


class SabreScheduler(Scheduler):
    """SABRE main scheduler."""

    name = "SABRE"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        flags: dict[str, bool] | None = None,
        baselines: dict[str, dict[str, float]] | None = None,
    ) -> None:
        cfg = config or {}
        sabre_cfg = cfg.get("sabre", {})
        feat_cfg = cfg.get("features", {})
        pred_cfg = cfg.get("predictor", {})
        risk_cfg = cfg.get("risk", {})
        costs_cfg = cfg.get("costs", {})

        self.flags = dict(DEFAULT_FLAGS)
        if flags is not None:
            self.flags.update(flags)
        if "flags" in sabre_cfg:
            self.flags.update(sabre_cfg["flags"])

        # 1. Feature extractor
        self.features = FeatureExtractor(
            W=feat_cfg.get("W", 8),
            T_share_ms=feat_cfg.get("T_share_ms", 500.0),
            cost_feature_ms=costs_cfg.get("feature", 0.02),
        )

        # 2. Predictor
        self.predictor = BurstPredictor(
            alpha=pred_cfg.get("alpha", 0.5),
            priors=pred_cfg.get("priors"),
            cost_ms=costs_cfg.get("ema", 0.01),
        )

        # 3. Classifier
        self.classifier = BehaviourClassifier(
            cost_ms=costs_cfg.get("classify", 0.01),
        )

        if baselines is None:
            baselines_file = Path(__file__).resolve().parent.parent.parent / "configs" / "baselines.json"
            if baselines_file.exists():
                try:
                    with baselines_file.open("r", encoding="utf-8") as f:
                        baselines = json.load(f)
                except Exception:
                    baselines = None

        # 4. Risk scorer
        self.risk_scorer = RiskScorer(
            baselines=baselines,
            Z_max=risk_cfg.get("Z_max", 4.0),
            weights=risk_cfg.get("weights"),
            decay_lambda=risk_cfg.get("lambda", 0.97),
            lineage_gamma=risk_cfg.get("gamma", 0.90) if self.flags["use_lineage"] else 0.0,
            D_max=risk_cfg.get("D_max", 30),
            cusum_k_sigma=risk_cfg.get("cusum_k_sigma", 0.5),
            cusum_h_sigma=risk_cfg.get("cusum_h_sigma", 5.0) if self.flags["use_cusum"] else 1e9,
            level_thresholds=risk_cfg.get("level_thresholds", [0.30, 0.55, 0.80]),
            up_hold=risk_cfg.get("up_hold", 2) if self.flags["use_hysteresis"] else 1,
            down_hold=risk_cfg.get("down_hold", 3) if self.flags["use_hysteresis"] else 1,
            down_margin=risk_cfg.get("down_margin", 0.10) if self.flags["use_hysteresis"] else 0.0,
            trust_gain=risk_cfg.get("trust_gain", 0.01) if self.flags["use_trust"] else 0.0,
            trust_cap=risk_cfg.get("trust_cap", 0.5) if self.flags["use_trust"] else 0.0,
            update_spacing_ms=feat_cfg.get("update_spacing_ms", 50.0),
            cost_risk_ms=costs_cfg.get("risk", 0.02),
            weight_age=risk_cfg.get("weight_age", 0.90),
            scale_age=risk_cfg.get("scale_age", 200.0),
            use_burst_age=self.flags.get("use_burst_age", True),
        )

        self.classifier_noise_rate = float(cfg.get("classifier_noise_rate", 0.0))
        self.feature_noise_rate = float(cfg.get("feature_noise_rate", 0.0))
        self.noise_rng = np.random.Generator(np.random.PCG64(int(cfg.get("seed", 42))))

        # 5. Controller
        self.controller = SabreController(
            q_base=sabre_cfg.get("q_base", 10.0),
            q_min=sabre_cfg.get("q_min", 1.0),
            q_max=sabre_cfg.get("q_max", 40.0),
            short_thr=sabre_cfg.get("short_thr", 5.0),
            long_thr=sabre_cfg.get("long_thr", 50.0),
            pred_ref=sabre_cfg.get("pred_ref", 10.0),
            age_ms=sabre_cfg.get("age_ms", 200.0),
            slot_period=sabre_cfg.get("slot_period", 10),
            caps=sabre_cfg.get("caps", [1.0, 0.5, 0.25, 0.10]),
            cap_window_ms=sabre_cfg.get("cap_window_ms", 200.0),
            min_share=sabre_cfg.get("min_share", 0.02),
            cost_ms=costs_cfg.get("decision", 0.01),
            use_predictor=self.flags["use_predictor"],
            use_classifier=self.flags["use_classifier"],
            use_aging=self.flags["use_aging"],
            use_elastic_cap=self.flags["use_elastic_cap"],
        )

        # State tracking
        self.proc_pred: dict[int, float] = {}
        self.proc_class: dict[int, str] = {}
        self.proc_level: dict[int, int] = {}
        self.proc_reasons: dict[int, list[str]] = {}

        self._flagged: dict[int, float] = {}
        self._decision_log: list[dict[str, Any]] = []

    def on_arrival(self, p: Process, now: float) -> None:
        """Handle process arrival or fork spawn."""
        self.features.register_process(p.pid, p.ppid)
        if p.ppid is not None:
            if self.features.records.get(p.ppid):
                self.features.on_spawn(p.ppid, None, 1, now)
                # Immediately evaluate parent's fork activity to update parent's risk
                if self.flags["use_risk"]:
                    f_parent = self.features.extract(p.ppid, now)
                    self.risk_scorer.update(p.ppid, f_parent, self.proc_class.get(p.ppid, "unknown"), now)
                    p_lvl = self.risk_scorer.level(p.ppid)
                    self.proc_level[p.ppid] = p_lvl
                    self.proc_reasons[p.ppid] = self.risk_scorer.reasons(p.ppid)
                    if p_lvl >= 2 and p.ppid not in self._flagged:
                        self._flagged[p.ppid] = now
            if self.flags["use_lineage"]:
                self.risk_scorer.inherit(p.pid, p.ppid)

        self.predictor.init_process(p.pid, p.kind)
        self.proc_pred[p.pid] = self.predictor.predict(p.pid)
        self.proc_class[p.pid] = "unknown"
        lvl = self.risk_scorer.level(p.pid) if self.flags["use_risk"] else 0
        self.proc_level[p.pid] = lvl
        self.proc_reasons[p.pid] = self.risk_scorer.reasons(p.pid) if self.flags["use_risk"] else []

        if lvl >= 2 and p.pid not in self._flagged:
            self._flagged[p.pid] = now

        self.controller.enqueue(p, self.proc_class[p.pid], self.proc_pred[p.pid], lvl)

    def on_ready(self, p: Process, now: float) -> None:
        """Handle process returning to ready state."""
        lvl = self.proc_level.get(p.pid, 0)
        pred = self.proc_pred.get(p.pid, 10.0)
        cls_name = self.proc_class.get(p.pid, "unknown")
        self.controller.enqueue(p, cls_name, pred, lvl)

    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None:
        """Update telemetry, intelligence modules, and risk state upon slice completion."""
        self.controller.record_slice(p.pid, ran, now)

        # Update features
        quantum = self.controller.q_base
        self.features.on_slice_end(p.pid, ran, burst_done, quantum, now, p.ppid)
        f = self.features.extract(p.pid, now)

        # Update predictor if burst ended
        if burst_done:
            self.predictor.update(p.pid, ran)
            self.classifier.record_burst(p.pid)
            self.proc_pred[p.pid] = self.predictor.predict(p.pid)

        # Update classifier
        if self.flags["use_classifier"]:
            pred_cls = self.classifier.classify(f, pid=p.pid)
            if self.classifier_noise_rate > 0.0 and self.noise_rng.random() < self.classifier_noise_rate:
                other_classes = [c for c in ["interactive", "io_bound", "cpu_bound", "bursty"] if c != pred_cls]
                pred_cls = str(self.noise_rng.choice(other_classes))
            self.proc_class[p.pid] = pred_cls
        else:
            self.proc_class[p.pid] = "unknown"

        # Apply feature noise if configured
        f_risk = f
        if self.feature_noise_rate > 0.0:
            jitter = float(self.noise_rng.normal(0.0, self.feature_noise_rate))
            f_risk = Features(
                cpu_share=float(np.clip(f.cpu_share * (1.0 + jitter), 0.0, 1.0)),
                fork_rate=float(max(0.0, f.fork_rate * (1.0 + jitter))),
                n_descendants=f.n_descendants,
                mean_burst=f.mean_burst,
                cv_burst=f.cv_burst,
                io_ratio=f.io_ratio,
                full_quantum_frac=f.full_quantum_frac,
                burst_age_ms=float(max(0.0, f.burst_age_ms * (1.0 + jitter))),
                slices_since_io=f.slices_since_io,
            )

        # Update risk
        if self.flags["use_risk"]:
            self.risk_scorer.update(p.pid, f_risk, self.proc_class[p.pid], now)
            lvl = self.risk_scorer.level(p.pid)
            self.proc_level[p.pid] = lvl
            self.proc_reasons[p.pid] = self.risk_scorer.reasons(p.pid)
            if lvl >= 2 and p.pid not in self._flagged:
                self._flagged[p.pid] = now
        else:
            self.proc_level[p.pid] = 0
            self.proc_reasons[p.pid] = []

    def on_exit(self, p: Process, now: float) -> None:
        self.controller.remove_process(p)

    def pick_next(self, now: float) -> tuple[Process, float] | None:
        decision = self.controller.select_next(
            now=now,
            pred_map=self.proc_pred,
            class_map=self.proc_class,
            level_map=self.proc_level,
            reasons_map=self.proc_reasons,
        )
        if decision is None:
            return None

        # Log decision for explainability (N7)
        self._decision_log.append(
            {
                "t": now,
                "pid": decision.process.pid,
                "queue": decision.queue_name,
                "quantum": decision.quantum,
                "level": decision.level,
                "class": decision.proc_class,
                "pred": decision.predicted_burst,
                "reasons": decision.reasons,
            }
        )

        return decision.process, decision.quantum

    def overhead_ms(self) -> float:
        """Aggregate and reset monitoring overhead across all intelligence subcomponents."""
        total = (
            self.features.overhead_ms()
            + self.predictor.overhead_ms()
            + self.classifier.overhead_ms()
            + self.risk_scorer.overhead_ms()
            + self.controller.overhead_ms()
        )
        return total

    def flagged(self) -> dict[int, float]:
        return dict(self._flagged)

    def decision_log(self) -> list[dict[str, Any]]:
        return list(self._decision_log)

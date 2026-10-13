"""Process risk scoring, containment ladder, and lineage tracking.

Follows Workdoc Section 6.5, Section 3, and Research Report F1:
- Class-conditional risk scoring against class baseline mu/sigma
- Signals:
  1. s_fork: clip(z_fork / Z_max, 0, 1)
  2. s_cpu: clip(z_cpu / Z_max, 0, 1)
  3. s_drift: CUSUM on cpu_share, clip(S_t / h, 0, 1)
  4. s_age (F1): clip((burst_age_ms - p99_class) / scale, 0, 1) to separate hogs from cpu_bound
- Tree budget: if n_descendants > D_max, root's s_fork = 1.0
- Noisy-OR combination:
  inst = 1 - (1 - w_fork*s_fork)(1 - w_cpu*s_cpu)(1 - w_drift*s_drift)(1 - w_age*s_age)
- Temporal decay: R_t = max(inst, lambda * R_{t-1})
- Lineage inheritance: R_eff(p) = max(R(p), gamma * R_eff(parent))
- Trust discount: grows by trust_gain when R < 0.2 (cap 0.5), resets upon level >= 2
- Containment ladder with hysteresis:
  Thresholds [0.30, 0.55, 0.80]
  Up-hold: 2 consecutive updates above threshold to upgrade
  Down-hold: 3 consecutive updates below (threshold - down_margin) to downgrade
- Charged cost: cost_risk = 0.02 ms per update
"""

from __future__ import annotations

from typing import Any
import numpy as np

from sabre.intel.features import Features


DEFAULT_BASELINES: dict[str, dict[str, float]] = {
    "interactive": {
        "mu_cpu_share": 0.05,
        "sigma_cpu_share": 0.05,
        "mu_fork_rate": 0.0,
        "sigma_fork_rate": 0.1,
        "p99_burst_age": 15.0,
    },
    "io_bound": {
        "mu_cpu_share": 0.10,
        "sigma_cpu_share": 0.08,
        "mu_fork_rate": 0.0,
        "sigma_fork_rate": 0.1,
        "p99_burst_age": 30.0,
    },
    "cpu_bound": {
        "mu_cpu_share": 0.95,
        "sigma_cpu_share": 0.15,
        "mu_fork_rate": 0.0,
        "sigma_fork_rate": 0.1,
        "p99_burst_age": 500.0,
    },
    "bursty": {
        "mu_cpu_share": 0.75,
        "sigma_cpu_share": 0.25,
        "mu_fork_rate": 0.0,
        "sigma_fork_rate": 0.1,
        "p99_burst_age": 400.0,
    },
    "unknown": {
        "mu_cpu_share": 0.85,
        "sigma_cpu_share": 0.20,
        "mu_fork_rate": 0.0,
        "sigma_fork_rate": 0.1,
        "p99_burst_age": 500.0,
    },
}


class _ProcRiskState:
    def __init__(self, pid: int, parent_pid: int | None = None) -> None:
        self.pid = pid
        self.parent_pid = parent_pid
        self.last_update_time: float = -1e9
        self.raw_risk: float = 0.0
        self.eff_risk: float = 0.0
        self.final_risk: float = 0.0
        self.cusum_s: float = 0.0
        self.trust: float = 0.0
        self.level: int = 0
        self.up_counts: list[int] = [0, 0, 0]  # consecutive counts for levels 1, 2, 3
        self.down_counts: list[int] = [0, 0, 0]
        self.reasons: list[str] = []


class RiskScorer:
    """Class-conditional, lineage-aware, and drift-detecting risk scorer."""

    def __init__(
        self,
        baselines: dict[str, dict[str, float]] | None = None,
        Z_max: float = 4.0,
        weights: list[float] | None = None,
        weight_age: float = 0.90,
        scale_age: float = 200.0,
        decay_lambda: float = 0.97,
        lineage_gamma: float = 0.90,
        D_max: int = 30,
        cusum_k_sigma: float = 0.5,
        cusum_h_sigma: float = 5.0,
        level_thresholds: list[float] | None = None,
        up_hold: int = 2,
        down_hold: int = 3,
        down_margin: float = 0.10,
        trust_gain: float = 0.01,
        trust_cap: float = 0.50,
        update_spacing_ms: float = 50.0,
        cost_risk_ms: float = 0.02,
        use_burst_age: bool = True,
    ) -> None:
        self.baselines = baselines if baselines is not None else DEFAULT_BASELINES
        self.Z_max = float(Z_max)
        # weights: [w_fork, w_cpu, w_drift]
        self.weights = list(weights) if weights is not None else [0.90, 0.70, 0.80]
        self.weight_age = float(weight_age)
        self.scale_age = float(scale_age)
        self.use_burst_age = bool(use_burst_age)
        self.decay_lambda = float(decay_lambda)
        self.lineage_gamma = float(lineage_gamma)
        self.D_max = int(D_max)
        self.cusum_k_sigma = float(cusum_k_sigma)
        self.cusum_h_sigma = float(cusum_h_sigma)
        self.level_thresholds = list(level_thresholds) if level_thresholds is not None else [0.30, 0.55, 0.80]
        self.up_hold = int(up_hold)
        self.down_hold = int(down_hold)
        self.down_margin = float(down_margin)
        self.trust_gain = float(trust_gain)
        self.trust_cap = float(trust_cap)
        self.update_spacing_ms = float(update_spacing_ms)
        self.cost_risk_ms = float(cost_risk_ms)

        self.proc_states: dict[int, _ProcRiskState] = {}
        self._accrued_cost_ms: float = 0.0

    def _get_or_create(self, pid: int) -> _ProcRiskState:
        if pid not in self.proc_states:
            self.proc_states[pid] = _ProcRiskState(pid)
        return self.proc_states[pid]

    def inherit(self, child_pid: int, parent_pid: int) -> None:
        """Inherit risk state and lineage from parent to child."""
        parent_st = self._get_or_create(parent_pid)
        child_st = self._get_or_create(child_pid)
        child_st.parent_pid = parent_pid
        # Child's initial risk equals gamma * R_eff(parent)
        child_init_risk = self.lineage_gamma * parent_st.eff_risk
        child_st.raw_risk = max(child_st.raw_risk, child_init_risk)
        child_st.eff_risk = max(child_st.eff_risk, child_init_risk)
        child_st.final_risk = child_st.eff_risk
        if child_st.final_risk >= 0.3:
            child_st.reasons.append("LINEAGE")
        # Initialize level
        for lvl_idx, thr in reversed(list(enumerate(self.level_thresholds))):
            if child_st.final_risk >= thr:
                child_st.level = max(child_st.level, lvl_idx + 1)
                break

    def update(self, pid: int, f: Features, cls: str, now: float) -> float:
        """Update risk score for process `pid` given features, class, and timestamp.

        Returns:
            Risk score in [0, 1].
        """
        self._accrued_cost_ms += self.cost_risk_ms
        st = self._get_or_create(pid)

        # Minimum spacing between updates
        if now - st.last_update_time < self.update_spacing_ms:
            return st.final_risk
        st.last_update_time = now

        base = self.baselines.get(cls, self.baselines.get("unknown", DEFAULT_BASELINES["unknown"]))
        mu_cpu = base.get("mu_cpu_share", 0.3)
        sigma_cpu = max(0.01, base.get("sigma_cpu_share", 0.1))
        mu_fork = base.get("mu_fork_rate", 0.0)
        sigma_fork = max(0.01, base.get("sigma_fork_rate", 0.1))
        p99_age = base.get("p99_burst_age", 500.0)

        # 1. Rate signals
        z_cpu = (f.cpu_share - mu_cpu) / sigma_cpu
        z_fork = (f.fork_rate - mu_fork) / sigma_fork
        s_cpu = float(np.clip(z_cpu / self.Z_max, 0.0, 1.0))
        s_fork = float(np.clip(z_fork / self.Z_max, 0.0, 1.0))

        # Descendants budget
        if f.n_descendants > self.D_max:
            s_fork = 1.0

        # 2. CUSUM drift detection on cpu_share
        k = self.cusum_k_sigma * sigma_cpu
        h = self.cusum_h_sigma * sigma_cpu
        st.cusum_s = max(0.0, st.cusum_s + (f.cpu_share - mu_cpu - k))
        s_drift = float(np.clip(st.cusum_s / max(1e-4, h), 0.0, 1.0))

        # 3. Burst age signal (F1): separates hog from benign cpu_bound
        if self.use_burst_age:
            s_age = float(np.clip((f.burst_age_ms - p99_age) / self.scale_age, 0.0, 1.0))
        else:
            s_age = 0.0

        # Reasons tracking
        reasons: list[str] = []
        if s_fork > 0.3:
            reasons.append("FORK_RATE" if f.n_descendants <= self.D_max else "DESCENDANT_BUDGET")
        if s_cpu > 0.3:
            reasons.append("CPU_SHARE")
        if s_drift > 0.3:
            reasons.append("CUSUM")
        if s_age > 0.3:
            reasons.append("BURST_AGE")
        st.reasons = reasons

        # 4. Noisy-OR combination
        w_fork = self.weights[0]
        w_cpu = self.weights[1]
        w_drift = self.weights[2]
        w_age = self.weight_age

        p_benign = (
            (1.0 - w_fork * s_fork)
            * (1.0 - w_cpu * s_cpu)
            * (1.0 - w_drift * s_drift)
            * (1.0 - w_age * s_age)
        )
        inst = float(np.clip(1.0 - p_benign, 0.0, 1.0))

        # 5. Temporal decay
        st.raw_risk = max(inst, self.decay_lambda * st.raw_risk)

        # 6. Lineage inheritance
        p_eff = 0.0
        if st.parent_pid is not None and st.parent_pid in self.proc_states:
            p_eff = self.proc_states[st.parent_pid].eff_risk
        st.eff_risk = max(st.raw_risk, self.lineage_gamma * p_eff)
        if self.lineage_gamma * p_eff > st.raw_risk and "LINEAGE" not in st.reasons:
            st.reasons.append("LINEAGE")

        # 7. Trust discount
        if st.raw_risk < 0.20 and inst < 0.20:
            st.trust = min(self.trust_cap, st.trust + self.trust_gain)
        else:
            # Immediate forfeit of trust upon suspicious burst/activity
            st.trust = 0.0

        st.final_risk = float(np.clip(st.eff_risk * (1.0 - st.trust), 0.0, 1.0))

        # 8. Containment ladder with hysteresis
        curr_lvl = st.level
        target_lvl = 0
        for i, thr in enumerate(self.level_thresholds):
            if st.final_risk >= thr:
                target_lvl = i + 1

        if target_lvl > curr_lvl:
            # Upgrade requires up_hold consecutive updates
            st.up_counts[target_lvl - 1] += 1
            st.down_counts = [0, 0, 0]
            if st.up_counts[target_lvl - 1] >= self.up_hold:
                st.level = target_lvl
                if st.level >= 2:
                    st.trust = 0.0  # reset trust on level >= 2
        elif target_lvl < curr_lvl:
            # Downgrade requires R < threshold - down_margin for down_hold consecutive updates
            threshold = self.level_thresholds[curr_lvl - 1]
            if st.final_risk < threshold - self.down_margin:
                st.down_counts[curr_lvl - 1] += 1
                st.up_counts = [0, 0, 0]
                if st.down_counts[curr_lvl - 1] >= self.down_hold:
                    st.level = curr_lvl - 1
                    if st.level == 0:
                        st.cusum_s *= 0.5  # reset CUSUM partially so recovery is possible
            else:
                st.down_counts = [0, 0, 0]
        else:
            st.up_counts = [0, 0, 0]
            st.down_counts = [0, 0, 0]

        return st.final_risk

    def level(self, pid: int) -> int:
        """Return containment level (0..3) for process `pid`."""
        if pid not in self.proc_states:
            return 0
        return self.proc_states[pid].level

    def reasons(self, pid: int) -> list[str]:
        """Return explainability reason codes for process `pid`."""
        if pid not in self.proc_states:
            return []
        return list(self.proc_states[pid].reasons)

    def overhead_ms(self) -> float:
        """Return monitoring cost accrued since last call, then reset."""
        cost = self._accrued_cost_ms
        self._accrued_cost_ms = 0.0
        return cost

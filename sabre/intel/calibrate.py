"""Calibration script to learn class baselines from benign-only runs.

Follows Workdoc Section 6.4 and Research Report F1 & F2:
- Runs benign-only simulations on calibration seeds (200-219)
- Collects per-class:
  * mu and sigma of cpu_share
  * mu and sigma of fork_rate
  * p99 of burst_age_ms (F1: to separate hogs from legitimate cpu_bound)
- Uses RoundRobin until SABRE is implemented (with TODO flag use_sabre_calib).
- Saves learned baselines to configs/baselines.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from sabre.config import load_seeds
from sabre.core.workload import WorkloadGenerator, clone_processes
from sabre.core.engine import Simulator
from sabre.schedulers.baselines import RoundRobin
from sabre.intel.features import FeatureExtractor


BASELINES_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "baselines.json"


def calibrate(
    seeds: list[int] | None = None,
    n_benign: int = 60,
    sim_end_ms: float = 60000.0,
    output_path: str | Path | None = None,
    use_sabre: bool = False,
) -> dict[str, dict[str, float]]:
    """Calibrate class baseline parameters on benign workloads."""
    # TODO (F2): When SABRE is implemented, support calibrate under SABRE with use_risk=False
    if seeds is None:
        seed_cfg = load_seeds()
        seeds = seed_cfg.get("calibration", list(range(200, 220)))

    # Per-class accumulators: class -> list of values
    class_cpu_shares: dict[str, list[float]] = {
        "interactive": [],
        "io_bound": [],
        "cpu_bound": [],
        "bursty": [],
    }
    class_fork_rates: dict[str, list[float]] = {
        "interactive": [],
        "io_bound": [],
        "cpu_bound": [],
        "bursty": [],
    }
    class_burst_ages: dict[str, list[float]] = {
        "interactive": [],
        "io_bound": [],
        "cpu_bound": [],
        "bursty": [],
    }

    gen = WorkloadGenerator(n_benign=n_benign, arrival_rate_per_s=4.0)

    for seed in seeds:
        procs = gen.generate(seed=seed)
        # Using RoundRobin for baseline observation (or SABRE with risk=False when available)
        sched = RoundRobin(quantum_ms=10.0)
        sim = Simulator(scheduler=sched, processes=procs, sim_end_ms=sim_end_ms)

        fe = FeatureExtractor(W=8, T_share_ms=500.0)

        # Instrument scheduler on_slice_end and on_ready to extract features
        orig_slice_end = sched.on_slice_end

        def wrap_slice_end(p: Any, ran: float, b_done: bool, now: float) -> None:
            fe.on_slice_end(p.pid, ran, b_done, 10.0, now, p.ppid)
            orig_slice_end(p, ran, b_done, now)
            # Sample features periodically
            if p.kind in class_cpu_shares:
                f = fe.extract(p.pid, now)
                class_cpu_shares[p.kind].append(f.cpu_share)
                class_fork_rates[p.kind].append(f.fork_rate)
                class_burst_ages[p.kind].append(f.burst_age_ms)

        sched.on_slice_end = wrap_slice_end
        sim.run()

    baselines: dict[str, dict[str, float]] = {}
    for cls in ["interactive", "io_bound", "cpu_bound", "bursty"]:
        shares = class_cpu_shares[cls] or [0.1]
        forks = class_fork_rates[cls] or [0.0]
        ages = class_burst_ages[cls] or [10.0]

        mu_cpu = float(np.mean(shares))
        sigma_cpu = float(max(0.02, np.std(shares)))

        mu_fork = float(np.mean(forks))
        sigma_fork = float(max(0.02, np.std(forks)))

        p99_age = float(np.percentile(ages, 99.0)) if len(ages) >= 10 else float(np.max(ages))

        # Ensure safety margins on p99 burst age and cpu_share baseline (Workdoc 6.4-6.5 and Research Report F1)
        # to reflect that CPU-bound and bursty workloads legitimately consume high CPU shares when active,
        # separating attackers primarily via sustained burst age (s_age) and excessive unconstrained rate.
        if cls == "cpu_bound":
            calib_mu = max(0.95, mu_cpu)
            calib_sigma = max(0.15, sigma_cpu)
            safe_p99_age = max(p99_age * 1.5, 500.0)
        elif cls == "bursty":
            calib_mu = max(0.75, mu_cpu)
            calib_sigma = max(0.25, sigma_cpu)
            safe_p99_age = max(p99_age * 1.5, 400.0)
        else:
            calib_mu = mu_cpu
            calib_sigma = max(0.05, sigma_cpu)
            safe_p99_age = max(p99_age * 1.5, 30.0)

        baselines[cls] = {
            "mu_cpu_share": round(calib_mu, 4),
            "sigma_cpu_share": round(calib_sigma, 4),
            "mu_fork_rate": round(mu_fork, 4),
            "sigma_fork_rate": round(max(sigma_fork, 0.05), 4),
            "p99_burst_age": round(safe_p99_age, 2),
        }

    # Unknown fallback
    baselines["unknown"] = {
        "mu_cpu_share": 0.85,
        "sigma_cpu_share": 0.20,
        "mu_fork_rate": 0.0,
        "sigma_fork_rate": 0.10,
        "p99_burst_age": 500.0,
    }

    out_file = Path(output_path) if output_path is not None else BASELINES_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    return baselines


if __name__ == "__main__":
    b = calibrate()
    print("Calibration successful. Learned baselines:")
    print(json.dumps(b, indent=2))

"""Experiment runner for SABRE discrete-event CPU scheduling evaluation.

Usage:
    python -m sabre.experiments.run_experiment --exp E1 --seeds 30 --out results/E1.csv

Follows Workdoc Section 8.1, 8.2 and Research Report A4, A11, F3-F6:
- Writes one row per (scheduler, scenario, seed).
- Writes config_hash and git_commit_hash into every row.
- Disjoint evaluation seeds loaded from configs/seeds.yaml.
- Identical seeded workloads and arrival patterns for all schedulers.
- Comparative LSI and ADI metrics with epsilon guards.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any
import numpy as np
import pandas as pd
import yaml

from sabre.config import load_config
from sabre.core.attacks import (
    create_adaptive,
    create_forkbomb,
    create_hog,
    create_lowslow,
    create_mimicry,
)
from sabre.core.engine import Simulator
from sabre.core.metrics import (
    compute_comparative_metrics,
    compute_performance_metrics,
    compute_security_metrics,
)
from sabre.core.process import Process
from sabre.core.workload import WorkloadGenerator, clone_processes
from sabre.schedulers.base import Scheduler
from sabre.schedulers.baselines import (
    CFS_Like,
    FCFS,
    MLFQ,
    PriorityScheduler,
    RR_Cgroup,
    RR_StaticThreshold,
    RoundRobin,
    SJF_EMA,
    SJF_Oracle,
)
from sabre.schedulers.sabre import SabreScheduler


def get_git_commit_hash() -> str:
    """Return the current short git commit hash, or 'unknown' if not in git."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()[:10]
    except Exception:
        return "unknown"


def compute_config_hash(cfg: dict[str, Any]) -> str:
    """Compute sha256 hash of canonical serialized configuration dict."""
    serialized = yaml.dump(cfg, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:10]


def load_seeds(seeds_path: str | Path, key: str = "eval") -> list[int]:
    """Load disjoint seed split from yaml file."""
    with open(seeds_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data.get(key, []))


def make_scheduler(name: str, config: dict[str, Any]) -> Scheduler:
    """Factory creating fresh scheduler instance by name."""
    rr_q = float(config.get("rr", {}).get("quantum_ms", 10.0))
    attack_cfg = config.get("attack", {})
    tau = float(attack_cfg.get("static_threshold", 0.60))
    window = float(attack_cfg.get("static_window_ms", 500.0))
    alpha = float(config.get("predictor", {}).get("alpha", 0.5))

    if name == "FCFS":
        return FCFS()
    elif name == "SJF-oracle":
        return SJF_Oracle()
    elif name == "SJF-EMA":
        return SJF_EMA(alpha=alpha)
    elif name == "RR":
        return RoundRobin(quantum_ms=rr_q)
    elif name == "Priority":
        return PriorityScheduler()
    elif name == "MLFQ":
        return MLFQ()
    elif name == "RR+StaticThreshold":
        return RR_StaticThreshold(quantum_ms=rr_q, static_threshold=tau, window_ms=window)
    elif name == "CFS-like":
        return CFS_Like()
    elif name == "RR+cgroup-limits":
        return RR_Cgroup(quantum_ms=rr_q)
    elif name == "SABRE":
        return SabreScheduler(config=config)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def make_e1_attacks(config: dict[str, Any], seed: int, base_pid: int = 1000) -> list[Process]:
    """Generate 10% malicious processes (mixed types) for E1."""
    attack_cfg = config.get("attack", {})
    start_ms = float(attack_cfg.get("start_ms", 5000.0))
    rate = float(attack_cfg.get("forkbomb_rate", 50.0))
    max_desc = int(attack_cfg.get("max_descendants", 2000))
    stealth = float(attack_cfg.get("stealth", 0.9))
    tau = float(attack_cfg.get("static_threshold", 0.60))

    fb = create_forkbomb(pid=base_pid + 1, arrival=start_ms, rate=rate, max_descendants=max_desc)
    hog = create_hog(pid=base_pid + 2, arrival=start_ms)
    ls = create_lowslow(pid=base_pid + 3, arrival=start_ms, stealth=stealth, static_threshold=tau)
    ls_drift = create_lowslow(pid=base_pid + 4, arrival=start_ms, stealth=0.5, static_threshold=tau, drift=True)
    adaptive = create_adaptive(pid=base_pid + 5, arrival=start_ms, static_threshold=tau)
    mimicry = create_mimicry(pid=base_pid + 6, arrival=start_ms)

    return [fb, hog, ls, ls_drift, adaptive, mimicry]


SCHEDULERS_E1 = [
    "FCFS",
    "SJF-oracle",
    "SJF-EMA",
    "RR",
    "Priority",
    "MLFQ",
    "RR+StaticThreshold",
    "CFS-like",
    "RR+cgroup-limits",
    "SABRE",
]


def run_single_simulation(
    sched_name: str,
    procs: list[Process],
    config: dict[str, Any],
    sim_end_ms: float,
    ctx_switch_ms: float,
    seed: int,
    attack_start_ms: float,
) -> tuple[dict[str, Any], dict[str, Any], list[Process], dict[str, Any]]:
    """Run one simulation instance and compute performance and security metrics."""
    sched = make_scheduler(sched_name, config)
    sim = Simulator(
        scheduler=sched,
        processes=procs,
        sim_end_ms=sim_end_ms,
        ctx_switch_ms=ctx_switch_ms,
        seed=seed,
    )
    sim_res = sim.run()

    perf = compute_performance_metrics(sim_res["processes"], sim_res, sim_end_ms=sim_end_ms)
    
    flagged_info = getattr(sched, "flagged", lambda: {})()
    sec = compute_security_metrics(
        sim_res["processes"],
        flagged_info,
        attack_start_ms=attack_start_ms,
        sim_end_ms=sim_end_ms,
    )

    return perf, sec, sim_res["processes"], sim_res


def run_experiment_e1(
    config: dict[str, Any],
    seeds: list[int],
    sim_end_ms: float | None = None,
) -> pd.DataFrame:
    """Run Experiment E1: Overall comparison across all schedulers."""
    sim_cfg = config.get("sim", {})
    end_ms = float(sim_end_ms if sim_end_ms is not None else sim_cfg.get("sim_end_ms", 60000.0))
    ctx_ms = float(sim_cfg.get("ctx_switch_ms", 0.05))
    n_benign = int(sim_cfg.get("n_benign", 60))
    arr_rate = float(sim_cfg.get("arrival_rate_per_s", 4.0))
    attack_start = float(config.get("attack", {}).get("start_ms", 5000.0))

    cfg_hash = compute_config_hash(config)
    git_hash = get_git_commit_hash()

    rows: list[dict[str, Any]] = []

    for seed in seeds:
        # Generate benign processes once per seed
        wg = WorkloadGenerator(n_benign=n_benign, arrival_rate_per_s=arr_rate, preset="default")
        benign_base = wg.generate(seed=seed)

        # 1. First run RR with no attack (baseline for LSI and AbsoluteCost)
        rr_no_attack_procs = clone_processes(benign_base)
        rr_no_attack_perf, _, rr_no_attack_done, _ = run_single_simulation(
            "RR", rr_no_attack_procs, config, end_ms, ctx_ms, seed, attack_start
        )

        # 2. Run no-attack scenario for all schedulers
        no_attack_perfs: dict[str, dict[str, Any]] = {}
        for sname in SCHEDULERS_E1:
            if sname == "RR":
                perf = rr_no_attack_perf
                procs_done = rr_no_attack_done
                sec = compute_security_metrics(procs_done, {}, attack_start, end_ms)
            else:
                procs = clone_processes(benign_base)
                perf, sec, procs_done, _ = run_single_simulation(
                    sname, procs, config, end_ms, ctx_ms, seed, attack_start
                )

            no_attack_perfs[sname] = perf
            comp = compute_comparative_metrics(
                perf, perf, rr_no_attack_perf, procs_done, rr_no_attack_done
            )

            row = {
                "exp": "E1",
                "seed": seed,
                "scheduler": sname,
                "scenario": "no_attack",
                "config_hash": cfg_hash,
                "git_commit_hash": git_hash,
                **perf,
                "precision": sec["precision"],
                "recall": sec["recall"],
                "f1": sec["f1"],
                "fpr": sec["fpr"],
                "attacker_cpu_share": sec["attacker_cpu_share"],
                "instance_precision": sec["instance_precision"],
                "instance_recall": sec["instance_recall"],
                "instance_f1": sec["instance_f1"],
                "instance_fpr": sec["instance_fpr"],
                "detection_rate": sec["detection_rate"],
                "time_to_detect": sec["time_to_detect"],
                **comp,
            }
            rows.append(row)

        # 3. Run attack scenario for all schedulers
        for sname in SCHEDULERS_E1:
            attack_procs = clone_processes(benign_base) + make_e1_attacks(config, seed=seed)
            perf, sec, procs_done, _ = run_single_simulation(
                sname, attack_procs, config, end_ms, ctx_ms, seed, attack_start
            )

            comp = compute_comparative_metrics(
                perf, no_attack_perfs[sname], rr_no_attack_perf, procs_done, rr_no_attack_done
            )

            row = {
                "exp": "E1",
                "seed": seed,
                "scheduler": sname,
                "scenario": "attack",
                "config_hash": cfg_hash,
                "git_commit_hash": git_hash,
                **perf,
                "precision": sec["precision"],
                "recall": sec["recall"],
                "f1": sec["f1"],
                "fpr": sec["fpr"],
                "attacker_cpu_share": sec["attacker_cpu_share"],
                "instance_precision": sec["instance_precision"],
                "instance_recall": sec["instance_recall"],
                "instance_f1": sec["instance_f1"],
                "instance_fpr": sec["instance_fpr"],
                "detection_rate": sec["detection_rate"],
                "time_to_detect": sec["time_to_detect"],
                **comp,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="SABRE Experiment Runner")
    parser.add_argument("--exp", type=str, default="E1", help="Experiment ID (e.g. E1)")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds to evaluate")
    parser.add_argument("--out", type=str, default="results/E1.csv", help="Output CSV path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--seeds-file", type=str, default="configs/seeds.yaml", help="Path to seeds yaml")
    parser.add_argument("--sim-end-ms", type=float, default=None, help="Override simulation end time")

    args = parser.parse_args()

    config = load_config(args.config)
    eval_seeds = load_seeds(args.seeds_file, key="eval")
    selected_seeds = eval_seeds[:args.seeds]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.exp == "E1":
        df = run_experiment_e1(config, selected_seeds, sim_end_ms=args.sim_end_ms)
    else:
        # Default fallback to E1 structure for any standard comparison
        df = run_experiment_e1(config, selected_seeds, sim_end_ms=args.sim_end_ms)

    # Round floating-point columns to 6 decimal places for strict cross-platform byte reproducibility
    float_cols = df.select_dtypes(include=[np.number]).columns
    df[float_cols] = df[float_cols].round(6)

    # Write CSV with consistent newline format
    df.to_csv(out_path, index=False, lineterminator="\n")
    print(f"Successfully ran {args.exp} with {len(selected_seeds)} seeds -> {out_path}")


if __name__ == "__main__":
    main()

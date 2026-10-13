"""Parameter tuning module for SABRE.

IMPORTANT: Strictly evaluates ONLY on tuning seeds (100-119 from configs/seeds.yaml).
Never touches evaluation seeds (0-29).
Saves the chosen optimal parameters to configs/tuned.yaml.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
import numpy as np
import yaml

from sabre.config import load_config
from sabre.core.attacks import create_forkbomb, create_hog, create_lowslow
from sabre.core.engine import Simulator
from sabre.core.metrics import (
    compute_comparative_metrics,
    compute_performance_metrics,
    compute_security_metrics,
)
from sabre.core.workload import WorkloadGenerator, clone_processes
from sabre.schedulers.baselines import RoundRobin
from sabre.schedulers.sabre import SabreScheduler


def load_tuning_seeds(seeds_file: str | Path = "configs/seeds.yaml") -> list[int]:
    """Load disjoint tuning seeds."""
    with open(seeds_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data.get("tuning", []))


def evaluate_candidate(
    config: dict[str, Any],
    tuning_seeds: list[int],
    sim_end_ms: float | None = None,
) -> dict[str, float]:
    """Evaluate a candidate configuration over tuning seeds."""
    sim_cfg = config.get("sim", {})
    if sim_end_ms is None:
        sim_end_ms = float(sim_cfg.get("sim_end_ms", 90000.0))
    ctx_ms = float(sim_cfg.get("ctx_switch_ms", 0.05))
    n_benign = int(sim_cfg.get("n_benign", 60))
    arr_rate = float(sim_cfg.get("arrival_rate_per_s", 4.0))
    attack_start = float(config.get("attack", {}).get("start_ms", 5000.0))
    rr_q = float(config.get("rr", {}).get("quantum_ms", 10.0))

    adis: list[float] = []
    lsis: list[float] = []
    recalls: list[float] = []
    benign_fprs: list[float] = []
    waits: list[float] = []

    for seed in tuning_seeds:
        wg = WorkloadGenerator(n_benign=n_benign, arrival_rate_per_s=arr_rate, preset="default")
        benign_base = wg.generate(seed=seed)

        # Baseline RR no-attack
        p_rr = clone_processes(benign_base)
        sim_rr = Simulator(RoundRobin(quantum_ms=rr_q), p_rr, sim_end_ms=sim_end_ms, ctx_switch_ms=ctx_ms, seed=seed)
        res_rr = sim_rr.run()
        perf_rr = compute_performance_metrics(res_rr["processes"], res_rr, sim_end_ms=sim_end_ms)

        # 1. Benign-only run under candidate SABRE (to test false positive rate)
        p_benign = clone_processes(benign_base)
        sched_benign = SabreScheduler(config=config)
        sim_benign = Simulator(sched_benign, p_benign, sim_end_ms=sim_end_ms, ctx_switch_ms=ctx_ms, seed=seed)
        res_benign = sim_benign.run()
        perf_benign = compute_performance_metrics(res_benign["processes"], res_benign, sim_end_ms=sim_end_ms)
        sec_benign = compute_security_metrics(
            res_benign["processes"],
            sched_benign.flagged(),
            attack_start_ms=attack_start,
            sim_end_ms=sim_end_ms,
        )
        benign_fprs.append(sec_benign["fpr"])

        # 2. Attack run under candidate SABRE
        p_attack = clone_processes(benign_base)
        fb = create_forkbomb(pid=1001, arrival=attack_start, rate=50.0, max_descendants=500)
        hog = create_hog(pid=1002, arrival=attack_start)
        ls = create_lowslow(pid=1003, arrival=attack_start, stealth=0.9, static_threshold=0.6)
        p_attack.extend([fb, hog, ls])

        sched_att = SabreScheduler(config=config)
        sim_att = Simulator(sched_att, p_attack, sim_end_ms=sim_end_ms, ctx_switch_ms=ctx_ms, seed=seed)
        res_att = sim_att.run()

        perf_att = compute_performance_metrics(res_att["processes"], res_att, sim_end_ms=sim_end_ms)
        sec_att = compute_security_metrics(
            res_att["processes"],
            sched_att.flagged(),
            attack_start_ms=attack_start,
            sim_end_ms=sim_end_ms,
        )
        comp = compute_comparative_metrics(
            perf_att,
            perf_benign,
            perf_rr,
            res_att["processes"],
            res_rr["processes"],
        )

        adis.append(comp["adi"])
        lsis.append(comp["lsi"])
        recalls.append(sec_att["instance_recall"])
        waits.append(perf_att["avg_wait"])

    return {
        "mean_adi": float(np.mean(adis)),
        "mean_lsi": float(np.mean(lsis)),
        "mean_recall": float(np.mean(recalls)),
        "mean_fpr": float(np.mean(benign_fprs)),
        "mean_wait": float(np.mean(waits)),
    }


def tune_parameters(
    base_config_path: str = "configs/default.yaml",
    seeds_file: str = "configs/seeds.yaml",
    out_tuned_path: str = "configs/tuned.yaml",
) -> dict[str, Any]:
    """Search candidate parameter grid on tuning seeds 100-119."""
    base_cfg = load_config(base_config_path)
    tuning_seeds = load_tuning_seeds(seeds_file)
    print(f"Running parameter search strictly on {len(tuning_seeds)} tuning seeds: {tuning_seeds}")

    # Search space around default values
    candidate_thresholds = [
        [0.30, 0.55, 0.80],  # Default
        [0.25, 0.50, 0.75],
    ]
    candidate_weights = [
        [0.90, 0.70, 0.80],  # Default: fork, cpu, drift
        [0.90, 0.75, 0.85],
    ]
    candidate_caps = [
        [1.0, 0.50, 0.25, 0.10],  # Default
    ]
    candidate_slot_periods = [10, 8]
    candidate_d_max = [12, 15]
    candidate_cusum_h = [8.0, 10.0]

    best_score = float("inf")
    best_cfg = copy.deepcopy(base_cfg)
    best_metrics: dict[str, float] = {}

    trial = 0
    # Evaluate a focused grid of candidate configurations on tuning seeds
    for th in candidate_thresholds:
        for w in candidate_weights:
            for caps in candidate_caps:
                for sp in candidate_slot_periods:
                    for d_max in candidate_d_max:
                        for c_h in candidate_cusum_h:
                            trial += 1
                            cfg_cand = copy.deepcopy(base_cfg)
                            cfg_cand["risk"]["level_thresholds"] = th
                            cfg_cand["risk"]["weights"] = w
                            cfg_cand["sabre"]["caps"] = caps
                            cfg_cand["sabre"]["slot_period"] = sp
                            cfg_cand["risk"]["D_max"] = d_max
                            cfg_cand["risk"]["cusum_h_sigma"] = c_h
                            cfg_cand["risk"]["cusum_k_sigma"] = 0.8
                            cfg_cand["risk"]["trust_cap"] = 0.20

                            # Evaluate over tuning seeds
                            metrics = evaluate_candidate(cfg_cand, tuning_seeds[:5], sim_end_ms=60000.0)

                    # Multi-objective optimization:
                    # Penalize FPR > 0.05 heavily; minimize ADI, LSI, and wait time; maximize recall
                    fpr_penalty = max(0.0, metrics["mean_fpr"] - 0.05) * 50.0
                    score = metrics["mean_adi"] + metrics["mean_lsi"] + (1.0 - metrics["mean_recall"]) * 5.0 + fpr_penalty

                    if score < best_score:
                        best_score = score
                        best_cfg = cfg_cand
                        best_metrics = metrics
                        print(
                            f"Trial {trial:02d}: New Best! score={score:.4f} | "
                            f"ADI={metrics['mean_adi']:.2f}, LSI={metrics['mean_lsi']:.2f}, "
                            f"Recall={metrics['mean_recall']:.2f}, FPR={metrics['mean_fpr']:.4f} | "
                            f"th={th}, w={w}, caps={caps}, sp={sp}"
                        )

    # Validate best candidate on all 20 tuning seeds
    print("\nValidating chosen configuration on all 20 tuning seeds (100-119)...")
    full_metrics = evaluate_candidate(best_cfg, tuning_seeds, sim_end_ms=float(best_cfg.get("sim", {}).get("sim_end_ms", 90000.0)))
    print(
        f"Validation on tuning seeds (100-119): "
        f"ADI={full_metrics['mean_adi']:.2f}, LSI={full_metrics['mean_lsi']:.2f}, "
        f"Recall={full_metrics['mean_recall']:.2f}, FPR={full_metrics['mean_fpr']:.4f}, "
        f"Wait={full_metrics['mean_wait']:.2f} ms"
    )

    # Save to configs/tuned.yaml
    with open(out_tuned_path, "w", encoding="utf-8") as f:
        yaml.dump(best_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Saved tuned configuration to {out_tuned_path}")

    return best_cfg


if __name__ == "__main__":
    tune_parameters()

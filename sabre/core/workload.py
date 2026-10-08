"""Benign workload generation for SABRE simulator.

Follows Workdoc Section 5.2:
- Poisson arrival process (arrival_rate_per_s)
- 4 process kinds: interactive, io_bound, cpu_bound, bursty
- Distributions:
  * interactive: burst lognormal(mean=2, sigma=0.5), IO exp(mean=40), total_cpu uniform(200, 1000)
  * io_bound: burst lognormal(mean=8, sigma=0.5), IO exp(mean=25), total_cpu uniform(300, 1500)
  * cpu_bound: burst lognormal(mean=150, sigma=0.4), IO exp(mean=3), total_cpu uniform(2000, 8000)
  * bursty: burst mixture (70% mean=3, 30% mean=80), IO exp(mean=15), total_cpu uniform(500, 3000)
- Presets:
  * default: 30% interactive, 25% io_bound, 25% cpu_bound, 20% bursty
  * benign-heavy: 70% cpu_bound, 10% interactive, 10% io_bound, 10% bursty
- Pre-samples bursts and IO waits so that deepcopy gives identical execution sequences across schedulers.
"""

from __future__ import annotations

import copy
import math
from typing import Any
import numpy as np

from sabre.core.process import Process, ProcState


def _lognormal_sample(rng: np.random.Generator, target_mean: float, sigma: float = 0.5) -> float:
    """Sample from lognormal distribution with given target mean and sigma."""
    # If X ~ Lognormal(mu, sigma), then E[X] = exp(mu + sigma^2 / 2)
    # => mu = ln(target_mean) - sigma^2 / 2
    mu = math.log(max(1e-4, target_mean)) - 0.5 * (sigma**2)
    sample = rng.lognormal(mean=mu, sigma=sigma)
    return float(max(0.1, sample))


def _sample_burst(kind: str, rng: np.random.Generator) -> float:
    """Sample one CPU burst for a given kind."""
    if kind == "interactive":
        return _lognormal_sample(rng, target_mean=2.0, sigma=0.5)
    elif kind == "io_bound":
        return _lognormal_sample(rng, target_mean=8.0, sigma=0.5)
    elif kind == "cpu_bound":
        return _lognormal_sample(rng, target_mean=150.0, sigma=0.4)
    elif kind == "bursty":
        if rng.random() < 0.70:
            return _lognormal_sample(rng, target_mean=3.0, sigma=0.5)
        else:
            return _lognormal_sample(rng, target_mean=80.0, sigma=0.5)
    return 10.0


def _sample_io_wait(kind: str, rng: np.random.Generator) -> float:
    """Sample one I/O wait duration for a given kind."""
    if kind == "interactive":
        return float(max(0.1, rng.exponential(scale=40.0)))
    elif kind == "io_bound":
        return float(max(0.1, rng.exponential(scale=25.0)))
    elif kind == "cpu_bound":
        return float(max(0.1, rng.exponential(scale=3.0)))
    elif kind == "bursty":
        return float(max(0.1, rng.exponential(scale=15.0)))
    return 10.0


class WorkloadGenerator:
    """Generates synthetic benign process workloads."""

    def __init__(
        self,
        n_benign: int = 60,
        arrival_rate_per_s: float = 4.0,
        preset: str = "default",
    ) -> None:
        self.n_benign = int(n_benign)
        self.arrival_rate_per_s = float(arrival_rate_per_s)
        self.preset = preset

    def generate(self, seed: int) -> list[Process]:
        """Generate benign workload processes for a given seed."""
        rng = np.random.Generator(np.random.PCG64(seed))

        # Class distribution probabilities
        if self.preset == "benign-heavy":
            kinds = ["cpu_bound", "interactive", "io_bound", "bursty"]
            probs = [0.70, 0.10, 0.10, 0.10]
        else:  # default
            kinds = ["interactive", "io_bound", "cpu_bound", "bursty"]
            probs = [0.30, 0.25, 0.25, 0.20]

        processes: list[Process] = []
        current_arrival = 0.0

        for pid in range(1, self.n_benign + 1):
            # Poisson arrivals: inter-arrival time is Exp(lambda), arrival_rate_per_s in processes/sec
            # In milliseconds: rate per ms = arrival_rate_per_s / 1000.0, scale = 1000.0 / arrival_rate_per_s
            inter_arrival_ms = float(rng.exponential(scale=1000.0 / self.arrival_rate_per_s))
            current_arrival += inter_arrival_ms

            kind = str(rng.choice(kinds, p=probs))

            # Total CPU demand
            if kind == "interactive":
                total_cpu = float(rng.uniform(200.0, 1000.0))
            elif kind == "io_bound":
                total_cpu = float(rng.uniform(300.0, 1500.0))
            elif kind == "cpu_bound":
                total_cpu = float(rng.uniform(2000.0, 8000.0))
            else:  # bursty
                total_cpu = float(rng.uniform(500.0, 3000.0))

            # Pre-sample bursts and IO waits so the process behavior is entirely deterministic
            # and identical regardless of the scheduling order/quantum.
            # We sample enough bursts and IO waits to comfortably cover total_cpu.
            bursts_pool: list[float] = []
            io_pool: list[float] = []
            acc = 0.0
            while acc < total_cpu * 1.5 + 500.0:
                b = _sample_burst(kind, rng)
                bursts_pool.append(b)
                acc += b
                io_pool.append(_sample_io_wait(kind, rng))

            # Closure generators that consume pre-sampled pools
            # We store state in lists so that copies/clones each have their own independent progress
            p = Process(
                pid=pid,
                ppid=None,
                arrival=round(current_arrival, 4),
                kind=kind,
                label=0,
                total_cpu=round(total_cpu, 4),
            )

            # Attach pre-sampled sequences
            burst_idx = [0]
            io_idx = [0]

            def make_burst_gen(pool: list[float], idx: list[int]):
                def gen(now: float, _rng: Any) -> float:
                    if idx[0] < len(pool):
                        val = pool[idx[0]]
                        idx[0] += 1
                        return val
                    return 10.0
                return gen

            def make_io_gen(pool: list[float], idx: list[int]):
                def gen(now: float, _rng: Any) -> float:
                    if idx[0] < len(pool):
                        val = pool[idx[0]]
                        idx[0] += 1
                        return val
                    return 10.0
                return gen

            p.burst_generator = make_burst_gen(list(bursts_pool), burst_idx)
            p.io_generator = make_io_gen(list(io_pool), io_idx)
            # Store the raw pools on the process for easy independent deep-cloning
            setattr(p, "_raw_burst_pool", bursts_pool)
            setattr(p, "_raw_io_pool", io_pool)

            processes.append(p)

        return processes


def clone_processes(processes: list[Process]) -> list[Process]:
    """Create an independent fresh copy of workload processes for another scheduler run.

    Resets simulation state fields while preserving pre-sampled burst and IO sequences.
    """
    clones: list[Process] = []
    for p in processes:
        new_p = Process(
            pid=p.pid,
            ppid=p.ppid,
            arrival=p.arrival,
            kind=p.kind,
            label=p.label,
            total_cpu=p.total_cpu,
            state=ProcState.NEW,
            remaining_burst=0.0,
            cpu_used=0.0,
            first_run=None,
            finish=None,
            wait_time=0.0,
            last_ready_at=0.0,
            spawn_generator=p.spawn_generator,
        )
        if hasattr(p, "_raw_burst_pool") and hasattr(p, "_raw_io_pool"):
            b_pool = list(getattr(p, "_raw_burst_pool"))
            io_pool = list(getattr(p, "_raw_io_pool"))
            b_idx = [0]
            io_idx = [0]

            def make_b_gen(pool: list[float], idx: list[int]):
                def gen(now: float, _rng: Any) -> float:
                    if idx[0] < len(pool):
                        val = pool[idx[0]]
                        idx[0] += 1
                        return val
                    return 10.0
                return gen

            def make_i_gen(pool: list[float], idx: list[int]):
                def gen(now: float, _rng: Any) -> float:
                    if idx[0] < len(pool):
                        val = pool[idx[0]]
                        idx[0] += 1
                        return val
                    return 10.0
                return gen

            new_p.burst_generator = make_b_gen(b_pool, b_idx)
            new_p.io_generator = make_i_gen(io_pool, io_idx)
            setattr(new_p, "_raw_burst_pool", b_pool)
            setattr(new_p, "_raw_io_pool", io_pool)
        clones.append(new_p)
    return clones

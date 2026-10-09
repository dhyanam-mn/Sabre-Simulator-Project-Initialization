"""Trace loader for public Standard Workload Format (SWF) files.

Implements Workdoc Section 8.2 (E10) and Research Report A6:
- Parses SWF format workload traces from the Parallel Workloads Archive.
- Provides a documented synthetic heavy-tailed stand-in trace generator
  when offline without external internet trace file downloads.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import numpy as np

from sabre.core.process import Process, ProcState


def load_swf_trace(
    filepath: str | Path,
    max_jobs: int = 100,
    time_scale: float = 1000.0,  # convert SWF seconds to simulator milliseconds
) -> list[Process]:
    """Load benign processes from a Standard Workload Format (SWF) file.

    SWF Columns:
    Field 1: Job Number
    Field 2: Submit Time (in seconds)
    Field 3: Wait Time (in seconds)
    Field 4: Run Time (in seconds)
    Field 5: Number of Allocated Processors
    Field 6: Average CPU Time Used
    ...
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"SWF trace file not found: {filepath}")

    processes: list[Process] = []
    pid_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue  # Comment line

            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                job_id = int(parts[0])
                submit_sec = float(parts[1])
                run_sec = float(parts[3])
            except ValueError:
                continue

            if run_sec <= 0.0:
                continue

            pid_count += 1
            arrival_ms = submit_sec * time_scale
            total_cpu_ms = run_sec * time_scale

            # Infer kind based on job duration
            if total_cpu_ms < 100.0:
                kind = "interactive"
                mean_burst = 2.0
            elif total_cpu_ms < 1000.0:
                kind = "io_bound"
                mean_burst = 8.0
            elif total_cpu_ms > 10000.0:
                kind = "cpu_bound"
                mean_burst = 150.0
            else:
                kind = "bursty"
                mean_burst = 20.0

            p = Process(
                pid=pid_count,
                ppid=None,
                arrival=arrival_ms,
                kind=kind,
                label=0,
                total_cpu=total_cpu_ms,
            )
            # Pre-sample burst sequence
            p.burst_generator = lambda _now, _rng, mb=mean_burst: float(max(0.5, _rng.exponential(mb) if hasattr(_rng, "exponential") else mb))
            p.io_generator = lambda _now, _rng: float(max(1.0, _rng.exponential(15.0) if hasattr(_rng, "exponential") else 15.0))
            processes.append(p)

            if len(processes) >= max_jobs:
                break

    return processes


def generate_synthetic_heavy_tailed_trace(
    seed: int,
    n_processes: int = 60,
    pareto_shape: float = 1.5,
    arrival_rate_per_s: float = 4.0,
) -> list[Process]:
    """Documented synthetic heavy-tailed stand-in trace for E10 / E11.

    Used when running offline without external SWF trace files. Models
    heavy-tailed CPU demands following a Pareto distribution (power law)
    typical of real cluster and supercomputing production traces.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    processes: list[Process] = []

    current_arrival = 0.0
    for pid in range(1, n_processes + 1):
        inter_arrival = float(rng.exponential(scale=1000.0 / arrival_rate_per_s))
        current_arrival += inter_arrival

        # Pareto-distributed job execution time: x = x_m / (U)^(1/alpha)
        # Scale x_m = 50.0 ms, alpha = 1.5 -> heavy tail with finite mean, infinite variance
        u = float(rng.uniform(1e-4, 1.0))
        total_cpu = float(min(15000.0, 50.0 / (u ** (1.0 / pareto_shape))))

        if total_cpu > 4000.0:
            kind = "cpu_bound"
            mean_b = 150.0
        elif total_cpu < 300.0:
            kind = "interactive"
            mean_b = 2.0
        elif total_cpu < 1500.0:
            kind = "io_bound"
            mean_b = 8.0
        else:
            kind = "bursty"
            mean_b = 25.0

        p = Process(
            pid=pid,
            ppid=None,
            arrival=current_arrival,
            kind=kind,
            label=0,
            total_cpu=total_cpu,
        )
        p.burst_generator = lambda _now, _rng, mb=mean_b: float(max(0.5, _rng.exponential(mb) if hasattr(_rng, "exponential") else mb))
        p.io_generator = lambda _now, _rng: float(max(1.0, _rng.exponential(15.0) if hasattr(_rng, "exponential") else 15.0))
        processes.append(p)

    return processes

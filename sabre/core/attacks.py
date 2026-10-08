"""Attack models for SABRE simulator.

Follows Workdoc Section 5.3 and Research Report A3:
1. forkbomb:
   - While holding CPU, spawns 1 forkchild per 1000/(forkbomb_rate * intensity) ms.
   - Child: burst 5 ms, total 20-50 ms, label 1, ppid=forkbomb.pid.
   - Capped at max_descendants (default 2000).
2. hog:
   - One or more processes with single infinite CPU burst (total_cpu=inf), no IO, label 1.
3. lowslow:
   - Duty-cycled process targeting CPU share s = stealth * tau_static over 500 ms window.
   - Drift variant: starts at 0.5 * tau and increments share by 1% (0.01) every 500 ms.
4. adaptive:
   - Black-box duty-cycled attacker measuring own recent wait/burst ratio over 500 ms.
   - If ratio rose (throttled), cuts duty by 20%.
   - If unthrottled for 2 consecutive windows, raises duty by 5%.
5. mimicry:
   - CPU miner (large/infinite total demand) padded with fake short IO waits (e.g., 10-15 ms)
     interspersed with bursts (e.g., 50-80 ms) to mimic bursty benign behavior.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from sabre.core.process import Process, ProcState


def make_forkchild(
    pid: int,
    ppid: int,
    now: float,
    rng: np.random.Generator,
    burst_ms: float = 5.0,
    total_cpu_range: tuple[float, float] = (20.0, 50.0),
) -> Process:
    """Create a short CPU spinner child process spawned by a fork bomb."""
    total_cpu = float(rng.uniform(total_cpu_range[0], total_cpu_range[1]))
    p = Process(
        pid=pid,
        ppid=ppid,
        arrival=now,
        kind="forkchild",
        label=1,
        total_cpu=total_cpu,
    )
    p.burst_generator = lambda _now, _rng: burst_ms
    p.io_generator = lambda _now, _rng: 0.0
    return p


def create_forkbomb(
    pid: int,
    arrival: float = 5000.0,
    rate: float = 50.0,
    intensity: float = 1.0,
    max_descendants: int = 2000,
    total_cpu: float = float("inf"),
    next_pid_fn: Any = None,
) -> Process:
    """Create a fork bomb process that spawns children as it consumes CPU."""
    effective_rate = float(rate * intensity)
    ms_per_child = 1000.0 / max(1e-3, effective_rate)

    descendants_count = [0]
    cpu_accumulator = [0.0]
    pid_counter = [pid + 100000]

    def alloc_pid() -> int:
        if next_pid_fn is not None:
            return next_pid_fn()
        pid_counter[0] += 1
        return pid_counter[0]

    def spawn_gen(cpu_delta: float, now: float, rng: Any) -> list[Process]:
        if descendants_count[0] >= max_descendants:
            return []
        cpu_accumulator[0] += cpu_delta
        new_children: list[Process] = []
        while cpu_accumulator[0] >= ms_per_child and descendants_count[0] < max_descendants:
            cpu_accumulator[0] -= ms_per_child
            descendants_count[0] += 1
            child = make_forkchild(
                pid=alloc_pid(),
                ppid=pid,
                now=now,
                rng=rng if isinstance(rng, np.random.Generator) else np.random.default_rng(),
            )
            new_children.append(child)
        return new_children

    p = Process(
        pid=pid,
        ppid=None,
        arrival=float(arrival),
        kind="forkbomb",
        label=1,
        total_cpu=float(total_cpu),
    )
    p.burst_generator = lambda _now, _rng: 100.0
    p.io_generator = lambda _now, _rng: 0.0
    p.spawn_generator = spawn_gen
    return p


def create_hog(
    pid: int,
    arrival: float = 5000.0,
    total_cpu: float = float("inf"),
) -> Process:
    """Create a CPU hog process with an infinite single burst and no IO."""
    p = Process(
        pid=pid,
        ppid=None,
        arrival=float(arrival),
        kind="hog",
        label=1,
        total_cpu=float(total_cpu),
    )
    p.burst_generator = lambda _now, _rng: float("inf")
    p.io_generator = lambda _now, _rng: 0.0
    return p


def create_lowslow(
    pid: int,
    arrival: float = 5000.0,
    stealth: float = 0.90,
    static_threshold: float = 0.60,
    window_ms: float = 500.0,
    drift: bool = False,
    drift_step: float = 0.01,
    drift_interval_ms: float = 500.0,
    total_cpu: float = float("inf"),
) -> Process:
    """Create a duty-cycled low-and-slow attacker."""
    target_share = [float(0.50 * static_threshold if drift else stealth * static_threshold)]
    last_drift_time = [arrival]

    # For a cycle of window_ms:
    # burst = s * window_ms
    # io = (1 - s) * window_ms
    def next_burst(now: float, _rng: Any) -> float:
        if drift and (now - last_drift_time[0] >= drift_interval_ms):
            target_share[0] = min(1.0, target_share[0] + drift_step)
            last_drift_time[0] = now
        s = min(0.99, max(0.01, target_share[0]))
        return float(s * window_ms)

    def next_io(now: float, _rng: Any) -> float:
        s = min(0.99, max(0.01, target_share[0]))
        return float((1.0 - s) * window_ms)

    p = Process(
        pid=pid,
        ppid=None,
        arrival=float(arrival),
        kind="lowslow",
        label=1,
        total_cpu=float(total_cpu),
    )
    p.burst_generator = next_burst
    p.io_generator = next_io
    return p


def create_adaptive(
    pid: int,
    arrival: float = 5000.0,
    static_threshold: float = 0.60,
    window_ms: float = 500.0,
    total_cpu: float = float("inf"),
) -> Process:
    """Create a black-box adaptive attacker.

    Measures its own wait/burst ratio every 500 ms:
    - If ratio rose (throttling detected), cuts duty by 20%.
    - If unthrottled for 2 consecutive windows, raises duty by 5%.
    """
    duty = [float(static_threshold)]
    last_eval_time = [arrival]
    last_wait = [0.0]
    last_cpu = [0.0]
    last_ratio = [0.0]
    unthrottled_windows = [0]

    def update_duty(now: float, proc: Process) -> None:
        if now - last_eval_time[0] >= window_ms:
            delta_w = proc.wait_time - last_wait[0]
            delta_c = proc.cpu_used - last_cpu[0]
            curr_ratio = delta_w / max(1e-3, delta_c)

            if curr_ratio > last_ratio[0] + 0.10:
                # Being throttled: cut duty by 20%
                duty[0] = max(0.05, duty[0] * 0.80)
                unthrottled_windows[0] = 0
            else:
                unthrottled_windows[0] += 1
                if unthrottled_windows[0] >= 2:
                    # Unthrottled: cautiously raise duty by 5%
                    duty[0] = min(0.95, duty[0] * 1.05)
                    unthrottled_windows[0] = 0

            last_wait[0] = proc.wait_time
            last_cpu[0] = proc.cpu_used
            last_ratio[0] = curr_ratio
            last_eval_time[0] = now

    p = Process(
        pid=pid,
        ppid=None,
        arrival=float(arrival),
        kind="adaptive",
        label=1,
        total_cpu=float(total_cpu),
    )

    def next_burst(now: float, _rng: Any) -> float:
        update_duty(now, p)
        s = min(0.95, max(0.05, duty[0]))
        return float(s * window_ms)

    def next_io(now: float, _rng: Any) -> float:
        update_duty(now, p)
        s = min(0.95, max(0.05, duty[0]))
        return float((1.0 - s) * window_ms)

    p.burst_generator = next_burst
    p.io_generator = next_io
    return p


def create_mimicry(
    pid: int,
    arrival: float = 5000.0,
    burst_mean: float = 60.0,
    io_mean: float = 15.0,
    total_cpu: float = float("inf"),
) -> Process:
    """Create a mimicry attacker (cryptominer padded with short IO to pose as bursty)."""
    p = Process(
        pid=pid,
        ppid=None,
        arrival=float(arrival),
        kind="mimicry",
        label=1,
        total_cpu=float(total_cpu),
    )
    p.burst_generator = lambda _now, rng: float(max(10.0, rng.normal(burst_mean, 10.0) if hasattr(rng, "normal") else burst_mean))
    p.io_generator = lambda _now, rng: float(max(2.0, rng.normal(io_mean, 3.0) if hasattr(rng, "normal") else io_mean))
    return p

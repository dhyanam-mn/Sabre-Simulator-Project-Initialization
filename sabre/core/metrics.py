"""Metrics calculation for SABRE simulator.

Implements Workdoc Section 5.5, Research Report Fixes F3, F4, F5:
- Performance metrics: avg_wait, avg_turnaround, throughput, resp_p50/p95/p99,
  jain_fairness (over benign processes), ctx_switches, overhead_ms, overhead_pct.
- Process-level security: precision, recall, f1, fpr, attacker_cpu_share.
- Instance-level security (F3): instance_precision, instance_recall, instance_f1, instance_fpr.
- Censored Time-to-Detect (F5): detection_rate, kaplan_meier_ttd, median TTD.
- Dual LSI and ADI metrics (F4): AttackSlowdown and AbsoluteCost pairs with epsilon guards.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from sabre.core.process import Process, ProcState


def compute_kaplan_meier(
    event_times: list[float],
    censored_times: list[float],
) -> list[tuple[float, float]]:
    """Compute Kaplan-Meier detection curve F(t) = 1 - S(t).

    Returns a list of (time, cumulative_detection_probability) sorted by time.
    """
    records: list[tuple[float, int]] = []
    for t in event_times:
        records.append((t, 1))  # 1 = detected
    for t in censored_times:
        records.append((t, 0))  # 0 = censored

    if not records:
        return [(0.0, 0.0)]

    records.sort(key=lambda x: (x[0], -x[1]))

    n_at_risk = len(records)
    surv = 1.0
    curve: list[tuple[float, float]] = [(0.0, 0.0)]

    i = 0
    while i < len(records):
        t_curr = records[i][0]
        events_at_t = 0
        censored_at_t = 0
        while i < len(records) and records[i][0] == t_curr:
            if records[i][1] == 1:
                events_at_t += 1
            else:
                censored_at_t += 1
            i += 1

        if n_at_risk > 0 and events_at_t > 0:
            surv *= (1.0 - (events_at_t / n_at_risk))
            curve.append((t_curr, 1.0 - surv))
        elif censored_at_t > 0:
            curve.append((t_curr, 1.0 - surv))

        n_at_risk -= (events_at_t + censored_at_t)

    return curve


def compute_performance_metrics(
    processes: list[Process],
    sim_res: dict[str, Any],
    sim_end_ms: float = 60000.0,
) -> dict[str, float]:
    """Compute standard system performance metrics."""
    benign_procs = [p for p in processes if p.label == 0]
    completed_procs = [p for p in processes if p.state == ProcState.DONE and p.finish is not None]
    completed_benign = [p for p in benign_procs if p.state == ProcState.DONE and p.finish is not None]

    # Average wait time across all benign processes
    if benign_procs:
        avg_wait = float(np.mean([p.wait_time for p in benign_procs]))
    elif processes:
        avg_wait = float(np.mean([p.wait_time for p in processes]))
    else:
        avg_wait = 0.0

    # Average turnaround across completed benign processes (or all completed)
    if completed_benign:
        avg_turnaround = float(np.mean([p.finish - p.arrival for p in completed_benign]))
    elif completed_procs:
        avg_turnaround = float(np.mean([p.finish - p.arrival for p in completed_procs]))
    else:
        avg_turnaround = 0.0

    # Throughput: completed processes per second
    throughput = float(len(completed_benign) / (sim_end_ms / 1000.0)) if sim_end_ms > 0 else 0.0

    # Interactive burst response percentiles
    interactive_procs = [p for p in processes if p.kind == "interactive"]
    resp_samples: list[float] = []
    for p in interactive_procs:
        resp_samples.extend(p.response_samples)

    if resp_samples:
        resp_p50 = float(np.percentile(resp_samples, 50))
        resp_p95 = float(np.percentile(resp_samples, 95))
        resp_p99 = float(np.percentile(resp_samples, 99))
    else:
        resp_p50 = 0.0
        resp_p95 = 0.0
        resp_p99 = 0.0

    # Jain's fairness index over benign processes: (sum x_i)^2 / (n * sum x_i^2)
    # where x_i is service ratio (cpu_used / total_cpu)
    if benign_procs:
        ratios = [min(1.0, max(0.0, p.cpu_used / max(1e-4, p.total_cpu))) for p in benign_procs]
        sum_r = sum(ratios)
        sum_sq = sum(r * r for r in ratios)
        n = len(ratios)
        if sum_sq > 0 and n > 0:
            jain_fairness = float((sum_r * sum_r) / (n * sum_sq))
        else:
            jain_fairness = 1.0
    else:
        jain_fairness = 1.0

    ctx_switches = float(sim_res.get("ctx_switches", 0))
    overhead_ms = float(sim_res.get("overhead_ms", 0.0))
    overhead_pct = float(overhead_ms / max(1e-4, sim_end_ms) * 100.0)

    return {
        "avg_wait": avg_wait,
        "avg_turnaround": avg_turnaround,
        "throughput": throughput,
        "resp_p50": resp_p50,
        "resp_p95": resp_p95,
        "resp_p99": resp_p99,
        "jain_fairness": jain_fairness,
        "ctx_switches": ctx_switches,
        "overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
    }


def compute_security_metrics(
    processes: list[Process],
    flagged: dict[int, float] | set[int] | list[int],
    attack_start_ms: float = 5000.0,
    sim_end_ms: float = 60000.0,
) -> dict[str, Any]:
    """Compute process-level and attack-instance-level security metrics.

    Follows Research Report F3 (instance-level metrics) and F5 (censored TTD).
    """
    if isinstance(flagged, dict):
        flagged_times = dict(flagged)
        flagged_pids = set(flagged.keys())
    else:
        flagged_pids = set(flagged)
        flagged_times = {pid: attack_start_ms for pid in flagged_pids}

    # 1. Process-level metrics
    tp_proc = sum(1 for p in processes if p.label == 1 and p.pid in flagged_pids)
    fp_proc = sum(1 for p in processes if p.label == 0 and p.pid in flagged_pids)
    fn_proc = sum(1 for p in processes if p.label == 1 and p.pid not in flagged_pids)
    tn_proc = sum(1 for p in processes if p.label == 0 and p.pid not in flagged_pids)

    precision = float(tp_proc / (tp_proc + fp_proc)) if (tp_proc + fp_proc) > 0 else 0.0
    recall = float(tp_proc / (tp_proc + fn_proc)) if (tp_proc + fn_proc) > 0 else (1.0 if fn_proc == 0 else 0.0)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp_proc / (fp_proc + tn_proc)) if (fp_proc + tn_proc) > 0 else 0.0

    total_cpu = sum(p.cpu_used for p in processes)
    attacker_cpu = sum(p.cpu_used for p in processes if p.label == 1)
    attacker_cpu_share = float(attacker_cpu / max(1e-4, total_cpu))

    # 2. Instance-level metrics (F3)
    # Map each malicious process to its root ancestor in the malicious set
    malicious_pids = {p.pid for p in processes if p.label == 1}
    proc_by_pid = {p.pid: p for p in processes}

    def find_malicious_root(p: Process) -> int:
        curr = p
        while curr.ppid is not None and curr.ppid in malicious_pids:
            curr = proc_by_pid[curr.ppid]
        return curr.pid

    # Group malicious processes into instances
    instances: dict[int, list[Process]] = {}
    for p in processes:
        if p.label == 1:
            root_pid = find_malicious_root(p)
            instances.setdefault(root_pid, []).append(p)

    event_times: list[float] = []
    censored_times: list[float] = []
    tp_inst = 0
    fn_inst = 0

    for root_pid, members in instances.items():
        root_p = proc_by_pid[root_pid]
        start_time = max(root_p.arrival, attack_start_ms)
        flag_times_in_inst = [flagged_times[m.pid] for m in members if m.pid in flagged_times]
        if flag_times_in_inst:
            tp_inst += 1
            earliest_flag = min(flag_times_in_inst)
            ttd = max(0.0, earliest_flag - start_time)
            event_times.append(ttd)
        else:
            fn_inst += 1
            censored_duration = max(0.0, sim_end_ms - start_time)
            censored_times.append(censored_duration)

    # Benign instances: each benign process is an independent benign instance
    benign_pids = [p.pid for p in processes if p.label == 0]
    fp_inst = sum(1 for pid in benign_pids if pid in flagged_pids)
    tn_inst = sum(1 for pid in benign_pids if pid not in flagged_pids)

    inst_precision = float(tp_inst / (tp_inst + fp_inst)) if (tp_inst + fp_inst) > 0 else 0.0
    inst_recall = float(tp_inst / (tp_inst + fn_inst)) if (tp_inst + fn_inst) > 0 else (1.0 if fn_inst == 0 else 0.0)
    inst_f1 = float(2 * inst_precision * inst_recall / (inst_precision + inst_recall)) if (inst_precision + inst_recall) > 0 else 0.0
    inst_fpr = float(fp_inst / (fp_inst + tn_inst)) if (fp_inst + tn_inst) > 0 else 0.0

    total_instances = tp_inst + fn_inst
    detection_rate = float(tp_inst / total_instances) if total_instances > 0 else 1.0
    time_to_detect = float(np.median(event_times)) if event_times else 0.0

    km_curve = compute_kaplan_meier(event_times, censored_times)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "attacker_cpu_share": attacker_cpu_share,
        "instance_precision": inst_precision,
        "instance_recall": inst_recall,
        "instance_f1": inst_f1,
        "instance_fpr": inst_fpr,
        "detection_rate": detection_rate,
        "time_to_detect": time_to_detect,
        "km_curve": km_curve,
    }


def compute_comparative_metrics(
    sched_attack_metrics: dict[str, float],
    sched_no_attack_metrics: dict[str, float],
    rr_no_attack_metrics: dict[str, float],
    sched_attack_procs: list[Process],
    rr_no_attack_procs: list[Process],
    eps: float = 1e-4,
) -> dict[str, float]:
    """Compute comparative AttackSlowdown and AbsoluteCost metrics (Fix F4).

    Definitions:
    - AttackSlowdown(X, metric) = metric(X, attack) / (metric(X, no attack) + eps)
    - AbsoluteCost(X, metric) = metric(X, no attack) / (metric(RR, no attack) + eps)
    - LSI = mean turnaround of benign cpu_bound under X (attack) / mean turnaround under RR (no attack)
    - LSI_all = mean turnaround of all benign under X (attack) / mean turnaround under RR (no attack)
    - ADI = interactive resp_p95 under X (attack) / (interactive resp_p95 under X (no attack) + eps)
    """
    # 1. LSI (Legacy Slowdown Index)
    cpubound_attack = [p for p in sched_attack_procs if p.kind == "cpu_bound" and p.label == 0 and p.finish is not None]
    cpubound_rr = [p for p in rr_no_attack_procs if p.kind == "cpu_bound" and p.label == 0 and p.finish is not None]

    if cpubound_attack and cpubound_rr:
        ta_cpubound_attack = float(np.mean([p.finish - p.arrival for p in cpubound_attack]))
        ta_cpubound_rr = float(np.mean([p.finish - p.arrival for p in cpubound_rr]))
        lsi = float(ta_cpubound_attack / (ta_cpubound_rr + eps))
    else:
        lsi = 1.0

    # LSI_all (all benign classes)
    benign_attack = [p for p in sched_attack_procs if p.label == 0 and p.finish is not None]
    benign_rr = [p for p in rr_no_attack_procs if p.label == 0 and p.finish is not None]

    if benign_attack and benign_rr:
        ta_benign_attack = float(np.mean([p.finish - p.arrival for p in benign_attack]))
        ta_benign_rr = float(np.mean([p.finish - p.arrival for p in benign_rr]))
        lsi_all = float(ta_benign_attack / (ta_benign_rr + eps))
    else:
        lsi_all = 1.0

    # 2. ADI (Attack Degradation Index)
    resp_attack = sched_attack_metrics.get("resp_p95", 0.0)
    resp_no_attack = sched_no_attack_metrics.get("resp_p95", 0.0)
    resp_rr = rr_no_attack_metrics.get("resp_p95", 0.0)

    adi = float(resp_attack / (resp_no_attack + eps))
    adi_absolute = float(resp_no_attack / (resp_rr + eps))

    # General AttackSlowdown and AbsoluteCost on wait time
    wait_attack = sched_attack_metrics.get("avg_wait", 0.0)
    wait_no_attack = sched_no_attack_metrics.get("avg_wait", 0.0)
    wait_rr = rr_no_attack_metrics.get("avg_wait", 0.0)

    wait_attack_slowdown = float(wait_attack / (wait_no_attack + eps))
    wait_absolute_cost = float(wait_no_attack / (wait_rr + eps))

    return {
        "lsi": lsi,
        "lsi_all": lsi_all,
        "adi": adi,
        "adi_absolute": adi_absolute,
        "attack_slowdown_wait": wait_attack_slowdown,
        "absolute_cost_wait": wait_absolute_cost,
    }

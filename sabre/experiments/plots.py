"""Plotting module for SABRE experiments.

Generates IEEE-style publication figures (both PNG and PDF at 300 dpi) in results/:
1. fig_e1_bars: Waiting time and interactive p95 latency bars with 95% CIs.
2. fig_e3_stealth: Recall and Time-to-Detect vs attack stealth.
3. fig_e4_malicious_frac: ADI vs malicious process fraction.
4. fig_e7_ablation: Single-component ablation impact on latency and security.
5. fig_e8_overhead: Overhead sensitivity and break-even frontier.
6. fig_e9_pareto: Security benefit (1/ADI) vs collateral damage (LSI) Pareto scatter.
7. fig_classifier_confusion: Classifier confusion matrix.
8. fig_timeline: Time series of CPU share and containment level (attacker vs benign).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# IEEE-friendly typography and style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.titlesize": 10,
    "figure.autolayout": True,
    "figure.dpi": 300,
})

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, base_name: str) -> None:
    """Save figure as both 300 dpi PNG and PDF."""
    png_path = RESULTS_DIR / f"{base_name}.png"
    pdf_path = RESULTS_DIR / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")


def plot_e1_bars(df_e1: pd.DataFrame) -> None:
    """Plot E1 waiting time and interactive p95 latency bars with 95% CIs."""
    # Filter for attack scenario
    df_att = df_e1[df_e1["scenario"] == "attack"] if "scenario" in df_e1.columns else df_e1

    sched_order = [
        "FCFS", "SJF-oracle", "SJF-EMA", "RR", "Priority",
        "MLFQ", "RR+StaticThreshold", "CFS-like", "RR+cgroup-limits", "SABRE"
    ]
    scheds = [s for s in sched_order if s in df_att["scheduler"].unique()]

    mean_waits: list[float] = []
    err_waits: list[float] = []
    mean_resp: list[float] = []
    err_resp: list[float] = []

    for s in scheds:
        sub = df_att[df_att["scheduler"] == s]
        w = sub["avg_wait"].values
        r = sub["resp_p95"].values
        mean_waits.append(float(np.mean(w)))
        err_waits.append(float(1.96 * np.std(w) / np.sqrt(max(1, len(w)))))
        mean_resp.append(float(np.mean(r)))
        err_resp.append(float(1.96 * np.std(r) / np.sqrt(max(1, len(r)))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    x = np.arange(len(scheds))
    colors = ["#4C72B0" if s != "SABRE" else "#C44E52" for s in scheds]

    # Subplot 1: Average Waiting Time
    ax1.bar(x, mean_waits, yerr=err_waits, capsize=3, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scheds, rotation=45, ha="right")
    ax1.set_ylabel("Avg Wait Time (ms)")
    ax1.set_title("Overall Wait Time (E1 Attack)")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Interactive p95 Latency
    ax2.bar(x, mean_resp, yerr=err_resp, capsize=3, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scheds, rotation=45, ha="right")
    ax2.set_ylabel("Interactive p95 Response (ms)")
    ax2.set_title("Interactive Latency Protection (E1)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    _save_fig(fig, "fig_e1_bars")


def plot_e3_stealth(df_e3: pd.DataFrame) -> None:
    """Plot recall and Time-to-Detect vs attack stealth (RR+StaticThreshold vs SABRE)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.6))

    stealths = sorted(df_e3["stealth"].unique())

    for sched, col, marker in [("RR+StaticThreshold", "#4C72B0", "s"), ("SABRE", "#C44E52", "o")]:
        sub = df_e3[df_e3["scheduler"] == sched]
        recalls = [sub[sub["stealth"] == st]["instance_recall"].mean() for st in stealths]
        ttds = [sub[sub["stealth"] == st]["time_to_detect"].mean() for st in stealths]

        ax1.plot(stealths, recalls, label=sched, color=col, marker=marker, linewidth=1.5)
        ax2.plot(stealths, ttds, label=sched, color=col, marker=marker, linewidth=1.5)

    ax1.axvline(1.0, color="gray", linestyle=":", label="Static Threshold (1.0)")
    ax1.set_xlabel("Stealth Multiplier")
    ax1.set_ylabel("Instance Recall")
    ax1.set_title("Detection Recall vs Stealth (E3)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.set_xlabel("Stealth Multiplier")
    ax2.set_ylabel("Time-to-Detect (ms)")
    ax2.set_title("Censored TTD vs Stealth (E3)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    _save_fig(fig, "fig_e3_stealth")


def plot_e4_malicious_fraction(df_e4: pd.DataFrame) -> None:
    """Plot ADI vs malicious process fraction."""
    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    fracs = sorted(df_e4["malicious_fraction"].unique())
    scheds = ["RR", "RR+StaticThreshold", "CFS-like", "SABRE"]
    styles = {
        "RR": ("#55A868", "--", "^"),
        "RR+StaticThreshold": ("#4C72B0", "-.", "s"),
        "CFS-like": ("#8172B2", ":", "d"),
        "SABRE": ("#C44E52", "-", "o"),
    }

    for s in scheds:
        if s not in df_e4["scheduler"].unique():
            continue
        sub = df_e4[df_e4["scheduler"] == s]
        adis = [sub[sub["malicious_fraction"] == mf]["adi"].mean() for mf in fracs]
        col, ls, m = styles.get(s, ("black", "-", "x"))
        ax.plot(fracs, adis, label=s, color=col, linestyle=ls, marker=m, linewidth=1.5)

    ax.set_xlabel("Malicious Process Fraction")
    ax.set_ylabel("Attack Degradation Index (ADI)")
    ax.set_title("Interactive Degradation (E4)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    _save_fig(fig, "fig_e4_malicious_frac")


def plot_e7_ablation(df_e7: pd.DataFrame) -> None:
    """Plot ablation results for all disabled flags and burst-age signal."""
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    # Sort by impact
    means = df_e7.groupby("variant")["avg_wait"].mean().sort_values(ascending=False)
    variants = means.index.tolist()
    vals = means.values

    colors = ["#C44E52" if v == "Full SABRE" else "#4C72B0" for v in variants]
    y = np.arange(len(variants))

    ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(variants)
    ax.set_xlabel("Avg Wait Time under Attack (ms)")
    ax.set_title("Ablation Study (E7)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    _save_fig(fig, "fig_e7_ablation")


def plot_e8_overhead(df_e8: pd.DataFrame) -> None:
    """Plot overhead sensitivity and identify break-even frontier against RR."""
    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    multipliers = sorted(df_e8["cost_multiplier"].unique())
    sabre_waits = [df_e8[(df_e8["scheduler"] == "SABRE") & (df_e8["cost_multiplier"] == m)]["avg_wait"].mean() for m in multipliers]
    rr_wait = df_e8[df_e8["scheduler"] == "RR"]["avg_wait"].mean()

    ax.plot(multipliers, sabre_waits, label="SABRE", color="#C44E52", marker="o", linewidth=1.5)
    ax.axhline(rr_wait, label="Round Robin", color="#55A868", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Monitoring Cost Multiplier")
    ax.set_ylabel("Average Wait Time (ms)")
    ax.set_title("Monitoring Overhead Sensitivity (E8)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    _save_fig(fig, "fig_e8_overhead")


def plot_e9_pareto(df_e9: pd.DataFrame) -> None:
    """Plot Pareto frontier: Security benefit (1/ADI) vs Collateral Damage (LSI)."""
    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    sabre_points = df_e9[df_e9["scheduler"] == "SABRE"]
    static_points = df_e9[df_e9["scheduler"] == "RR+StaticThreshold"]

    ax.scatter(sabre_points["lsi"], 1.0 / np.maximum(1e-3, sabre_points["adi"]), color="#C44E52", alpha=0.7, label="SABRE Configurations", s=25)
    ax.scatter(static_points["lsi"], 1.0 / np.maximum(1e-3, static_points["adi"]), color="#4C72B0", marker="s", alpha=0.7, label="Static Threshold", s=30)

    ax.set_xlabel("Collateral Damage (LSI: Benign Slowdown)")
    ax.set_ylabel("Security Benefit (1 / ADI)")
    ax.set_title("Security vs Collateral Damage (E9)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    _save_fig(fig, "fig_e9_pareto")


def plot_classifier_confusion(matrix: np.ndarray, classes: list[str]) -> None:
    """Plot confusion matrix of behaviour classifier."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() / 2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Ground Truth Class")
    ax.set_title("Classifier Confusion Matrix")

    _save_fig(fig, "fig_classifier_confusion")


def plot_timeline(
    time_series_attacker: list[tuple[float, float, int]],  # (time, cpu_share, level)
    time_series_benign: list[tuple[float, float, int]],    # (time, cpu_share, level)
) -> None:
    """Plot 'money figure': attacker contained to Q3 / level 3 while benign CPU-bound stays at level 0."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 3.2), sharex=True)

    t_att = [x[0] for x in time_series_attacker]
    share_att = [x[1] for x in time_series_attacker]
    lvl_att = [x[2] for x in time_series_attacker]

    t_ben = [x[0] for x in time_series_benign]
    share_ben = [x[1] for x in time_series_benign]
    lvl_ben = [x[2] for x in time_series_benign]

    # CPU Share Subplot
    ax1.plot(t_att, share_att, label="Attacker (CPU Hog)", color="#C44E52", linewidth=1.5)
    ax1.plot(t_ben, share_ben, label="Benign CPU-Bound", color="#4C72B0", linewidth=1.5)
    ax1.axhline(0.60, color="gray", linestyle=":", label="Static Threshold (0.60)")
    ax1.set_ylabel("CPU Share")
    ax1.set_title("SABRE Dynamic Containment Timeline")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Risk Level Subplot
    ax2.step(t_att, lvl_att, label="Attacker Risk Level", color="#C44E52", linewidth=1.5, where="post")
    ax2.step(t_ben, lvl_ben, label="Benign Risk Level", color="#4C72B0", linewidth=1.5, where="post")
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(["L0: Normal", "L1: Warn", "L2: Throttled", "L3: Contained"])
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Risk Level")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="center right")

    _save_fig(fig, "fig_timeline")

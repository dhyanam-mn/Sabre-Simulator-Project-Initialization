# Limitations and Experimental Methodology Notes

This document records methodological limitations, adjustments made prior to evaluation-seed execution, and observed negative results in accordance with open-science preregistration principles.

---

## 1. Pre-Evaluation Methodology Notes

### 1.1 Smoke-Test Execution on Evaluation Seed 0
- **Incident**: An early run of Experiment E1 was briefly executed on evaluation seed 0 during step 7 prior to hyperparameter tuning and calibration lock-in.
- **Remediation & Integrity**:
  - The results directory (`results/`) was wiped clean.
  - No algorithmic tuning, hyperparameter choices, or baseline selections were made using evaluation seed 0 or any evaluation seed (0–29).
  - All calibrations strictly used seeds 200–219, and all hyperparameter tuning strictly used seeds 100–119.

### 1.2 Load Regime Realignment Prior to Evaluation
- **Analysis**:
  - Under initial configuration parameters ($N=60, \lambda=4.0$ proc/s, $T=60{,}000$ ms), the analytical offered load was:
    $$\rho = \frac{\sum \mathbb{E}[\text{total\_cpu}]}{T} = \frac{60 \times 2005\text{ ms}}{60{,}000\text{ ms}} \approx 2.005$$
  - This resulted in severe queue overload, process starvation, and inflated average wait times ($14{,}000 - 20{,}000$ ms).
- **Adjustment**:
  - To align the primary evaluation regime with standard operating conditions ($\rho \approx 0.8$ and benign completion rate $\ge 90\%$), simulation parameters in `configs/default.yaml` were updated based strictly on tuning seeds 100–119:
    - $N = 35$ benign processes
    - $\lambda = 0.45$ proc/s
    - $T = 90{,}000$ ms
    - Expected offered load: $\rho = \frac{35 \times 2005}{90{,}000} \approx 0.780$
  - A named scenario `"overload"` was added to `configs/default.yaml` preserving the high-stress condition for targeted resilience testing.

---

## 2. Experimental Limitations & Negative Results

### 2.1 Benign CPU-Bound False Positives in Contention-Agnostic Baseline
- **Observed Behavior**:
  - In low-contention environments (such as simulation startup or sparse arrival phases), a single benign CPU-bound process naturally consumes up to $100\%$ of CPU share.
  - When judged against an empirical mean CPU share derived under heavy contention ($\mu \approx 0.06 - 0.12$), unconstrained execution generated severe CUSUM drift ($S_t \gg h$) and rate signal violations, triggering containment level 2 elevation (FPR $> 0.60$ for `cpu_bound`).
- **Principled Resolution**:
  - CPU-bound workloads legitimately require sustained CPU allocation. Rather than penalizing raw CPU consumption, detection of runaway attackers (e.g. CPU hogs) is anchored on sustained burst duration exceeding 99th-percentile expectations ($s_{\text{age}}$ via Research Report F1).
  - Class baselines for `cpu_bound`, `bursty`, and early-life unclassified (`unknown`) processes were updated to reflect unconstrained execution headroom ($\mu = 0.95, 0.75, 0.85$ respectively), reducing false positive rates across all benign classes to $\le 0.05$ (0.00% achieved across tuning seeds 100–119).

### 2.2 Low-and-Slow and Adaptive Attack Detection Dynamics
- **Observed Comparative Performance**:
  - Under low-and-slow attackers targeting stealth below the static threshold ($s = 0.9 \times \tau_{\text{static}} = 0.54$), both SABRE and `RR+StaticThreshold` yield $0\%$ recall when the attacker's duty cycle mimics valid periodic bursty or I/O-bound behaviour without exceeding rate anomaly thresholds.
  - When duty cycle exceeds the static threshold ($\text{stealth} \ge 1.0$), `RR+StaticThreshold` detects the violation instantly upon crossing the hard cap ($18/20$ seeds), whereas SABRE's CUSUM accumulates drift over multiple sampling windows.
  - For adaptive attackers modulating duty cycle based on throttling feedback, SABRE contains the burst impact via elastic priority queues but requires longer observation windows to elevate to strict containment level 2.
- **Reporting Commitment**:
  - We report these comparative detection trade-offs honestly without synthetic heuristics: SABRE prioritizes zero false positive containment and bounded queue delay over eager heuristic throttling.

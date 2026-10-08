# SABRE Research Study: Pre-Registration of Hypotheses

**Title:** Security- and Behaviour-Aware, Risk-Elastic CPU Scheduler (SABRE)  
**Authors:** Dhyanam Mahajan (Role C), Khushi (Role A), Aadya (Role B)  
**Institution:** Vellore Institute of Technology (VIT), Vellore  
**Date of Pre-Registration:** 2026-10-09  
**Status:** Pre-registered before running evaluation seeds (Evaluation Seeds: 0–29 from `configs/seeds.yaml`)

---

## 1. Context and Protocol

All evaluation experiments (E1–E12) are evaluated across 30 disjoint seeds (`eval: 0–29`) specified in `configs/seeds.yaml`.
Calibration and baseline tuning are strictly performed on separate non-overlapping seeds (`calibration: 200–219`, `tuning: 100–119`) to prevent data leakage.
To control Family-Wise Error Rate (FWER) across multiple comparisons against baseline schedulers, we apply the step-down Holm-Bonferroni correction to all primary hypothesis tests.

---

## 2. Pre-Registered Hypotheses (H1 – H7)

### Hypothesis H1: Protection of Interactive Workload Latency (ADI)
- **Statement:** Under malicious attacks (E1), SABRE provides significantly superior protection of interactive job responsiveness compared to unmitigated and standard baselines (Round Robin, RR+StaticThreshold, and CFS-like). Specifically, the Attack Degradation Index (ADI = $\text{resp\_p95}_{\text{attack}} / \text{resp\_p95}_{\text{no\_attack}}$) under SABRE is lower than under RR, RR+StaticThreshold, and CFS-like.
- **Metric:** Attack Degradation Index (ADI) for interactive jobs ($\text{resp\_p95}$).
- **Comparison / Baselines:** SABRE vs. RR, SABRE vs. RR+StaticThreshold, SABRE vs. CFS-Like.
- **Statistical Test:** Paired Wilcoxon signed-rank test across 30 evaluation seeds.
- **Threshold / Effect Size:** $p < 0.05$ after Holm-Bonferroni correction; non-parametric effect size Cliff's $\delta \ge 0.33$ (medium-to-large effect size).
- **Multiple-Testing Correction:** Holm-Bonferroni across the family of comparative baselines.

### Hypothesis H2: Sub-Threshold Low-and-Slow Attack Detection (CUSUM)
- **Statement:** SABRE detects duty-cycled low-and-slow attacks operating near or below the static detection threshold ($\text{stealth} \in [0.9, 1.0]$ in E3) with substantially higher recall than a naive static CPU-share threshold baseline, due to sequential CUSUM drift accumulation.
- **Metric:** Attack instance-level detection recall (and censored Time-to-Detect via Kaplan-Meier estimator).
- **Comparison / Baselines:** SABRE vs. RR+StaticThreshold.
- **Statistical Test:** Paired Wilcoxon signed-rank test on recall over 30 evaluation seeds.
- **Threshold / Effect Size:** SABRE recall statistically significantly higher than RR+StaticThreshold ($p < 0.05$); detection rate $> 80\%$ at stealth = 0.9.
- **Multiple-Testing Correction:** Pre-declared primary metric for E3.

### Hypothesis H3: Low False-Positive Rate and Benign Overhead Under Workload Stress
- **Statement:** Under a benign-heavy workload consisting of 70% CPU-bound processes with zero malicious actors (E5), SABRE limits false containment actions.
- **Metric:** False Positive Rate (FPR $\le 0.05$) and Legacy Slowdown Index (LSI: benign CPU-bound turnaround under SABRE vs. RR no-attack).
- **Comparison / Baselines:** SABRE vs. ideal threshold ($0.05$).
- **Statistical Test:** One-sample t-test / bootstrap 95% confidence interval for FPR $\le 0.05$; paired t-test for LSI within $\pm 10\%$ of 1.0 (RR baseline).
- **Threshold / Effect Size:** Mean FPR $\le 5\%$, LSI $\le 1.10$.
- **Multiple-Testing Correction:** Standalone stress test (E5).

### Hypothesis H4: Monitoring Cost Break-Even Frontier
- **Statement:** There exists an empirical cost-multiplier break-even point in E8 where charged scheduler overhead outweighs containment benefits, establishing the feasibility boundary of dynamic software-based CPU risk monitoring.
- **Metric:** Overall average waiting time and interactive p95 latency under scaled cost factors $\in \{0.5\times, 1.0\times, 2.0\times, 5.0\times, 10.0\times\}$.
- **Comparison / Baselines:** SABRE vs. Round Robin.
- **Statistical Test:** Identification and reporting of intersection / break-even point with 95% bootstrap confidence intervals.
- **Threshold / Effect Size:** Break-even factor identified and documented honestly (expected break-even $> 2.0\times$ default cost model).
- **Multiple-Testing Correction:** Exploratory sensitivity analysis.

### Hypothesis H5: Containment of Adaptive and Gaming Attackers
- **Statement:** An adaptive attacker that duty-cycles and reacts to throttling delays (E6) achieves a lower long-run CPU resource share and lower attacker utility under SABRE than under static threshold containment.
- **Metric:** Long-run attacker CPU share ($\text{attacker\_cpu\_share}$) and cumulative attacker utility (CPU gained minus penalty).
- **Comparison / Baselines:** SABRE vs. RR+StaticThreshold.
- **Statistical Test:** Paired Wilcoxon signed-rank test across 30 evaluation seeds.
- **Threshold / Effect Size:** Attacker CPU share under SABRE is strictly lower than under RR+StaticThreshold ($p < 0.05$, Cliff's $\delta \ge 0.40$).
- **Multiple-Testing Correction:** Pre-declared primary metric for E6.

### Hypothesis H6: Burst-Age Signal Necessity for Cryptominer/Hog Separation
- **Statement:** The burst-age feature ($s_{\text{age}} = \text{clip}((\text{burst\_age\_ms} - p99_{\text{class}})/\text{scale}, 0, 1)$) is necessary to differentiate a persistent unyielding CPU hog from a legitimate benign CPU-bound process. Disabling this signal in ablation (E7) causes CPU hog recall to collapse.
- **Metric:** Process-level and instance-level recall on CPU hog workloads.
- **Comparison / Baselines:** Full SABRE vs. SABRE with `burst_age` signal disabled.
- **Statistical Test:** Paired Wilcoxon signed-rank test across 30 evaluation seeds.
- **Threshold / Effect Size:** Hog recall with burst-age $\ge 90\%$, without burst-age $< 30\%$ ($p < 0.001$).
- **Multiple-Testing Correction:** Ablation family test.

### Hypothesis H7: Graceful Degradation Under Behaviour Classifier Noise
- **Statement:** Under 20% random classifier label perturbation (E11), SABRE maintains robust containment due to multi-signal noisy-OR evidence fusion and lineage tracking, suffering an instance-level recall drop of less than 15 percentage points.
- **Metric:** Attack instance-level detection recall under 0% vs. 20% classifier noise.
- **Comparison / Baselines:** Full SABRE (baseline) vs. SABRE under 20% classifier noise.
- **Statistical Test:** Paired t-test / Wilcoxon signed-rank test on $\Delta \text{recall}$.
- **Threshold / Effect Size:** Mean recall reduction $\le 15.0$ percentage points ($95\%\text{ CI}$ upper bound $< 15\%$).
- **Multiple-Testing Correction:** Robustness family test.

---

## 3. Seed Allocation Manifest

- **Tuning Seeds (`configs/seeds.yaml`):** 100 – 119 (20 seeds) — strictly for parameter tuning.
- **Calibration Seeds (`configs/seeds.yaml`):** 200 – 219 (20 seeds) — strictly for benign baseline profiling ($\mu, \sigma$).
- **Evaluation Seeds (`configs/seeds.yaml`):** 0 – 29 (30 seeds) — strictly for evaluation of H1 – H7.

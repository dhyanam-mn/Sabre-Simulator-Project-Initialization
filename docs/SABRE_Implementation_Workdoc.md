# SABRE: Security- And Behaviour-aware, Risk-Elastic CPU Scheduler
## Implementation Work Document (team of 3, AI-assistant ready)

Project: *Security- and Behaviour-Aware Predictive CPU Scheduling with Adaptive Resource Allocation* (VIT Vellore, Review 1 paper).
Team: Khushi Srivastava (Role A), Aadya Singh (Role B), Dhyanam Mahajan (Role C). Roles can be swapped, but keep the three-way split.

> **How to use this document with any AI:** paste Section 0 (Master Prompt) first, then paste ONE module section (Sections 5 to 9) per request. Each module has a fixed interface contract (Section 3), so three people can work in parallel with different AIs and the code still plugs together.

---

## 0. Master Prompt (paste this first into any AI)

```
You are a senior Python engineer. We are building a deterministic discrete-event
CPU-scheduling SIMULATOR (not a kernel module) called SABRE. Rules:
1. Python 3.10+, only numpy, pandas, scipy, matplotlib, pytest, pyyaml (streamlit optional).
2. Everything is deterministic given a seed (use numpy.random.Generator(seed), never global random).
3. Follow the repo layout and the interface contracts in the spec EXACTLY. Do not rename
   classes, fields or method signatures.
4. Write type hints, docstrings, and pytest tests for each module. Code must run with
   `pytest -q` passing before you say you are done.
5. Time unit is milliseconds (float). All tunable numbers live in configs/default.yaml,
   never hard-coded.
6. If something in the spec is ambiguous, pick the simplest option, state the assumption in a
   comment at the top of the file, and continue. Do not ask me questions.
7. Output complete files, not fragments.
Acknowledge, then wait for the module I paste.
```

---

## 1. What is novel (claimable contributions)

The Review-1 paper identifies the gap (prediction, classification and risk scoring never combined). This document turns that into six concrete, testable ideas, which are what make the work more than "RR plus a detector":

| # | Novel element | Why it is new / useful |
|---|---|---|
| N1 | **Class-conditional risk scoring**: risk is deviation from what is normal *for the process's behaviour class*, not from a global threshold | A video encoder (CPU-bound, benign) is not penalised; a "interactive" process suddenly hogging is |
| N2 | **CUSUM drift detector for low-and-slow abuse** | Catches a process that creeps upward while staying under any static threshold (existing CPU-usage detectors use fixed thresholds) |
| N3 | **Lineage-aware risk**: children inherit parent risk; a process-tree budget caps descendants | Stops fork-bomb variants that spread work over many short-lived children, each of which looks harmless |
| N4 | **Graduated, reversible, work-conserving containment ladder** (levels 0 to 3, hysteresis, guaranteed minimum share, caps only enforced when others are waiting) | Security action cannot starve anyone and cannot waste idle CPU; false positives recover |
| N5 | **Two new metrics**: *Legitimate Slowdown Index (LSI)* and *Attack Damage Index (ADI)*, plus a security-vs-performance Pareto plot | Quantifies the exact trade-off the paper says is the main motivation |
| N6 | **Adaptive threshold-probing attacker** in the evaluation | Most papers test static attackers. This one learns from its own throttling and tries to evade |

Optional N7 (stretch): **Explainable decision log**: every scheduling decision stores reason codes (e.g., `LVL2_CUSUM+LINEAGE`), which enables a "why was my process slowed" audit view.

---

## 2. Repository layout

```
sabre/
  configs/default.yaml            # all parameters (Section 11)
  sabre/
    core/                         # ROLE A
      process.py                  # Process, ProcState
      engine.py                   # Simulator (event loop)
      workload.py                 # benign generator
      attacks.py                  # 3 attacks + adaptive attacker
      metrics.py                  # perf + LSI/ADI + detection metrics
    schedulers/
      base.py                     # Scheduler abstract class        (A)
      baselines.py                # FCFS, SJF, RR, Priority, MLFQ, RR+StaticThreshold (A)
      sabre.py                    # proposed scheduler              (C)
      controller.py               # adaptive controller             (C)
    intel/                        # ROLE B
      features.py                 # FeatureExtractor
      predictor.py                # EMA + optional online model
      classifier.py               # behaviour classifier
      risk.py                     # RiskScorer (N1, N2, N3)
      calibrate.py                # learns class baselines from benign runs
    experiments/                  # ROLE C
      run_experiment.py           # CLI runner
      stats.py                    # CI, paired tests
      plots.py
    dashboard/app.py              # optional streamlit
  tests/
  results/                        # CSVs + figures (git-ignored)
  report/                         # paper sections, figures
```

---

## 3. Interface contracts (do not change; this is what lets 3 people work in parallel)

```python
# core/process.py
from dataclasses import dataclass, field
from enum import Enum
class ProcState(Enum): NEW=0; READY=1; RUNNING=2; WAITING=3; DONE=4

@dataclass
class Process:
    pid: int
    ppid: int | None
    arrival: float
    kind: str            # benign kinds: "interactive","io_bound","cpu_bound","bursty"
                         # malicious: "forkbomb","hog","lowslow","adaptive","forkchild"
    label: int           # ground truth 0 benign / 1 malicious (forkchild = 1)
    total_cpu: float     # total CPU demand; inf allowed for hog
    state: ProcState = ProcState.NEW
    remaining_burst: float = 0.0       # remaining time in current CPU burst
    cpu_used: float = 0.0
    first_run: float | None = None
    finish: float | None = None
    wait_time: float = 0.0
    last_ready_at: float = 0.0
    # hooks: engine calls these to get the next phase
    def next_cpu_burst(self, now: float, rng) -> float: ...
    def next_io_wait(self, now: float, rng) -> float: ...
    def children_to_spawn(self, cpu_delta: float, now: float, rng) -> list["Process"]: ...

# schedulers/base.py
class Scheduler:
    name: str
    def on_arrival(self, p: Process, now: float) -> None: ...
    def on_ready(self, p: Process, now: float) -> None: ...      # after IO or preemption
    def on_slice_end(self, p: Process, ran: float, burst_done: bool, now: float) -> None: ...
    def on_exit(self, p: Process, now: float) -> None: ...
    def pick_next(self, now: float) -> tuple[Process, float] | None: ...  # (process, quantum)
    def overhead_ms(self) -> float: ...        # monitoring cost accrued since last call, then reset
    def flagged(self) -> dict[int, float]: ... # pid -> first time level>=2 (for detection metrics)
    def decision_log(self) -> list[dict]: ...  # optional explainability

# intel/features.py
@dataclass
class Features:
    mean_burst: float; cv_burst: float; io_ratio: float
    cpu_share: float          # share of wall time on CPU over last T_share ms
    full_quantum_frac: float  # fraction of recent slices that used the whole quantum
    fork_rate: float          # children per 100 ms
    n_descendants: int

# intel/predictor.py      class BurstPredictor:  predict(pid)->float ; update(pid, observed_burst)
# intel/classifier.py     class BehaviourClassifier: classify(f: Features)->str  (4 classes)
# intel/risk.py           class RiskScorer: update(pid, f, cls, now)->float in [0,1]; level(pid)->int 0..3
#                         inherit(child_pid, parent_pid); reasons(pid)->list[str]
```

---

## 4. Build order and milestones (8 weeks)

| Week | A (Khushi): engine | B (Aadya): intelligence | C (Dhyanam): scheduler + experiments |
|---|---|---|---|
| 1 | Process, engine skeleton, FCFS and RR working | FeatureExtractor + EMA predictor | Repo, config loader, CI with pytest, metrics skeleton |
| 2 | Benign workload generator, SJF, Priority, MLFQ | Rule-based classifier + tests | SABRE controller v0 (class + predicted burst only, no risk) |
| 3 | 3 attacks (forkbomb, hog, lowslow) | RiskScorer v0 (rate z-scores, decay) | Integrate B into controller, first end-to-end run |
| 4 | Adaptive attacker, RR+StaticThreshold baseline | CUSUM (N2) + lineage (N3) + calibrate.py | Containment ladder with hysteresis + elastic caps (N4) |
| 5 | Metrics: LSI, ADI, TTD, Jain | Optional online ML predictor / decision-tree classifier | Experiment runner, 30 seeds, stats |
| 6 | Trace-driven workload loader (optional) | Threshold/weight tuning sweeps | Ablations, sweeps E1 to E7 |
| 7 | Bug-fixing, performance profiling | Sensitivity analysis | Plots, Pareto, dashboard (optional) |
| 8 | Freeze code, README | Freeze calibration | Paper results sections, final report |

**Integration checkpoints:** end of week 3 (first full run), end of week 5 (all experiments runnable), end of week 7 (code freeze).

---

## 5. MODULE A: Simulator core (Role A, Khushi)

### 5.1 Engine semantics (single core first; multi-core is a stretch goal)
- Event-driven loop with a min-heap of events: `ARRIVAL`, `IO_COMPLETE`, `SLICE_END`, `SPAWN`.
- Context switch cost `ctx_switch_ms` (default 0.05) is charged whenever the running pid changes.
- When CPU is idle, jump time to the next event.
- Loop: `pick_next()` returns `(p, q)`. Run for `ran = min(q, p.remaining_burst)`. Call `children_to_spawn(ran, ...)` and schedule the children as arrivals. If the burst finished, ask `next_io_wait`; if `cpu_used >= total_cpu`, mark DONE, else go WAITING and schedule `IO_COMPLETE`. If the burst is unfinished, the process is preempted and goes back to ready. Always call `sched.on_slice_end(...)`.
- Add the scheduler's `overhead_ms()` to the clock (monitoring is charged as real CPU time that nobody else uses).
- Hard stop: `sim_end_ms` (default 60000) or all benign processes done. Processes still alive at the end are recorded with `finish=None`.
- Wait time = time in READY. Response time = first_run minus arrival, and for interactive processes also per-burst response (time from IO completion to the next dispatch). Record per-burst response samples for percentile metrics.

### 5.2 Benign workload (`workload.py`)
Generate `N_benign` (default 60) processes with arrivals from a Poisson process (rate configurable). Four kinds:

| Kind | CPU burst (ms) | IO wait (ms) | total_cpu (ms) |
|---|---|---|---|
| interactive | lognormal mean 2, sigma 0.5 | exponential mean 40 | 200 to 1000 |
| io_bound | lognormal mean 8 | exponential mean 25 | 300 to 1500 |
| cpu_bound | lognormal mean 150, sigma 0.4 | exponential mean 3 | 2000 to 8000 |
| bursty | mixture: 70% mean 3, 30% mean 80 | exponential mean 15 | 500 to 3000 |

Default mix: 30% interactive, 25% io_bound, 25% cpu_bound, 20% bursty. Include a "benign-heavy" workload preset (70% cpu_bound, e.g., compilers/encoders) for the false-positive stress test.

### 5.3 Attacks (`attacks.py`), all with `attack_start_ms` and an `intensity` parameter
1. **forkbomb**: while it holds the CPU, spawns one `forkchild` per `1000/rate` ms of CPU consumed (rate default 50 children/s at intensity 1; scales with intensity). Each child is a short CPU spinner (burst 5 ms, total 20 to 50 ms, label 1). Cap total descendants at `max_descendants` (default 2000).
2. **hog**: one process, a single infinite CPU burst, no IO (a cryptominer). Optionally N hog processes.
3. **lowslow**: duty-cycled process with target CPU share `s = stealth × τ_static` where `τ_static` is the static detection threshold (default 0.6 over 500 ms) and `stealth` is swept in {0.7, 0.9, 1.0, 1.1}. Variant **drift**: starts at 0.5×τ and increases share by 1% every 500 ms (tests CUSUM).
4. **adaptive** (N6): a lowslow attacker that measures its own recent effective CPU share and wait time. Every 500 ms: if its wait/burst ratio rose (it is being throttled), it cuts its duty by 20%. If not throttled for 2 windows, it raises duty by 5%. It never sees SABRE internals. Tests whether the containment ladder can be gamed.

Ground truth: `label=1` for every malicious process including `forkchild`.

### 5.4 Baselines (`baselines.py`)
FCFS, SJF (non-preemptive, using *true* next burst as an oracle upper bound AND a variant using EMA prediction), RR (q = 10 ms, also a q-sweep), static Priority (priority by kind, with aging off), MLFQ (3 levels, standard demotion; extra baseline), and **RR+StaticThreshold**: Round Robin plus a naive detector that throttles any process whose CPU share over 500 ms exceeds τ_static (in the spirit of the CPU-usage detectors in [6], [14]). This last baseline is essential: it separates "the benefit of any security awareness" from "the benefit of SABRE's design".

### 5.5 Metrics (`metrics.py`)
Per run, return a dict: `avg_wait, avg_turnaround, throughput, resp_p50/p95/p99 (interactive burst responses), jain_fairness (over benign processes), ctx_switches, overhead_ms, overhead_pct`.
Security: process-level `precision, recall, f1, fpr` (flagged = `scheduler.flagged()` contains the pid), `time_to_detect` (flag time minus attack start, median over attackers), `attacker_cpu_share` (share of total CPU taken by malicious processes).
New: 
- **LSI** = mean turnaround of benign *cpu_bound* processes under scheduler X with attack, divided by the same under RR with **no** attack and the same seed. (Measures collateral damage; ideal is 1.0, lower than RR-under-attack is good.)
- **ADI** = interactive `resp_p95` under attack divided by interactive `resp_p95` with no attack, same scheduler, same seed. (Measures how well the scheduler protects users; 1.0 is perfect.)

**Tests for A:** deterministic repeat (same seed gives identical metrics); FCFS on a hand-built 3-process example matches hand-computed waiting times; RR conserves total CPU time; no process finishes before arrival plus total_cpu.

**AI prompt for A:** "Implement Module A from the spec: core/process.py, engine.py, workload.py, attacks.py, metrics.py, schedulers/base.py and baselines.py, with pytest tests. Use the interface contract exactly."

---

## 6. MODULE B: Intelligence (Role B, Aadya)

### 6.1 FeatureExtractor
Keeps per-pid ring buffers of the last `W=8` bursts, IO waits, and slices. Update on `on_slice_end`. Compute:
`mean_burst`, `cv_burst` (std/mean), `io_ratio = io/(io+cpu)` over the window, `cpu_share` over `T_share=500 ms` (sliding, from slice records), `full_quantum_frac`, `fork_rate` (children per 100 ms over last 500 ms), `n_descendants` (via parent links). Cost is charged: `cost_feature=0.02 ms` per update.

### 6.2 BurstPredictor
- Baseline: `τ_{n+1} = α·t_n + (1−α)·τ_n`, α = 0.5, initial τ = class prior (interactive 2, io 8, cpu 150, bursty 10).
- Optional (stretch): online linear/SGD regressor on `[last 4 bursts, io_ratio, cv]` with `partial_fit`; cost `0.1 ms`. Add a flag `predictor: ema|online`.
- Report MAE/MAPE of prediction in the results (a sanity table: EMA vs oracle vs online).

### 6.3 BehaviourClassifier (rule-based first; decision tree optional)
```
if mean_burst < 5 and io_ratio > 0.6:            "interactive"
elif cv_burst > 1.0:                             "bursty"
elif io_ratio > 0.5:                             "io_bound"
elif io_ratio < 0.1 and full_quantum_frac > 0.8: "cpu_bound"
else: argmin distance to class centroids (from calibrate.py)
```
Needs a minimum of 3 observed bursts, otherwise class = "unknown" (treated as the "normal" class with a mild prior). Measure classifier accuracy against the true benign `kind` (report confusion matrix). Optional: train `sklearn`-free decision tree from benign calibration runs.

### 6.4 calibrate.py (class baselines)
Run several benign-only simulations (e.g., 20 seeds, **separate from the evaluation seeds**) under RR. For each class `c` and signal `s ∈ {cpu_share, fork_rate}` store `mu[c][s]` and `sigma[c][s]` (floor sigma to avoid divide-by-zero) in `baselines.json`. This is how risk becomes class-conditional.

### 6.5 RiskScorer (N1, N2, N3). The main novelty, so implement carefully
Per process, every update (each slice end, with a minimum spacing of 50 ms between updates):

1. **Rate signals (N1):**
   `z_cpu = (cpu_share − mu[c].cpu_share) / sigma[c].cpu_share`
   `z_fork = (fork_rate − mu[c].fork_rate) / sigma[c].fork_rate`
   `s_cpu = clip(z_cpu/ Z_max, 0, 1)`, `s_fork = clip(z_fork/ Z_max, 0, 1)`, `Z_max = 4`.
2. **Drift signal (N2, CUSUM):** per process, on the sequence `x_t = cpu_share` sampled every 100 ms:
   `S_t = max(0, S_{t−1} + (x_t − mu[c].cpu_share − k))`, `k = 0.5·sigma[c].cpu_share`.
   `s_drift = clip(S_t / h, 0, 1)`, with `h = 5·sigma`. Reset S to 0.5·S when risk level drops back to 0 (so recovery is possible).
3. **Combine:** `inst = 1 − (1−w1·s_fork)(1−w2·s_cpu)(1−w3·s_drift)` (noisy-OR, so any strong signal raises risk; weights default 0.9, 0.7, 0.8).
4. **Decay (temporal):** `R_t = max(inst, λ·R_{t−1})`, λ = 0.97 per update, so risk rises instantly and falls slowly.
5. **Lineage (N3):** `R_eff(p) = max(R(p), γ·R_eff(parent))`, γ = 0.9. Also a **tree budget**: if `n_descendants > D_max` (default 30 concurrent), the tree root's `s_fork = 1`. A child's initial risk equals `γ·R_eff(parent)` through `inherit()`.
6. **Trust discount (optional ablation):** `trust` grows by 0.01 per update while `R<0.2` (cap 0.5). `R_final = R_eff·(1 − trust)`. Reset trust to 0 upon any level ≥ 2. (Caveat to evaluate: can a patient attacker "farm" trust? CUSUM should still catch drift. Report it either way.)
7. **Ladder with hysteresis (N4 input):** level thresholds `[0.30, 0.55, 0.80]`. Upgrade requires the threshold exceeded for 2 consecutive updates. Downgrade requires `R < threshold − 0.10` for 3 consecutive updates. `level(pid)` returns 0 to 3. `reasons(pid)` returns which signals exceeded 0.3.
8. Cost charged: `0.02 ms` per update.

**Tests for B:** on a synthetic stationary benign series, false alarms (level ≥ 2) per 1000 updates < 1%; a step increase in cpu_share triggers level ≥ 2 within 1 s; slow ramp that never crosses τ_static triggers via CUSUM; a CPU-bound process at its own normal share stays at level 0 even though share is 0.9; a child of a level-3 parent starts at level ≥ 2; risk decays to level 0 after the behaviour stops.

**AI prompt for B:** "Implement Module B from the spec: intel/features.py, predictor.py, classifier.py, risk.py, calibrate.py, with pytest tests (use synthetic series since the engine may not exist yet). Follow the interface contract exactly."

---

## 7. MODULE C1: SABRE scheduler and adaptive controller (Role C, Dhyanam)

### 7.1 Queues
Four ready queues: `Q0` (interactive / short predicted burst), `Q1` (normal), `Q2` (long / CPU-bound), `Q3` (contained, risk level ≥ 2).
Class and prediction assignment for risk level 0 or 1:
- `Q0` if class == interactive or predicted burst < `short_thr` (default 5 ms)
- `Q2` if class == cpu_bound or predicted burst > `long_thr` (default 50 ms)
- `Q1` otherwise
Risk level 2 or 3 always goes to `Q3`, regardless of class. Level 1 processes stay in their queue but receive a smaller quantum and priority penalty.

### 7.2 Quantum rule
```
q = q_base * class_factor[class] * pred_factor(pred) * level_factor[level]
  class_factor  = {interactive:0.5, io_bound:0.75, bursty:0.75, cpu_bound:1.5, unknown:1.0}
  pred_factor   = clip(pred / pred_ref, 0.5, 2.0)      # pred_ref=10 ms
  level_factor  = [1.0, 0.75, 0.5, 0.25]
q is clipped to [q_min=1 ms, q_max=40 ms]
```

### 7.3 Dispatch (strict priority with aging, plus a guaranteed slot for Q3)
- Pick from Q0, then Q1, then Q2 (shortest predicted burst first within a queue, to approximate SJF).
- **Aging:** a process waiting more than `age_ms` (default 200) in Q1/Q2 is promoted one queue (stops starvation of long jobs).
- **Guaranteed service (never zero):** every `slot_period` dispatches (default 10), if Q3 is non-empty, dispatch from Q3 instead. If Q0 to Q2 are all empty, Q3 is served freely (work-conserving).
- **Elastic share cap (N4):** a process at level L may use at most `cap[L] = [1.0, 0.5, 0.25, 0.10]` of CPU over a 200 ms sliding window, enforced **only if other processes are waiting**. If it exceeds the cap it is skipped for this dispatch round (but the min-share slot rule above still guarantees ≥ `min_share=2%`).
- Interactive boost: a process returning from IO whose class is interactive or whose predicted burst < `short_thr` can preempt a running process in Q1/Q2/Q3 (one preemption per `preempt_gap` = 2 ms to limit context-switch inflation).

### 7.4 Wiring (`sabre.py`)
Implements `Scheduler`. On each `on_slice_end`: `features.update` → `predictor.update` → `classifier.classify` → `risk.update` (all costs summed into `overhead_ms()`). On fork (child arrival with `ppid`): `risk.inherit`. `flagged()` records the first time a pid reached level ≥ 2. `decision_log()` stores `{t, pid, queue, quantum, level, class, pred, reasons}` (N7).

### 7.5 Feature flags (for ablations), in config under `sabre.flags`
`use_predictor, use_classifier, use_risk, use_lineage, use_cusum, use_hysteresis, use_trust, use_elastic_cap, use_aging`. With all false, SABRE must reduce to near-RR behaviour (test this).

**Tests for C1:** with `use_risk=false`, no process is ever in Q3; contained process still gets ≥ ~2% CPU while others are busy (starvation test); a lone contained process uses 100% CPU (work conservation); hysteresis prevents level flapping (count level changes in a noisy-but-benign series); aging promotes a starving cpu_bound.

**AI prompt for C1:** "Implement Module C1 from the spec: schedulers/controller.py and schedulers/sabre.py, plus tests. Assume intel/* and core/* exist with the contracts from Section 3; write minimal stubs if they are missing."

---

## 8. MODULE C2: Experiments, statistics, plots (Role C, Dhyanam)

### 8.1 Runner
`python -m sabre.experiments.run_experiment --exp E1 --seeds 30 --out results/E1.csv` writes one row per (scheduler, scenario, seed) with every metric from Section 5.5. Every scenario is run **with and without attack** on the same seed so ADI and LSI can be computed. Use identical workloads and arrival patterns for all schedulers on a seed.

### 8.2 Experiments

| ID | Question | Design |
|---|---|---|
| E1 | Overall comparison | 7 schedulers (FCFS, SJF-oracle, SJF-EMA, RR, Priority, MLFQ, RR+StaticThreshold) vs SABRE; default mix, 10% malicious (mixed types) |
| E2 | Attack intensity | Fork-bomb rate ×{0.5,1,2,4}; hog count {1,2,4,8} |
| E3 | Stealth sweep | lowslow `stealth` ∈ {0.7, 0.9, 1.0, 1.1}; plus drift variant. Report recall and TTD, for RR+StaticThreshold vs SABRE (N2) |
| E4 | Malicious fraction | {0, 5, 10, 20, 30}% |
| E5 | False-positive stress | benign-heavy preset (70% cpu_bound), **no attackers**. Report FPR, LSI (N1 test) |
| E6 | Adaptive attacker | adaptive attacker vs static-threshold baseline vs SABRE; attacker CPU share and ADI over time (N6) |
| E7 | Ablation | all single-flag removals of Section 7.5; each against the full SABRE on E1 (shows what each novelty contributes) |
| E8 | Overhead sensitivity | cost model ×{0.5, 1, 2, 5, 10}; find the break-even where SABRE loses to RR (the paper's stated risk) |
| E9 | Pareto | sweep level thresholds and caps (grid), plot ADI (security benefit) vs LSI (collateral damage); SABRE should dominate RR+StaticThreshold if the design works |
| E10 (optional) | Real traces | benign processes from a public workload trace (e.g., Parallel Workloads Archive SWF files), same attacks injected |

### 8.3 Statistics (`stats.py`)
Report mean and 95% CI (t-distribution) over 30 seeds. For SABRE vs each baseline use a **paired** Wilcoxon signed-rank test on per-seed metrics and report effect size (Cliff's delta). Never report a single run. Report negative results honestly; the paper's framing is the *trade-off*, not "always best".

### 8.4 Plots (matplotlib, saved as PDF + PNG, 300 dpi, IEEE-friendly single-column widths)
Bar+CI of waiting time and interactive p95 (E1); line plots of recall and TTD vs stealth (E3); ADI vs malicious fraction (E4); ablation bar chart (E7); overhead break-even (E8); Pareto scatter (E9); timeline of one run showing level changes and CPU share of an attacker vs a benign CPU-bound process (the "money figure" for the paper); confusion matrix of the classifier.

### 8.5 Optional dashboard
Streamlit app: choose scenario, run it, see Gantt-style timeline, per-process risk and level over time, and the decision log ("why was pid 42 contained?").

**AI prompt for C2:** "Implement Module C2: experiments/run_experiment.py, stats.py, plots.py exactly as specified in Section 8, using the Simulator and schedulers from Section 3."

---

## 9. Cost model for overhead (so monitoring is *charged*, not free)

| Operation | Cost (ms of CPU) |
|---|---|
| Feature update | 0.02 |
| EMA prediction | 0.01 |
| Online ML prediction | 0.10 |
| Classification | 0.01 |
| Risk update | 0.02 |
| Controller decision (queue and quantum) | 0.01 |

Overhead is added to the simulation clock via `overhead_ms()` and reported as `overhead_pct` of total time. Sensitivity in E8.

---

## 10. Acceptance criteria ("done" means all of these)

1. `pytest -q` passes; every experiment is reproducible from `--seed`.
2. With no attack, SABRE's average waiting time is within ±10% of RR (it should not hurt the benign case). If not, tune before proceeding.
3. SABRE with all flags off ≈ RR (within 5%).
4. Starvation test passes: the contained process never falls below the minimum share while others are busy.
5. E1 to E9 each produce a CSV and a figure; stats table with CIs exists.
6. Hypotheses to *test* (not to force): H1 SABRE lowers interactive p95 under attack vs RR and RR+StaticThreshold (lower ADI). H2 SABRE detects lowslow at stealth 0.9 to 1.0 with higher recall than the static threshold (CUSUM). H3 SABRE keeps FPR under 5% and LSI near RR on the benign-heavy preset. H4 The overhead break-even in E8 is reported. H5 The adaptive attacker's long-run CPU share is lower under SABRE than under the static baseline. Report whichever way each comes out.

---

## 11. configs/default.yaml (starting values)

```yaml
sim: {sim_end_ms: 60000, ctx_switch_ms: 0.05, n_benign: 60, arrival_rate_per_s: 4}
attack: {start_ms: 5000, malicious_fraction: 0.10, forkbomb_rate: 50, max_descendants: 2000,
         static_threshold: 0.6, static_window_ms: 500, stealth: 0.9}
features: {W: 8, T_share_ms: 500, update_spacing_ms: 50}
predictor: {type: ema, alpha: 0.5, priors: {interactive: 2, io_bound: 8, cpu_bound: 150, bursty: 10}}
risk: {Z_max: 4, weights: [0.9, 0.7, 0.8], lambda: 0.97, gamma: 0.9, D_max: 30,
       cusum_k_sigma: 0.5, cusum_h_sigma: 5, level_thresholds: [0.30, 0.55, 0.80],
       up_hold: 2, down_hold: 3, down_margin: 0.10, trust_gain: 0.01, trust_cap: 0.5}
sabre: {q_base: 10, q_min: 1, q_max: 40, short_thr: 5, long_thr: 50, pred_ref: 10,
        age_ms: 200, slot_period: 10, caps: [1.0, 0.5, 0.25, 0.10], cap_window_ms: 200,
        min_share: 0.02, preempt_gap_ms: 2, level_factor: [1.0, 0.75, 0.5, 0.25]}
rr: {quantum_ms: 10}
costs: {feature: 0.02, ema: 0.01, online: 0.10, classify: 0.01, risk: 0.02, decision: 0.01}
experiments: {seeds: 30, calib_seeds: 20}   # calibration seeds must not overlap eval seeds
```

---

## 12. Common pitfalls (tell your AI to avoid these)

- **Leakage:** do not calibrate class baselines on the evaluation seeds.
- **Unfair comparison:** all schedulers must see identical arrivals, bursts and attacks on a seed. Generate the workload once per seed, then deep-copy per scheduler.
- **Free monitoring:** forgetting to charge overhead makes SABRE look unrealistically good.
- **Oracle SJF** uses true bursts; label it clearly as an upper bound, not a fair competitor.
- **Starvation / flapping:** always test the min-share rule and hysteresis.
- **Infinite loops with hog (`total_cpu=inf`):** engine needs `sim_end_ms`, and metrics must handle `finish=None`.
- **Tuning on the test seeds:** tune on a separate tuning seed set (e.g., 100 to 119), evaluate on 0 to 29.

---

## 13. Mapping to later reviews and the paper

- **Review 2:** working simulator + baselines + 3 attacks + first results from E1 and E3.
- **Review 3 / final:** full E1 to E10, ablation, Pareto, adaptive attacker, overhead break-even, decision-log demo, and a paper whose Section V "Methodology" and Section VI "Results" are filled from this document and `results/`.
- **Claims to make in the paper:** N1 to N6 (Section 1) each backed by one experiment: N1 by E5, N2 by E3, N3 by E2 (forkbomb), N4 by starvation/work-conservation tests and E4, N5 by E9, N6 by E6. The ablation E7 ties everything together.

---

## 14. Stretch goals (only if ahead of schedule)

1. Multi-core extension with per-core queues and migration.
2. "Live advisory mode": read real process stats with `psutil` on Linux, compute risk, and *print* recommended `nice` values (no enforcement), then validate on a real fork-bomb-like Python script inside a sandbox/VM only.
3. Learned classifier and learned risk weights (logistic regression on calibration data) compared with the hand-set rules.
4. Per-user or per-container lineage budgets (cgroup-like).

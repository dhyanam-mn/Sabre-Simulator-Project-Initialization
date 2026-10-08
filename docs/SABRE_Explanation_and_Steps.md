# SABRE Project: Explanation and Step-by-Step Guide

This is the plain-language companion to `SABRE_Implementation_Workdoc.md`. Read this first to understand *what* you are building and *what to do each week*. Then use the work document (and its AI prompts) to actually write the code.

---

## Part 1: What are we building? (in simple words)

**The problem.** A normal CPU scheduler (FCFS, SJF, Round Robin, Priority) decides who gets the CPU next using only arrival order, job length or a fixed priority. It does not notice if a process is *misbehaving*: a fork bomb creating thousands of children, a miner using 100% CPU, or a sneaky process that stays just under the limit.

**The idea.** Build a scheduler that watches each process and uses three signals:
1. **Predicted next burst:** how long will it probably run next?
2. **Behaviour class:** is it interactive, I/O-bound, CPU-bound or bursty?
3. **Risk score:** how suspicious is it *compared with what is normal for its class*?

These three signals decide the process's **queue, priority and time quantum**. A suspicious process is slowed down gradually, but **never starved**, and it can recover if it behaves again.

**What we actually deliver.** Not an operating-system kernel. We build a **simulator** (a Python program that imitates a CPU and its processes), run our scheduler and the four classic ones on the same fake workloads, inject attacks, and measure who does better. The result is a set of experiments and graphs, plus a paper.

**The picture in your head:**

```
Fake workload (normal + attack processes)
        |
   Simulator (pretend CPU, clock, processes)
        |
   Scheduler  <--  asks: prediction? class? risk?
        |
   Metrics (waiting time, response time, detection, overhead)
        |
   Graphs + paper
```

---

## Part 2: Key terms you must be able to explain in your review

| Term | Meaning |
|---|---|
| Burst | A stretch of time a process uses the CPU before it waits for I/O or is preempted |
| Quantum | The maximum time a process may run before being preempted |
| Exponential averaging | Prediction rule: new guess = α × last burst + (1−α) × old guess |
| Risk score | Number from 0 to 1 for how suspicious a process looks |
| CUSUM | A method that adds up small, repeated increases so slow creeping is noticed |
| Lineage | The parent-child family tree of processes; children inherit parent risk |
| Containment ladder | Levels 0 to 3: normal, watched, limited, contained (each reversible) |
| Hysteresis | Needs several readings before changing level, so it does not flip back and forth |
| Work-conserving | The CPU never sits idle if someone is waiting; limits apply only when others need the CPU |
| LSI | Legitimate Slowdown Index: damage done to innocent processes (lower is better) |
| ADI | Attack Damage Index: how much an attack hurts interactive users (lower is better) |
| Ablation | Switching one feature off to see how much it contributed |

---

## Part 3: What each person does

| Person | Role | Owns (folder) | In one sentence |
|---|---|---|---|
| **Khushi** | A: Simulator | `core/`, `schedulers/baselines.py` | Builds the pretend computer, the workloads, the attacks and the classic schedulers |
| **Aadya** | B: Intelligence | `intel/` | Builds the "brain": features, burst predictor, classifier, risk scorer |
| **Dhyanam** | C: Scheduler + experiments | `schedulers/sabre.py`, `controller.py`, `experiments/` | Builds our scheduler, runs all experiments, makes graphs and statistics |

All three agree first on the **interfaces** (Section 3 of the work document). That agreement is what lets you work in parallel.

---

## Part 4: Step-by-step plan

### Step 0: Setup (everyone, day 1 to 2)
- [ ] Create one shared GitHub repo with the folder layout from the work document.
- [ ] Install Python 3.10+ and `numpy pandas scipy matplotlib pytest pyyaml`.
- [ ] Copy `configs/default.yaml` from the work document into the repo.
- [ ] Each person reads Sections 0, 1, 3 and their own module section.
- [ ] Agree on the interface file contents (`Process`, `Scheduler`, `Features`...) and commit them before anyone codes. Do not change them later without telling the others.
- [ ] Set up a branch per person; merge to `main` only when `pytest -q` passes.

### Step 1 (Week 1): Foundations
- **Khushi:** `Process` class and the simulation engine; FCFS and Round Robin working on a hand-made 3-process example. Check the waiting times by hand.
- **Aadya:** `FeatureExtractor` (burst history, I/O ratio, CPU share) and the exponential-averaging predictor, tested on made-up numbers.
- **Dhyanam:** config loader, test setup, empty metrics file, experiment runner skeleton.
- **Checkpoint:** engine runs FCFS and RR; predictor passes its tests.

### Step 2 (Week 2): Normal behaviour
- **Khushi:** benign workload generator (4 process kinds), then SJF, Priority and MLFQ.
- **Aadya:** rule-based behaviour classifier; measure its accuracy against the true kind.
- **Dhyanam:** first version of the SABRE controller using only class and predicted burst (no risk yet).
- **Checkpoint:** SABRE-without-risk runs on a benign workload and is close to Round Robin.

### Step 3 (Week 3): Attacks and risk
- **Khushi:** the three attacks: fork-bomb, CPU hog, low-and-slow (with a drift variant).
- **Aadya:** first RiskScorer (rate z-scores plus decay), and the calibration script that learns "normal" per class from benign-only runs.
- **Dhyanam:** plug Aadya's modules into the controller; first full end-to-end run.
- **Checkpoint (integration 1):** one run with attackers produces a metrics row and a flagged-process list.

### Step 4 (Week 4): The novel parts
- **Khushi:** adaptive attacker and the RR + static-threshold baseline.
- **Aadya:** add CUSUM drift detection, lineage inheritance and the descendant budget.
- **Dhyanam:** containment ladder with hysteresis, elastic share caps, guaranteed minimum share.
- **Checkpoint:** attacker is slowed but never fully starved; lone contained process still gets 100% CPU.

### Step 5 (Week 5): Metrics and experiment machinery
- **Khushi:** all metrics (waiting, turnaround, response percentiles, fairness, precision/recall/FPR, time-to-detect, LSI, ADI).
- **Aadya:** optional online ML predictor; prediction-error table.
- **Dhyanam:** experiment runner (30 seeds), confidence intervals, paired statistical tests.
- **Checkpoint (integration 2):** `run_experiment --exp E1` produces a full CSV.

### Step 6 (Week 6): Run the experiments
- **Aadya:** tune thresholds and weights on **separate tuning seeds** (never the test seeds).
- **Dhyanam:** run E1 to E9 (comparison, intensity, stealth, malicious fraction, false-positive stress, adaptive attacker, ablation, overhead, Pareto).
- **Khushi:** optional trace-driven workload (E10); help debug odd results.

### Step 7 (Week 7): Graphs and cleanup
- **Dhyanam:** all plots, including the timeline figure showing an attacker being contained while a benign CPU-bound process is not.
- **Aadya:** sensitivity analysis of the main parameters.
- **Khushi:** performance profiling, bug fixes, README, code freeze.

### Step 8 (Week 8): Write-up
- [ ] Fill the paper's Methodology and Results sections from the work document and `results/`.
- [ ] Add the acceptance checklist results (below) as a table.
- [ ] Prepare a 5-minute demo: run one scenario, show the timeline plot and the decision log.
- [ ] Rehearse: each person explains their own module.

---

## Part 5: Checks before you trust any result

1. Same seed gives exactly the same numbers when re-run.
2. No attack: SABRE's average waiting time is within ±10% of Round Robin.
3. All SABRE features switched off behaves like Round Robin.
4. A contained process never drops below the minimum CPU share while others are busy.
5. A benign CPU-heavy process is **not** flagged (false-positive test, E5).
6. All schedulers saw identical workloads on each seed.
7. Monitoring overhead is charged to the clock (not free).
8. Calibration and tuning seeds are different from evaluation seeds.

---

## Part 6: How to work with an AI assistant efficiently

1. Paste the **Master Prompt** (work document Section 0) at the start of every new chat.
2. Paste **only one module** at a time.
3. Ask for code **and tests together**; run `pytest -q` yourself before accepting.
4. If it breaks, paste the **exact error** back and say "fix without changing the interface".
5. Commit after every working module.
6. Do not let the AI rename classes or fields; that breaks teammates' code.
7. Understand what you are submitting: be ready to explain every formula (risk score, CUSUM, quantum rule) at the review.

---

## Part 7: Questions the reviewers may ask (prepare answers)

- *Why a simulator and not a real kernel?* Controlled workloads, exact ground truth, safe to run attacks, reproducible.
- *How do you avoid punishing legitimate CPU-heavy programs?* Risk is judged against the process's own behaviour class, and containment is graduated and reversible.
- *What stops starvation?* Guaranteed minimum share and aging.
- *What is the cost of your idea?* Monitoring overhead, which is measured and charged, and we find the break-even point (E8).
- *What if an attacker adapts?* The adaptive attacker experiment (E6) tests that directly.
- *What is new compared with existing work?* Class-conditional risk, CUSUM for low-and-slow, lineage-aware risk, the reversible containment ladder, and the LSI/ADI metrics.
- *Does it always win?* No, and we say so: the contribution is understanding the security-versus-performance trade-off.

---

## Part 8: One-page summary checklist

- [ ] Repo + interfaces agreed
- [ ] Engine + FCFS/RR (Wk 1)
- [ ] Workload + all baselines + classifier (Wk 2)
- [ ] Attacks + risk v0 + first end-to-end run (Wk 3)
- [ ] CUSUM + lineage + containment ladder (Wk 4)
- [ ] All metrics + experiment runner (Wk 5)
- [ ] Experiments E1 to E9 run on 30 seeds (Wk 6)
- [ ] Graphs, sensitivity, code freeze (Wk 7)
- [ ] Paper results + demo + rehearsal (Wk 8)

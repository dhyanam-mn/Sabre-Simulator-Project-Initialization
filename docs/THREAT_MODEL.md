# Threat Model: SABRE CPU Scheduling Architecture

## 1. System Model & Operating Assumptions
SABRE operates as an OS kernel CPU scheduler on a single or multicore shared system running mixed, multi-tenant workloads. The system executes both benign client applications (interactive user interfaces, background I/O workers, numerical compute pipelines, and bursty servers) and untrusted, potentially malicious workloads.

- **System Environment:** Monolithic or hybrid OS kernel managing discrete time-sliced CPU allocation.
- **Scheduler Visibility:** The scheduler observes execution telemetry strictly through runtime scheduling events: slice durations, CPU run-time over rolling windows, burst arrival/completion timestamps, context switch counts, fork/spawn events, and process hierarchy (PID/PPID).
- **Execution Cost Model:** Security telemetry extraction, Bayesian classification, CUSUM drift updates, and containment decisions accrue non-zero CPU execution overhead charged directly to system accounting.

---

## 2. Attacker Goals
The adversary seeks to compromise system utility through one or more of the following objectives:
1. **Denial of Service (Interactive Starvation):** Induce severe latency degradation and response spikes on benign interactive processes by hoarding CPU cycles, exhausting scheduler runqueues, or flooding the process tree.
2. **Resource Theft (Unauthorized Compute / Cryptomining):** Sustain near-100% CPU utilization over prolonged intervals without authorization, depriving competing batch or computational tasks of fair execution share.
3. **Stealth & Evasion:** Evade rate monitors, static thresholds, and anomaly detectors by duty-cycling computation, mimicking legitimate bursty/I/O profiles, or adapting execution behavior dynamically based on perceived throttling.

---

## 3. Attacker Knowledge & Capabilities

### Attacker Knowledge
- **Black-Box Access:** The attacker possesses no direct read access to kernel internal risk states, threshold vectors, rolling CUSUM sums, or peer process metadata.
- **Latency/Throughput Oracle:** The attacker can measure its own wall-clock progress, scheduling latency, and wait-to-run ratios over historical windows (e.g., measuring cycle delay between yield/sleep intervals).

### Attacker Capabilities
- **Unprivileged Execution:** All attacks execute strictly as unprivileged user-space processes without root/CAP_SYS_NICE capabilities.
- **Process Hierarchy Flooding:** The attacker can invoke `fork()` up to system descendant budgets to create daughter processes sharing or multiplying compute demand.
- **Duty-Cycle Modulation:** The attacker can alternate between intensive compute bursts and voluntary sleep/I/O intervals (`nanosleep`, `select`, file/socket I/O) to target specific average CPU shares.
- **Behavioral Mimicry:** The attacker can structure compute duration and sleep periods to mimic benign statistical distributions (e.g., imitating interactive or bursty workloads).
- **Adaptive Feedback:** The attacker can modify its duty cycle dynamically upon detecting scheduling delays (cutting compute demand when throttled, escalating demand when unconstrained).

---

## 4. Security Boundaries & Out-of-Scope Threats

### Out of Scope (Explicit Non-Goals)
To maintain defensive focus on CPU scheduling integrity, the following vectors are explicitly treated as out of scope:
1. **Memory & Cache Exhaustion:** RAM depletion (OOM-killer exploitation), swap thrashing, cache-side-channel attacks (Spectre/Meltdown), and memory bus locking.
2. **I/O & Network Saturation:** Exhausting disk bandwidth, filesystem inode exhaustion, network socket exhaustion, or SYN flooding.
3. **Kernel Privilege Escalation:** Local privilege escalation exploits, kernel memory corruption, or direct manipulation of scheduler memory data structures.
4. **Hardware/Firmware Compromises:** Microarchitectural side-channels, power analysis, or hardware fault injection.

---

## 5. Defender Goals & Guarantees
Under concurrent execution of benign and adversarial workloads, SABRE enforces the following invariants:
1. **Starvation-Freedom:** Every runnable process (even under maximum containment level 3) receives a guaranteed non-zero CPU slice within each scheduler cycle $T_{\text{slot}}$.
2. **Bounded Interactive Latency (ADI):** Interactive degradation under attack is strictly bounded; adversarial processes cannot indefinitely monopolize the execution quantum.
3. **Low Collateral Damage (FPR Control):** Benign processes (particularly legitimate CPU-bound computational batch jobs) are not falsely elevated into punitive containment tiers ($\text{FPR} \le 5\%$).
4. **Work-Conservation:** The CPU is never allowed to remain idle when runnable processes exist, ensuring optimal aggregate system throughput.
5. **Reversibility & Elastic Containment:** Containment tiers are elastic and reversible; processes that cease anomalous behavior are eligible for step-wise rehabilitation via hysteresis margin recovery.

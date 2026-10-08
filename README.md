# SABRE: Security- And Behaviour-Aware, Risk-Elastic CPU Scheduler

Deterministic discrete-event CPU-scheduling simulator evaluating adaptive, class-conditional risk scoring and elastic resource containment against misbehaving and adversarial workloads.

## Project Structure
- `sabre/core`: Discrete-event engine, process representation, workloads, attack models, and metrics.
- `sabre/schedulers`: Standard scheduling baselines (FCFS, SJF, RR, Priority, MLFQ, RR+StaticThreshold) and the SABRE adaptive scheduler.
- `sabre/intel`: Feature extraction, burst prediction, behavior classification, and risk scoring.
- `sabre/experiments`: Experiment execution runners, statistics evaluation, and plotting.
- `sabre/dashboard`: Interactive analysis dashboard.
- `configs/`: Simulator settings and experimental seed configurations.

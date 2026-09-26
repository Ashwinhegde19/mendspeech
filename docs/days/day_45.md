# Day 45: Load test to saturation and failure recovery

> **Week 7 • Day 3 of 7**
> **Navigation:** [← Day 44](day_44.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 46 →](day_46.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 44](day_44.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Load methodology, saturation/queue growth, and throughput-at-saturation versus user experience.

---

### 2. Build in MendSpeech
- Build a load harness in src/serve/loadtest.py (configurable concurrency, fixed input, bounded budgets).
- Reproduce overload, disconnect and recovery scenarios.

---

### 3. Experiment and Measure
- Sweep concurrency within the authorized limit; report the knee, offered vs achieved rates, per-stream p50/p95/p99, queue wait and rejected/failed work.
- Show that rejection does not masquerade as capacity; document recovery in docs/day45_failure_recovery.md.

---

### 4. Required Output Artifacts
- `src/serve/loadtest.py`
- `results/day45_load_curve.csv`
- `docs/day45_failure_recovery.md`
- `reports/day45_serving.md`

---

### 5. Completion Check
> **Definition of Done for Day 45:**
> A load curve to saturation, a named concurrency knee, and a reproduced failure-and-recovery case within budget.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

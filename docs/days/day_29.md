# Day 29: Batching, concurrency and queueing

> **Week 5 • Day 1 of 7**
> **Navigation:** [← Day 28](day_28.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 30 →](day_30.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 27](day_27.md)
> **Effort:** 2–4 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Static versus dynamic batching, queue wait versus service time, batch-size latency/throughput knee.

---

### 2. Build in MendSpeech
- Add bounded batching and per-stream queues in src/serve/batching.py; keep the co-residency pilot constraint in force.

---

### 3. Experiment and Measure
- Sweep batch size/concurrency within budget; report throughput, per-stream p50/p95/p99, queue wait, achieved concurrency and memory.
- Verify no cross-stream state contamination and report starvation/fairness; batching is not required to be selected.

---

### 4. Required Output Artifacts
- `src/serve/batching.py`
- `tests/test_batching.py`
- `results/day29_batch_sweep.csv`
- `docs/day29_queueing.md`

---

### 5. Completion Check
> **Definition of Done for Day 29:**
> Throughput and single-stream latency are reported separately with the knee and fairness behavior.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

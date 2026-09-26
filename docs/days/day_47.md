# Day 47: Per-stage correlated latency budget

> **Week 7 • Day 5 of 7**
> **Navigation:** [← Day 46](day_46.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 48 →](day_48.md)

> **v4 STATUS: CORE** Release headline artifact.
> **Prerequisites:** [Day 44](day_44.md), [Day 45](day_45.md), [Day 46](day_46.md)
> **Effort:** 2–4 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Critical-path latency attribution; tail ownership by request ID, not percentile sum.

---

### 2. Build in MendSpeech
- Instrument the full path in src/bench/budget.py with trace IDs, clock sync notes, stage start/end, queue/prefill/decode/generation/guard/network.
- Produce correlated per-request critical paths for the shipped configuration.

---

### 3. Experiment and Measure
- Report per-stage p50/p95/p99, counts, cold/warm and length slices.
- Identify the p99 owner by inspecting the same slow requests; do not sum percentiles or claim a sub-500ms end-to-end guarantee.

---

### 4. Required Output Artifacts
- `src/bench/budget.py`
- `results/day47_latency_budget.csv`
- `results/day47_latency_budget.png`
- `docs/day47_latency_budget.md`

---

### 5. Completion Check
> **Definition of Done for Day 47:**
> A reproducible, request-level decomposition that names the tail owner and the largest optimization target, with no unsupported end-to-end claim.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

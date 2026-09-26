# Day 27: Profile the current end-to-end path

> **Week 4 • Day 6 of 7**
> **Navigation:** [← Day 26](day_26.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 28 →](day_28.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 14](day_14.md), [Day 26](day_26.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Kernel versus wall time, CPU launch overhead, queue/prefill/decode/generation and cold-versus-warm.

---

### 2. Build in MendSpeech
- Add per-stage instrumentation to src/bench/profile_ops.py on the Day26 pipeline; no optimization technique is applied yet.
- Separate start/compile/load from steady state; report per-request critical paths, not just aggregated percentiles.

---

### 3. Experiment and Measure
- Rank measured time contributors and note overlapping/serialized stages. Identify where p99 requests diverge from median, using the same request IDs.
- State expected ceiling for each candidate; unsupported profiling granularity is recorded, not guessed.

---

### 4. Required Output Artifacts
- `src/bench/profile_ops.py`
- `results/day27_profile.csv`
- `docs/day27_optimization_targets.md`

---

### 5. Completion Check
> **Definition of Done for Day 27:**
> The top measured contributors and the p99 divergence are named with request-level evidence, not assumption.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

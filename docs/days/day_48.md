# Day 48: End-to-end optimization round informed by the budget

> **Week 7 • Day 6 of 7**
> **Navigation:** [← Day 47](day_47.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 49 →](day_49.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 47](day_47.md)
> **Effort:** 2–4 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Choosing one change from measured evidence and separating it from drift.

---

### 2. Build in MendSpeech
- Apply the change the Day47 budget identifies as the largest target in src/.
- Re-run the full Day47 decomposition and the Day32 scorecard after the change.

---

### 3. Experiment and Measure
- Report before/after p50/p95/p99 and quality with enough repetitions to separate a real gain from noise.
- A change that does not help, or that breaks a guard, is recorded as a negative result.

---

### 4. Required Output Artifacts
- `results/day48_e2e_optimization.csv`
- `docs/day48_optimization_outcome.md`
- `app/audio_lab.py`

---

### 5. Completion Check
> **Definition of Done for Day 48:**
> A measured end-to-end before/after tied to the budget, or a documented negative outcome with evidence.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

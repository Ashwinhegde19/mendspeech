# Day 53: Optimization, serving and editor ablations on the frozen harness

> **Week 8 • Day 4 of 7**
> **Navigation:** [← Day 52](day_52.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 54 →](day_54.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 32](day_32.md), [Day 45](day_45.md), [Day 46](day_46.md), [Day 48](day_48.md), [Day 52](day_52.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Pareto frontiers; holding inputs fixed so comparisons mean something.

---

### 2. Build in MendSpeech
- Run every optimization variant, serving configuration and editor variant on the identical frozen subset in src/bench/run_ablations.py.
- Keep live measurements separate from simulated estimates.

---

### 3. Experiment and Measure
- Plot WER vs p99 latency and mark Pareto-efficient points.
- Report editor variants by quality/latency and state the shipped configuration.

---

### 4. Required Output Artifacts
- `src/bench/run_ablations.py`
- `results/day53_ablations.csv`
- `results/day53_pareto.png`

---

### 5. Completion Check
> **Definition of Done for Day 53:**
> A measured ablation set with Pareto frontiers and a stated shipping configuration.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

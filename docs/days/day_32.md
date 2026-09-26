# Day 32: Optimization scorecard and selection

> **Week 5 • Day 4 of 7**
> **Navigation:** [← Day 31](day_31.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 33 →](day_33.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 26](day_26.md), [Day 28](day_28.md), [Day 29](day_29.md), [Day 30](day_30.md), [Day 31](day_31.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Multi-objective selection, Pareto frontiers and honest negative reporting.

---

### 2. Build in MendSpeech
- Build the scorecard generator in src/bench/scorecard.py over all measured variants.
- Record a what_did_not_help section; select one shipping candidate and mark the selection provisional pending final-stack revalidation.

---

### 3. Experiment and Measure
- Tabulate WER, latency percentiles, RTF, memory and calibration status per variant.
- Select on measured grounds and identify the largest remaining bottleneck for the end-to-end path.

---

### 4. Required Output Artifacts
- `src/bench/scorecard.py`
- `results/day32_optimization_scorecard.csv`
- `docs/day32_optimization_report.md`

---

### 5. Completion Check
> **Definition of Done for Day 32:**
> A defensible provisional shipping configuration with Pareto evidence including the techniques that failed.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

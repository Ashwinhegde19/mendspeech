# Day 52: Robustness matrix on the frozen set

> **Week 8 • Day 3 of 7**
> **Navigation:** [← Day 51](day_51.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 53 →](day_53.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 38](day_38.md), [Day 39](day_39.md), [Day 51](day_51.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Sliced evaluation, multiple comparisons and variance.

---

### 2. Build in MendSpeech
- Run the full corruption x severity x decoder grid in src/bench/run_matrix.py on the frozen set.
- Report the robustness-adapted checkpoint and the base checkpoint in the same matrix.

---

### 3. Experiment and Measure
- Report WER/CER, names/numbers and confidence behaviour per cell.
- Estimate variance on a representative subset; identify the worst cell.

---

### 4. Required Output Artifacts
- `src/bench/run_matrix.py`
- `results/day52_robustness_matrix.csv`
- `results/day52_robustness_matrix.png`

---

### 5. Completion Check
> **Definition of Done for Day 52:**
> The complete robustness matrix is measured on the frozen set, with the worst cell and variance identified.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

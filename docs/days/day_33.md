# Day 33: Rebuild and revalidate the end-to-end pipeline

> **Week 5 • Day 5 of 7**
> **Navigation:** [← Day 32](day_32.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 34 →](day_34.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 25](day_25.md), [Day 26](day_26.md), [Day 32](day_32.md)
> **Effort:** 2–4 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Provisional optimization results must be re-checked after pipeline integration.

---

### 2. Build in MendSpeech
- Rebuild src/streaming, src/serve and app/audio_lab.py around the selected candidate with the Day25 calibration and guard contract intact.
- Re-verify streaming parity, confidence binding and the joint memory/interference pilot on the selected stack.

---

### 3. Experiment and Measure
- Re-measure Day26 baseline quality/latency/memory end-to-end; deltas vs provisional are explained.
- If the candidate is infeasible jointly, record the blocker and scope-review options instead of forcing the stack.

---

### 4. Required Output Artifacts
- `results/day33_rebuild_revalidation.csv`
- `docs/day33_rebuild_notes.md`

---

### 5. Completion Check
> **Definition of Done for Day 33:**
> The optimized end-to-end pipeline is revalidated, and any change in behavior versus the provisional baseline is explained.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

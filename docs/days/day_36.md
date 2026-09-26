# Day 36: Training pipeline anatomy for the editor

> **Week 6 • Day 1 of 7**
> **Navigation:** [← Day 35](day_35.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 37 →](day_37.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 35](day_35.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Loss curves, overfitting detection, tokenizer/label masking and checkpoint reproducibility.

---

### 2. Build in MendSpeech
- Diagnose the Day35 SFT loss/eval curves and add reproducibility checks in training/editor_sft.py.
- Document data mix, label masking and checkpoint reload determinism.

---

### 3. Experiment and Measure
- Verify checkpoint reload reproduces validation numbers.
- Relate loss/overfit behaviour to editor data size so RL budgets are set on evidence.

---

### 4. Required Output Artifacts
- `docs/day36_editor_training_diagnosis.md`
- `results/day36_editor_loss_curves.csv`

---

### 5. Completion Check
> **Definition of Done for Day 36:**
> Editor training behaviour is diagnosable and reproducible, giving evidence-based budgets for the RL run.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

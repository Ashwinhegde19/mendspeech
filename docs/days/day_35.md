# Day 35: Editor SFT and continued-SFT control

> **Week 5 • Day 7 of 7**
> **Navigation:** [← Day 34](day_34.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 36 →](day_36.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 16](day_16.md), [Day 34](day_34.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Supervised fine-tuning for constrained text editing and the compute-matched continued-SFT control.

---

### 2. Build in MendSpeech
- Run editor SFT via training/editor_sft.py on the Day13 train split; freeze the SFT checkpoint as the RL reference.
- Run a continued-SFT control matched to the future RL wall-time/token budget, so extra compute is not mistaken for the RL algorithm.

---

### 3. Experiment and Measure
- Evaluate SFT and continued-SFT on validation (protected-content violations, formatting accuracy, identity vs needs-edit, risk-coverage).
- Log trainable-parameter names/counts, memory, step time and cost; artifacts for both arms.

---

### 4. Required Output Artifacts
- `training/editor_sft.py`
- `configs/editor_sft.yaml`
- `results/day35_sft_vs_continued.csv`
- `docs/day35_sft_notes.md`

---

### 5. Completion Check
> **Definition of Done for Day 35:**
> An SFT editor and a compute-matched continued-SFT control exist so any later RL gain cannot be explained by extra training alone.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

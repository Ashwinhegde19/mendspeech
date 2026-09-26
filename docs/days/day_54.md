# Day 54: Adaptation and RL final comparison

> **Week 8 • Day 5 of 7**
> **Navigation:** [← Day 53](day_53.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 55 →](day_55.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 38](day_38.md), [Day 40](day_40.md), [Day 41](day_41.md), [Day 53](day_53.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Separating acoustic robustness adaptation from text-editor post-training.

---

### 2. Build in MendSpeech
- Run base, robustness-adapted, SFT-editor and RL-editor conditions through the frozen harness in src/bench/run_personalization.py.
- Report each condition on its own axis: ASR WER/robustness for the checkpoint; editor quality for the editor.

---

### 3. Experiment and Measure
- Report the frozen-test numbers for all conditions, including clean-speech regression.
- State whether adaptation and RL each earned their place; a null result is reported, not hidden.

---

### 4. Required Output Artifacts
- `src/bench/run_personalization.py`
- `results/day54_conditions_final.csv`
- `reports/day54_conditions_final.md`

---

### 5. Completion Check
> **Definition of Done for Day 54:**
> The final conditions are compared on frozen evidence, with adaptation and editor quality kept distinct and nulls reported.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

# Day 10: WER/CER, data roles, and evaluation protocol

> **Week 2 • Day 3 of 7**
> **Navigation:** [← Day 09](day_09.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 11 →](day_11.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 09](day_09.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Local CPU`

---

### 1. Learn
- Substitution/deletion/insertion counts, normalization and empty-reference conventions.
- Group-level leakage, repeated corruptions versus independent samples.

---

### 2. Build in MendSpeech
- Implement/test scoring and named-entity/number/negation slices without adding a model-based judge.
- Audit frozen benchmark roles and source/speaker separation. Record immutable checksums and separate training, calibration, validation and final-test roles before downstream tuning.
- Declare diagnostic primary metrics and failure thresholds in experiments/protocol.md; new editor data is a separate manifest, not a changed core set.

---

### 3. Experiment and Measure
- Hand-check small examples and empty/identical inputs; run validation diagnostics by damage/severity.
- Report actual corpus counts/roles; insufficient training or calibration data blocks training, not permission to reuse test.

---

### 4. Required Output Artifacts
- `src/metrics/wer.py`
- `tests/test_wer.py`
- `results/day10_wer_by_damage.csv`
- `results/day10_error_types.csv`
- `reports/data_roles.md`
- `experiments/protocol.md`

---

### 5. Completion Check
> **Definition of Done for Day 10:**
> Scoring conventions and frozen data roles are explicit and tested; development uses validation, not repeated test selection.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

# Day 11: Confidence definitions and failure cases

> **Week 2 • Day 4 of 7**
> **Navigation:** [← Day 10](day_10.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 12 →](day_12.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 10](day_10.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Local CPU`

---

### 1. Learn
- Acoustic frame score, emitted-token confidence and word correctness are different quantities.
- Blank-heavy averages can hide errors; confidence is not calibrated merely because it lies in [0,1].

---

### 2. Build in MendSpeech
- Build token/word score records with model/head/tokenizer provenance; inspect the current baseline averaging semantics rather than trusting its docstring.
- Define label alignment and valid/missing states, with empty/blank/repeat tests.

---

### 3. Experiment and Measure
- On validation clips compare score versus correctness by corruption. Collect confident errors when present; report the observed count, never invent a required ten.
- Keep raw softmax and later calibrated probability distinct.

---

### 4. Required Output Artifacts
- `src/asr/confidence.py`
- `tests/test_confidence.py`
- `results/day11_confidence_by_damage.csv`
- `docs/day11_confident_wrong.md`

---

### 5. Completion Check
> **Definition of Done for Day 11:**
> Every score has a tested definition and correctness target; no raw confidence is called calibrated.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

# Day 13: Conservative editor contract and evaluation data

> **Week 2 • Day 6 of 7**
> **Navigation:** [← Day 12](day_12.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 14 →](day_14.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 10](day_10.md), [Day 11](day_11.md), [Day 12](day_12.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Local CPU`

---

### 1. Learn
- Formatting versus rewriting and protected content.
- Training/validation/test group splits and proposed versus delivered output.

---

### 2. Build in MendSpeech
- Implement/test identity and deterministic whitespace/casing/punctuation baseline under docs/EDITOR_AND_RL_CONTRACT.md. Define protected spans and fail-closed output guard.
- Create data/editor_manifest.jsonl with licensed/consented source groups, annotated allowed outputs and sealed held-out roles. Audit duplicates and instruction-like input.
- Declare pilot/full-set counts and human review protocol; exact wording/negation/numbers cannot be silently changed.

---

### 3. Experiment and Measure
- Score identity and deterministic formatting on validation, including already-correct and needs-edit slices.
- Exercise deleted negation, altered entities/numbers, blank/long output, prompt injection and punctuation-induced meaning changes; log manual reviewer limits.

---

### 4. Required Output Artifacts
- `src/llm/contracts.py`
- `src/llm/deterministic.py`
- `tests/test_editor_contract.py`
- `data/editor_manifest.jsonl`
- `reports/editor_data_audit.md`
- `results/day13_editor_baselines.csv`

---

### 5. Completion Check
> **Definition of Done for Day 13:**
> Editing rules and independent quality metrics exist before LLM training; the small pilot and full evaluation data are separated and incomplete annotation is reported.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

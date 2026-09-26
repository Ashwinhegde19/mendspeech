# Day 24: Fixed-context quality and latency frontier

> **Week 4 • Day 3 of 7**
> **Navigation:** [← Day 23](day_23.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 25 →](day_25.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 21](day_21.md), [Day 22](day_22.md), [Day 23](day_23.md)
> **Effort:** 2–4 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Algorithmic lookahead, wall-clock finalization and WER versus responsiveness.

---

### 2. Build in MendSpeech
- Add only documented fixed context configurations and preserve model/head/cache compatibility.
- Use the shared replay/harness with trace IDs, warm/cold and settings recorded.

---

### 3. Experiment and Measure
- Compare supported context settings on validation at fixed decoder and batch; report time-to-first-partial and post-utterance finalization plus WER/CER.
- Fewer than two supported settings yields a single-setting baseline and explicit limit, not invented adaptive gains.

---

### 4. Required Output Artifacts
- `src/streaming/context.py`
- `tests/test_context.py`
- `results/day24_context_tradeoff.csv`

---

### 5. Completion Check
> **Definition of Done for Day 24:**
> Context costs are measured with a working chunk loop and endpoint detector; scope of supported settings is honest.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

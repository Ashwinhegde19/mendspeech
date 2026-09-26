# Day 43: Serving contract and WebSocket message schema

> **Week 7 • Day 1 of 7**
> **Navigation:** [← Day 42](day_42.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 44 →](day_44.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 26](day_26.md), [Day 33](day_33.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- WebSocket message schemas, per-stream isolation, cancellation and backpressure semantics.

---

### 2. Build in MendSpeech
- Define schema in src/serve/schema.py: audio chunks in; partial/final transcripts, confidence, stage events and latency fields out.
- Specify timeout, disconnect, cancellation and bounded-queue semantics; write contract tests in tests/test_serve_schema.py.

---

### 3. Experiment and Measure
- Verify the schema round-trips a recorded session.
- Ensure latency fields match the contract definitions and never expose unmeasured values.

---

### 4. Required Output Artifacts
- `src/serve/schema.py`
- `tests/test_serve_schema.py`
- `docs/day43_serving_contract.md`

---

### 5. Completion Check
> **Definition of Done for Day 43:**
> A testable WebSocket contract with explicit latency, confidence, failure and backpressure semantics.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

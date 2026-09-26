# Day 44: Async streaming service

> **Week 7 • Day 2 of 7**
> **Navigation:** [← Day 43](day_43.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 45 →](day_45.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 33](day_33.md), [Day 43](day_43.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- FastAPI/async WebSocket handling, per-stream state isolation and clean cancellation.

---

### 2. Build in MendSpeech
- Implement the service in src/serve/app.py around the Day33 candidate, pinned one L4, serialized queue and bounded state.
- Containerize reproducibly under infra/serve/.

---

### 3. Experiment and Measure
- Verify concurrent streams do not share or corrupt cache/session state.
- Confirm a mid-utterance disconnect leaves no orphaned GPU work; report cold start separately from warm latency.

---

### 4. Required Output Artifacts
- `src/serve/app.py`
- `tests/test_serve_isolation.py`
- `infra/serve/Dockerfile`
- `infra/serve/README.md`

---

### 5. Completion Check
> **Definition of Done for Day 44:**
> Concurrent streams are isolated, disconnects are clean, and cold start is measured separately from warm latency.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

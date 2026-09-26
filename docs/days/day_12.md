# Day 12: Transcript timestamps and correlated event tracing

> **Week 2 • Day 5 of 7**
> **Navigation:** [← Day 11](day_11.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 13 →](day_13.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 10](day_10.md), [Day 11](day_11.md)
> **Effort:** 2–4 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Audio frame offsets versus wall-clock events; forced-alignment limitations.
- Monotonic clocks, request/stream IDs and event ordering.

---

### 2. Build in MendSpeech
- Implement transcript timestamp records plus a shared trace schema; separate word alignment from service-clock events.
- Test synthetic frame-index mapping, real short manually aligned speech, empty output and resampling; a click/tone is not a word-timestamp ground truth.

---

### 3. Experiment and Measure
- Report ms error on annotated validation speech and corruption cases with uncertainty; do not demand unchanged word timing when the recognized words differ.
- Round-trip trace IDs and start/end ordering for a replayed request.

---

### 4. Required Output Artifacts
- `src/asr/timestamps.py`
- `tests/test_timestamps.py`
- `src/bench/tracing.py`
- `tests/test_tracing.py`
- `results/day12_timestamp_error.csv`

---

### 5. Completion Check
> **Definition of Done for Day 12:**
> Word timing is validated against relevant annotations and trace clocks are explicit; the two are not conflated.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

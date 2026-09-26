# Day 21: Correct chunk loop and per-stream state

> **Week 3 • Day 7 of 7**
> **Navigation:** [← Day 20](day_20.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 22 →](day_22.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 18](day_18.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Chunk clocks, partial/final decoding, cache ownership and last-chunk flush.

---

### 2. Build in MendSpeech
- Implement src/streaming/session.py with start/push/finish/reset, bounded state, variable final chunks and sequence validation.
- Add deterministic audio replay preserving input cadence; test isolated/interleaved streams and flush/reset exactly once.

---

### 3. Experiment and Measure
- Verify repeated single-stream replay agrees with isolated interleaved streams under identical model context. Offline full-context text need not match streaming text; compare the documented same-context reference.
- Log first partial, final timestamps and state sizes. Future-context dependence cannot be called causal.

---

### 4. Required Output Artifacts
- `src/streaming/session.py`
- `tests/test_streaming_session.py`
- `experiments/replay_audio.py`
- `results/day21_streaming_correctness.csv`

---

### 5. Completion Check
> **Definition of Done for Day 21:**
> An explicit tested chunk/session loop exists before profiling, with valid finals and no cross-stream state leakage.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

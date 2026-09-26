# Day 31: Streaming fast path and cache-failure evidence

> **Week 5 • Day 3 of 7**
> **Navigation:** [← Day 30](day_30.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 32 →](day_32.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 21](day_21.md), [Day 24](day_24.md), [Day 28](day_28.md), [Day 30](day_30.md)
> **Effort:** 2–4 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- State reuse versus recomputation across chunks and reset/truncation failure modes.

---

### 2. Build in MendSpeech
- Add a streaming fast path in src/streaming/fast_path.py reusing the best supported variant, with state equivalence tests.
- Break the cache deliberately at chosen boundaries via src/streaming/cache_stress.py to characterize failure.

---

### 3. Experiment and Measure
- Report steady-state per-chunk latency separate from first chunk; verify cached vs uncached transcripts.
- Record WER changes and whether errors cluster or propagate at reset points; this failure evidence is required release material.

---

### 4. Required Output Artifacts
- `src/streaming/fast_path.py`
- `tests/test_fast_path_parity.py`
- `src/streaming/cache_stress.py`
- `results/day31_cache_failures.md`
- `results/day31_fast_path.csv`

---

### 5. Completion Check
> **Definition of Done for Day 31:**
> A measured streaming fast path with parity and a concrete, reproducible cache-state failure.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

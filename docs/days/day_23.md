# Day 23: VAD, endpointing and finalization state machine

> **Week 4 • Day 2 of 7**
> **Navigation:** [← Day 22](day_22.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 24 →](day_24.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 21](day_21.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Frame energy/spectral VAD, hangover, pause/endpoint trade-offs and speech-end labels.

---

### 2. Build in MendSpeech
- Implement deterministic framing/VAD with one compatible local reference detector; tune thresholds only on validation.
- Integrate start/end/hangover state with streaming/session.py. Preserve too-quiet/short/silence failure cases; no diarization.

---

### 3. Experiment and Measure
- Measure precision/recall, onset/offset ms error and CPU RTF on a separate annotated validation slice; untouched core-test membership.
- Test mid-sentence pauses, trailing silence, short clips and delayed chunks; record endpoint-to-final and annotated-speech-end-to-final separately.

---

### 4. Required Output Artifacts
- `src/vad/baseline.py`
- `src/streaming/endpoint.py`
- `tests/test_vad.py`
- `tests/test_endpoint.py`
- `results/day23_endpointing.csv`

---

### 5. Completion Check
> **Definition of Done for Day 23:**
> Live finalization is tested and endpoint delay is measured, not postponed until after system profiling.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

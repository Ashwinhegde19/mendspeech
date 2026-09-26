# Day 26: Early streaming-to-editor integration and budget

> **Week 4 • Day 5 of 7**
> **Navigation:** [← Day 25](day_25.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 27 →](day_27.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 15](day_15.md), [Day 21](day_21.md), [Day 23](day_23.md), [Day 24](day_24.md), [Day 25](day_25.md)
> **Effort:** 2–4 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- End-of-speech finalization, editor trigger semantics and guard-before-delivery.

---

### 2. Build in MendSpeech
- Extend one app/audio_lab.py with replay/microphone streaming, final-ASR→editor request, raw bypass and visibly provisional output.
- Instrument all clocks in the latency contract; execute one-L4 joint-memory/interference test, not isolated-model timing only.

---

### 3. Experiment and Measure
- Collect raw/deterministic/LLM quality and correlated request-level stage intervals on validation.
- Record first partial, post-utterance ASR finalization, server/client TTFT and final usable text. State microphone versus paced replay and model residence/cold state.

---

### 4. Required Output Artifacts
- `app/audio_lab.py`
- `tests/test_pipeline_contract.py`
- `results/day26_e2e_baseline.csv`
- `docs/day26_latency_baseline.md`

---

### 5. Completion Check
> **Definition of Done for Day 26:**
> The full baseline, including streaming, endpointing, guarded editing and combined resource use, exists before optimization.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

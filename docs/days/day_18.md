# Day 18: Streaming checkpoint and decoder capability gate

> **Week 3 • Day 4 of 7**
> **Navigation:** [← Day 17](day_17.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 19 →](day_19.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 14](day_14.md), [Day 15](day_15.md)
> **Effort:** 2–4 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Cache-aware versus buffered inference, CTC/RNN-T head compatibility and chunk lookahead.

---

### 2. Build in MendSpeech
- Select one streaming-capable ASR checkpoint compatible with the planned runtime; record revision/head/tokenizer, cache signatures, chunk/right context and inference/export support in configs/model_baseline.yaml.
- Smoke one utterance offline and via the documented streaming interface; model/decoder changes invalidate old confidence thresholds. Verify the combined Day15 editor memory with the new checkpoint.

---

### 3. Experiment and Measure
- Record output/length/context evidence, offline/streaming and LM support separately. Do not rewrite a framework to force missing support.
- Compare supported baseline transcripts on validation and classify dependency blockers before writing optimizations.

---

### 4. Required Output Artifacts
- `src/asr/streaming_runner.py`
- `tests/test_streaming_runner.py`
- `configs/model_baseline.yaml`
- `results/day18_capability_check.csv`
- `docs/day18_capabilities.md`

---

### 5. Completion Check
> **Definition of Done for Day 18:**
> Pinned streaming/cache/head behavior and combined-resource feasibility are verified; unsupported live inference is a blocker, not a filename-based success.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

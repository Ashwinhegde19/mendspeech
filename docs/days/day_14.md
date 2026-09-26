# Day 14: Greedy, beam and one small LM comparison

> **Week 2 • Day 7 of 7**
> **Navigation:** [← Day 13](day_13.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 15 →](day_15.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 10](day_10.md), [Day 11](day_11.md), [Day 12](day_12.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- CTC collapse versus transducer search and external LM fusion.
- Decoder-only cached cost versus fresh end-to-end latency.

---

### 2. Build in MendSpeech
- Verify the current acoustic checkpoint/head and one supported decoder backend before comparing greedy, beam-only and beam+one n-gram LM. No second acoustic model or custom search engine.
- Record LM corpus license/hash/deduplication and exclude evaluation references; select a small predeclared beam/LM-weight list on validation only.
- Test token/blank/repeat mapping and alignment; initialize the shared repeatable benchmark harness with trace IDs, warm/cold and CPU worker metadata.

---

### 3. Experiment and Measure
- On identical validation cases report WER/CER, names/numbers, helpful and harmful changes and cached decoder versus fresh timing.
- Freeze selection for the later final test. Unsupported backend leaves decoding incomplete; independent baseline work may proceed with that blocker.

---

### 4. Required Output Artifacts
- `src/asr/decoding.py`
- `tests/test_asr_decoding.py`
- `configs/decoding.yaml`
- `data/lm_text_manifest.csv`
- `src/bench/benchmark_asr.py`
- `src/bench/environment.py`
- `tests/test_benchmark_asr.py`
- `results/day14_decoding_comparison.csv`
- `docs/day14_harmful_lm_changes.md`

---

### 5. Completion Check
> **Definition of Done for Day 14:**
> Three decoder conditions have controlled evidence and provenance; offline decoding is not claimed as live streaming and LM scores are not calibrated confidence.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

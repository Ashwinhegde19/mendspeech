# Day 30: Precision and export parity

> **Week 5 • Day 2 of 7**
> **Navigation:** [← Day 29](day_29.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 31 →](day_31.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 27](day_27.md)
> **Effort:** 2–4 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- FP16/INT8 (dynamic and static), calibration sets, exported-versus-original parity.

---

### 2. Build in MendSpeech
- Smoke-check export/precision support for the selected backend in an isolated pinned environment; document in docs/day30_quant_notes.md.
- Verify original-versus-exported parity first; then apply supported FP16/INT8 with a calibration slice from training/calibration only, never the frozen test.

---

### 3. Experiment and Measure
- Measure WER/CER, latency percentiles, RTF, memory and confidence/logit shifts per precision; recheck Day25 calibration binding.
- State explicitly if a precision is slower or degrades accuracy; blocked precision leaves the required comparison incomplete with recorded blocker.

---

### 4. Required Output Artifacts
- `src/asr/quantized_runner.py`
- `docs/day30_quant_notes.md`
- `results/day30_quantization_tradeoffs.csv`
- `app/audio_lab.py`

---

### 5. Completion Check
> **Definition of Done for Day 30:**
> Parity is verified before any precision claim, and every supported precision has measured accuracy/latency/memory or a documented blocker.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

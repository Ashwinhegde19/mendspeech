# Day 28: torch.compile and graph capture experiment

> **Week 4 • Day 7 of 7**
> **Navigation:** [← Day 27](day_27.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 29 →](day_29.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 27](day_27.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Graph capture, fusion, recompilation on new shapes/frames and static versus dynamic cost.

---

### 2. Build in MendSpeech
- Apply torch.compile to a pinned steady-state configuration and experiment with CUDA graphs in src/asr/optimized_runner.py. Pin static shapes; treat chunked streaming as dynamic.
- Test parity on same input/precision; separate warmup/compile time from p50/p95/p99.

---

### 3. Experiment and Measure
- Measure WER/latency/RTF/memory against Day27 including joint editor; count recompiles and reject measures that slow p99 or break parity.
- Negative/zero gains are documented with numbers; do not chase a checklist.

---

### 4. Required Output Artifacts
- `src/asr/optimized_runner.py`
- `tests/test_optimized_parity.py`
- `results/day28_compile.csv`

---

### 5. Completion Check
> **Definition of Done for Day 28:**
> A measured, parity-checked before/after for compilation/graph capture, or an honest zero-gain result.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

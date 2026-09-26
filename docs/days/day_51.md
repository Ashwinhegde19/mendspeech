# Day 51: Release SpeechDamageBench v1 and freeze the evaluation set

> **Week 8 • Day 2 of 7**
> **Navigation:** [← Day 50](day_50.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 52 →](day_52.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 37](day_37.md), [Day 50](day_50.md)
> **Effort:** 1–2 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Local CPU`

---

### 1. Learn
- Severity grids, speaker-separated evaluation, seed control and checksum verification.

---

### 2. Build in MendSpeech
- Finalize the standalone package and lock manifest checksums under benchmarks/.
- Document a one-command reproduction example in speechdamagebench/README.md.

---

### 3. Experiment and Measure
- Reinstall in a clean environment; regenerate a sample and verify its checksum.
- Verify clean references are byte-identical after regeneration.

---

### 4. Required Output Artifacts
- `speechdamagebench/CHANGELOG.md`
- `benchmarks/manifest.csv`
- `benchmarks/README.md`

---

### 5. Completion Check
> **Definition of Done for Day 51:**
> A clean environment reproduces a benchmark item from the manifest and clean references are provably unchanged.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

# Day 37: ASR robustness adaptation dataset and leakage audit

> **Week 6 • Day 2 of 7**
> **Navigation:** [← Day 36](day_36.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 38 →](day_38.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 10](day_10.md), [Day 18](day_18.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Acoustic corruption manifests and the frozen benchmark invariant.

---

### 2. Build in MendSpeech
- Build data/train_manifest.jsonl and val_manifest.jsonl from separate non-frozen source audio; keep data/test_manifest.jsonl byte-identical to the frozen set.
- Audit source/speaker leakage and corruption provenance; document in reports/data_audit.md.

---

### 3. Experiment and Measure
- Prove no source/speaker crosses splits; report severity distribution.
- The frozen test set is not used for training, tuning or checkpoint selection in this phase.

---

### 4. Required Output Artifacts
- `data/train_manifest.jsonl`
- `data/val_manifest.jsonl`
- `data/test_manifest.jsonl`
- `reports/data_audit.md`

---

### 5. Completion Check
> **Definition of Done for Day 37:**
> The adaptation dataset is leakage-free and the frozen evaluation set is provably untouched.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

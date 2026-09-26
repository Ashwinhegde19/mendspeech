# Day 38: ASR robustness fine-tuning (not personalization)

> **Week 6 • Day 3 of 7**
> **Navigation:** [← Day 37](day_37.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 39 →](day_39.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 24](day_24.md), [Day 25](day_25.md), [Day 37](day_37.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Transfer learning, frozen versus trainable layers, mixed precision. This is acoustic robustness, not user personalization.

---

### 2. Build in MendSpeech
- Fine-tune the pinned streaming checkpoint with training/asr_finetune.py under configs/asr_finetune.yaml (steps, LR, seed, sampling frozen).
- Bind and re-validate Day25 calibration for the adapted checkpoint before it informs triage.

---

### 3. Experiment and Measure
- Compare base vs adapted on the frozen test set per corruption/severity with clean-speech regression.
- An adaptation that helps damaged speech but harms clean speech is a documented trade-off; no personalization claim is made.

---

### 4. Required Output Artifacts
- `training/asr_finetune.py`
- `configs/asr_finetune.yaml`
- `results/day38_base_vs_adapted.csv`
- `reports/day38_robustness_adaptation.md`

---

### 5. Completion Check
> **Definition of Done for Day 38:**
> Acoustic robustness is measured with clean-speech regression on the frozen set, and is reported as adaptation rather than personalization.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

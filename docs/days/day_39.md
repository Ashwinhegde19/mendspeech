# Day 39: Augmentation ablation on corrupted audio

> **Week 6 • Day 4 of 7**
> **Navigation:** [← Day 38](day_38.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 40 →](day_40.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 38](day_38.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- SpecAugment, room impulse-response augmentation and training-time confounds.

---

### 2. Build in MendSpeech
- Run one controlled augmentation arm with identical steps/seed via experiments/augmentation_ablation.py.
- Test augmentation strength and label-preserving transforms; never alter the frozen set.

---

### 3. Experiment and Measure
- Compare no-augmentation vs augmentation at equal budget, then give the extra steps to the unaugmented baseline.
- Record the gain (or its absence) and per-corruption effect.

---

### 4. Required Output Artifacts
- `experiments/augmentation_ablation.py`
- `results/day39_augmentation.csv`

---

### 5. Completion Check
> **Definition of Done for Day 39:**
> The effect of augmentation is separated from the effect of extra training time.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

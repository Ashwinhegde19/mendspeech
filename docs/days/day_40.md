# Day 40: Bounded GRPO post-training on the editor

> **Week 6 • Day 5 of 7**
> **Navigation:** [← Day 39](day_39.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 41 →](day_41.md)

> **v4 STATUS: CORE** Highest-variance and most expensive session; feasibility and budget were gated on Day16.
> **Prerequisites:** [Day 16](day_16.md), [Day 34](day_34.md), [Day 35](day_35.md), [Day 36](day_36.md)
> **Effort:** 4–8 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4; spend/spend-capped`

---

### 1. Learn
- Group-relative policy optimization, KL to the SFT reference, reward variance, rollout cost.

---

### 2. Build in MendSpeech
- Run the bounded GRPO run using training/editor_rl.py with the Day34 reward and Day35 SFT reference; default to LoRA-scale updates, modest steps, and a declared stop/spend budget.
- Log reward curves, KL to reference, group reward variance, completion length, adapter grad norms and refusals.

---

### 3. Experiment and Measure
- Evaluate against the SFT and continued-SFT controls on validation: protected-content violations, formatting accuracy, identity vs needs-edit, risk-coverage.
- Stop on nonfinite loss, repeated OOM, or safety violations rising >2pp above the SFT baseline at two consecutive evals; a valid null is kept, a failed run is blocked not disguised.

---

### 4. Required Output Artifacts
- `training/editor_rl.py`
- `configs/editor_rl.yaml`
- `results/day40_rl_vs_sft.csv`
- `docs/day40_rl_findings.md`
- `results/day40_reward_curve.csv`

---

### 5. Completion Check
> **Definition of Done for Day 40:**
> A bounded, controlled GRPO run on the text editor with SFT/continued-SFT comparison and explicit stop criteria, or a documented blocked/null outcome.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

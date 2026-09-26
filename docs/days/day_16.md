# Day 16: Early SFT and GRPO feasibility on the text editor

> **Week 3 • Day 2 of 7**
> **Navigation:** [← Day 15](day_15.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 17 →](day_17.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 13](day_13.md), [Day 15](day_15.md)
> **Effort:** 4–6 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Completion-only SFT, LoRA gradients, group-relative advantages and reference KL.
- Why CTC acoustic outputs cannot be passed to a causal-LM GRPO trainer.

---

### 2. Build in MendSpeech
- Use the selected Day15 editor and a pinned compatible optional Transformers/PEFT/TRL environment. Implement training/editor_pilot.py with the exact bounded SFT/GRPO route in the editor contract, no scratch PPO or reward model.
- Implement provisional reward component tests; one pilot group uses fresh current-policy generations. Check names/counts of trainable adapters, completion masks/EOS and group/batch divisibility.
- Declare <=10 SFT and <=5 GRPO steps, group2, fixed token caps, wall-time/spend and nonfinite/OOM stop conditions. Unload ASR while training.

---

### 3. Experiment and Measure
- Show finite loss/logprobs/gradients and adapter weight changes, reload checkpoint and inspect completions. Measure policy/reference/optimizer/rollout memory, step time and cost.
- Log reward variance/zero-variance groups and KL; failed update or insufficient rollout diversity is blocked, not a null RL result. Forecast later training cost before approval.

---

### 4. Required Output Artifacts
- `training/editor_pilot.py`
- `configs/editor_pilot.yaml`
- `src/rl/reward.py`
- `tests/test_reward.py`
- `reports/day16_training_feasibility.md`
- `results/day16_training_pilot.csv`

---

### 5. Completion Check
> **Definition of Done for Day 16:**
> An actual tiny SFT and GRPO update on the causal text editor is verified, or the full training track remains explicitly blocked before further training expenditure.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

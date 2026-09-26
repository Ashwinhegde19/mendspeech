# Day 34: RL reward definition and falsifiability

> **Week 5 • Day 6 of 7**
> **Navigation:** [← Day 33](day_33.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 35 →](day_35.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 13](day_13.md), [Day 16](day_16.md), [Day 25](day_25.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Reward hacking, faithful rewards, and a bounded group-relative RL objective on the text editor.

---

### 2. Build in MendSpeech
- Design the conservative reward on the Day13 contract: edit/format fidelity, protected-span safety, length and fluency penalties, with a per-example safety floor.
- Show a short-falsifiable prediction before training: which validation failures the reward should reduce and which must not increase.
- Use the pilot path from Day16; do not train on the CTC acoustic model and do not handcraft a per-example rewrite.

---

### 3. Experiment and Measure
- Construct adversarial cases (empty/truncated output, negation removal, entity edits, prompt-echo, runaway length) and verify each is penalized.
- Confirm the reward is computable offline on validation; log reward variance and any zero-variance groups. A reward that cannot be gamed by refusal earns nothing on needs-edit cases.

---

### 4. Required Output Artifacts
- `src/rl/reward.py`
- `configs/editor_reward.yaml`
- `tests/test_reward.py`
- `docs/day34_reward_design.md`

---

### 5. Completion Check
> **Definition of Done for Day 34:**
> A written falsifiable prediction plus reward tests that demonstrate the failure modes the reward is designed to penalize.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

# Day 40: RL reward design

> **Week 6 • Day 5 of 7**  
> **Navigation:** [← Day 39](day_39.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 41 →](day_41.md)

> **v3 STATUS: CORE** The reward must be falsifiable, or the run proves nothing. This is the most novel session in the plan.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Policy-gradient and PPO intuition for sequence output.
- Reward hacking: what a model does when the reward is exploitable.
- Designing a reward that is falsifiable in advance.

---

### 2. Build in MendSpeech
- Define the reward in `src/rl/reward.py`: penalize fluent output that the acoustics do not support.
- Write the falsifiable prediction in `configs/rl.yaml` BEFORE running anything.
- Implement a minimal policy-gradient or PPO-style update in `src/rl/ppo.py`.

---

### 3. Experiment and Measure
- Show the reward can be gamed: construct at least one input where a naive reward rewards a wrong transcript.
- Verify the reward is computable offline from cached logits before spending GPU time.
- Unit-test reward components in `tests/test_reward.py`.

---

### 4. Required Output Artifacts
['- `src/rl/reward.py`', '- `src/rl/ppo.py`', '- `configs/rl.yaml`', '- `tests/test_reward.py`', '- `docs/day40_reward_design.md`']

---

### 5. Completion Check
> **Definition of Done for Day 40:**  
> You have a written falsifiable prediction, a reward shown to be gameable in at least one case, and a tested implementation.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- PPO and policy gradient references
- Reward design and specification gaming literature
- RLHF and ASR post-training

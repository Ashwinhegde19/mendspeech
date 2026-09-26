# Day 41: RL post-training run

> **Week 6 • Day 6 of 7**  
> **Navigation:** [← Day 40](day_40.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 42 →](day_42.md)

> **v3 STATUS: CORE** Base versus fine-tuned versus RL, on the same held-out data. A null result here is still a result.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Reward/advantage computation.
- KL regularization against the reference model.
- Why RL can degrade a well-calibrated model.

---

### 2. Build in MendSpeech
- Run the bounded RL post-training from `training/rl_train.py` using the Day 38 checkpoint as reference.
- Track reward, KL, and held-out WER together; reward rising while WER worsens is the key diagnostic.

---

### 3. Experiment and Measure
- Compare base, fine-tuned, and RL variants on held-out data.
- Re-run the Day 13 risk-coverage analysis for the RL model; improved WER does not imply improved triage safety.
- Record total GPU cost against the declared budget in `results/day41_rl_vs_baseline.csv`.

---

### 4. Required Output Artifacts
['- `training/rl_train.py`', '- `results/day41_rl_vs_baseline.csv`', '- `docs/day41_rl_findings.md`', '- `results/day41_risk_coverage.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 41:**  
> You can state whether RL helped, did nothing, or hurt, with evidence, and you checked triage safety rather than WER alone.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- PPO implementation references
- KL-regularized policy optimization
- RL fine-tuning stability

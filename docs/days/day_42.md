# Day 42: Personalization comparison and robustness milestone

> **Week 6 • Day 7 of 7**  
> **Navigation:** [← Day 41](day_41.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 43 →](day_43.md)

> **v3 STATUS: CORE** One table answers the personalization question and closes Phase P5.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Separating adaptation effects from training-time effects.
- Reporting a null result without overclaiming.

---

### 2. Build in MendSpeech
- Produce the final base/fine-tuned/RL comparison in `reports/day42_personalization.md`.
- Extend `app/audio_lab.py` to switch between the base, fine-tuned, and RL checkpoints.

---

### 3. Experiment and Measure
- Report WER per corruption and severity for all three checkpoints, plus clean-speech regression.
- Report the risk-coverage curve for each checkpoint.
- State plainly which checkpoint ships and why the choice rests on measured evidence.

---

### 4. Required Output Artifacts
['- `reports/day42_personalization.md`', '- `results/day42_personalization_matrix.csv`', '- `app/audio_lab.py`']

---

### 5. Completion Check
> **Definition of Done for Day 42:**  
> The personalization question is answered with a table and a shipping recommendation, including any null results.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Phase 5 evidence requirements in the execution plan

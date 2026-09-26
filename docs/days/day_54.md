# Day 54: Personalization comparison on the frozen harness

> **Week 8 • Day 5 of 7**  
> **Navigation:** [← Day 53](day_53.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 55 →](day_55.md)

> **v3 STATUS: CORE** Base, fine-tuned, and RL, measured once on data frozen before any of them ran.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Why the final comparison must use the frozen set, not a convenient one.
- Reporting regression as carefully as improvement.

---

### 2. Build in MendSpeech
- Run the three checkpoints through the frozen harness in `src/bench/run_personalization.py`.

---

### 3. Experiment and Measure
- Report WER, risk-coverage, and clean-speech regression for base, fine-tuned, and RL.
- Report what RL cost in GPU time against what it bought.
- State plainly whether personalization earned its place in the pipeline.

---

### 4. Required Output Artifacts
['- `src/bench/run_personalization.py`', '- `results/day54_personalization_final.csv`', '- `reports/day54_personalization.md`']

---

### 5. Completion Check
> **Definition of Done for Day 54:**  
> The personalization decision is made on frozen evidence, including the case where it did not pay off.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Phase 5 requirements in the execution plan

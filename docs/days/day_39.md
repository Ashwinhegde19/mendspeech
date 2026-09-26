# Day 39: Augmentation ablation

> **Week 6 • Day 4 of 7**  
> **Navigation:** [← Day 38](day_38.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 40 →](day_40.md)

> **v3 STATUS: CORE** Confound control: augmentation must be separated from extra training time.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Time masking.
- Frequency masking.
- Data augmentation as invariance training.

---

### 2. Build in MendSpeech
- Add one augmentation intervention to a controlled short run in `experiments/specaugment_ablation.py`.

---

### 3. Experiment and Measure
- Compare no augmentation versus selected augmentation with the same seed and the same step budget.
- Report whether the gain survives when the extra steps are given to the unaugmented baseline.

---

### 4. Required Output Artifacts
['- `experiments/specaugment_ablation.py`', '- `results/day39_augmentation.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 39:**  
> You can separate the effect of augmentation from the effect of extra training time.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- SpecAugment paper
- NVIDIA NeMo augmentation documentation

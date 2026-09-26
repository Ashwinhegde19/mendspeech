# Day 36: Training pipeline anatomy

> **Week 6 • Day 1 of 7**  
> **Navigation:** [← Day 35](day_35.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 37 →](day_37.md)

> **v3 STATUS: CORE** Phase P5 begins. You cannot fine-tune responsibly if you cannot read a loss curve.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Manifest format.
- Batching variable-duration audio.
- Loss curves.
- Learning rate, validation split, checkpointing.

---

### 2. Build in MendSpeech
- Create one reproducible training configuration in `configs/train_smoke.yaml`.
- Implement the training loop in `training/train.py` with checkpointing and validation hooks.

---

### 3. Experiment and Measure
- Run a short smoke job and verify the loss decreases.
- Deliberately use a bad learning rate and record the failure signature in `results/day36_training_smoke.csv`.
- Verify checkpoints reload and reproduce the same validation number.

---

### 4. Required Output Artifacts
['- `configs/train_smoke.yaml`', '- `training/train.py`', '- `tests/test_train_loop.py`', '- `results/day36_training_smoke.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 36:**  
> You can diagnose whether a run is learning, diverging, or overfitting from basic evidence, and a checkpoint reloads reproducibly.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- NVIDIA NeMo ASR training documentation
- Mixed precision and gradient accumulation

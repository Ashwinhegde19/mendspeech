# Day 38: Fine-tune for damaged-speech robustness

> **Week 6 • Day 3 of 7**  
> **Navigation:** [← Day 37](day_37.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 39 →](day_39.md)

> **v3 STATUS: CORE** The personalization experiment: base versus adapted, measured with a clean-speech regression check.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Transfer learning.
- Frozen versus trainable layers.
- Mixed precision.
- Gradient accumulation.

---

### 2. Build in MendSpeech
- Fine-tune the Day 24 checkpoint on the Day 37 dataset using `training/finetune.py`.
- Freeze trainable layers, learning rate, steps, seed, and sampling in `configs/finetune.yaml`.

---

### 3. Experiment and Measure
- Compare base and adapted models on the frozen test set, reporting WER per corruption and severity.
- Measure clean-speech regression explicitly; an adaptation that helps damaged speech but harms clean speech is a documented tradeoff, not a win.
- Record actual training time and cost.

---

### 4. Required Output Artifacts
['- `training/finetune.py`', '- `configs/finetune.yaml`', '- `reports/day38_adaptation.md`', '- `results/day38_base_vs_adapted.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 38:**  
> You can state exactly what improved, what did not, and whether clean speech regressed.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- NVIDIA NeMo ASR training documentation
- Transfer learning for speech recognition

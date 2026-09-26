# Day 16: Conformer convolution module

> **Week 3 • Day 2 of 7**  
> **Navigation:** [← Day 15](day_15.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 17 →](day_17.md)

> **v3 STATUS: CORE** Depthwise convolution is what makes a Conformer cheaper than attention alone at long sequence lengths.
---

### Compute Target
`Local CPU`

---

### 1. Learn
- Depthwise separable convolution.
- Receptive field and locality.
- Causality assumptions for a streaming encoder.

---

### 2. Build in MendSpeech
- Implement a Conformer-style convolution module in `src/models/conv_module.py`.
- Test causality assumptions and receptive field growth in `tests/test_conv_module.py`.

---

### 3. Experiment and Measure
- Feed synthetic impulses and inspect how local information spreads.
- Measure receptive field against layer count and compare with the analytic prediction.

---

### 4. Required Output Artifacts
['- `src/models/conv_module.py`', '- `tests/test_conv_module.py`', '- `results/day16_receptive_field.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 16:**  
> You can explain why depthwise convolution is cheap and what local context it captures relative to attention.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Gulati et al., Conformer
- NVIDIA NeMo Conformer implementation

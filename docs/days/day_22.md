# Day 22: Why FastConformer exists

> **Week 4 • Day 1 of 7**  
> **Navigation:** [← Day 21](day_21.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 23 →](day_23.md)

> **v3 STATUS: LEARN-ONLY** Merged into Day 23; paper notes and compute estimates only.
---

### Compute Target
`Local CPU`

---

### 1. Learn
- Sequence length as an attention cost driver.
- Subsampling before expensive encoder blocks.
- Depthwise separable convolution.
- Local and limited context attention.

---

### 2. Build in MendSpeech
- No standalone build. Day 23 incorporates the comparison checklist and diagram.

---

### 3. Experiment and Measure
- Day 23 incorporates attention-matrix estimates before and after temporal subsampling.

---

### 4. Required Output Artifacts
['None for this learn-only session; retained notes and estimate paths are produced within Day 23.']

---

### 5. Completion Check
> **Definition of Done for Day 22:**  
> You can explain FastConformer as a set of concrete efficiency choices, not just a faster model name.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Rekesh et al., FastConformer
- NVIDIA NeMo FastConformer documentation

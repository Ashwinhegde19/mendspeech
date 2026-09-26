# Day 21: Week 3 architecture review

> **Week 3 • Day 7 of 7**  
> **Navigation:** [← Day 20](day_20.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 22 →](day_22.md)

> **v2 STATUS: CORE — static architecture review.** Keep the review milestone and shape evidence; no architecture-inspector UI.

---

### Compute Target
`Local CPU`

---

### 1. Learn
- Review attention, convolution, feed forward, normalization, residual paths, and sequence cost.

---

### 2. Build in MendSpeech
- Write a static architecture report showing the single block's stage shapes, mask propagation, and context assumptions, using `results/day19_shape_trace.md` from Day 18.

---

### 3. Experiment and Measure
- Give yourself a ten minute whiteboard explanation from waveform features through one Conformer block.

---

### 4. Required Output Artifacts
- `reports/week3_conformer.md`

---

### 5. Completion Check
> **Definition of Done for Day 21:**  
> You can explain which parts are local, which are global, and which become
> problematic for streaming, with the static report linked to the tested shape trace.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Conformer primary paper
- A mature Conformer implementation such as NVIDIA NeMo

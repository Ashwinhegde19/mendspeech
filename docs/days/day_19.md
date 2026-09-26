# Day 19: Real-log-Mel block validation

> **Week 3 • Day 5 of 7**  
> **Navigation:** [← Day 18](day_18.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 20 →](day_20.md)

> **v3 STATUS: MERGED** into Day 18. No standalone session.
---

### Compute Target
`Local CPU — within Day 18`

---

### 1. Learn
- Input projection and mask propagation.

---

### 2. Build in MendSpeech
- Validate projection, masks, shapes, and gradients on Day 18's same block; do not build a separate encoder.

---

### 3. Experiment and Measure
- No depth sweep. The shape trace in `results/day19_shape_trace.md` is produced within Day 18.

---

### 4. Required Output Artifacts
['- `results/day19_shape_trace.md` — produced in Day 18']

---

### 5. Completion Check
> **Definition of Done for Day 19:**  
> A real log-Mel tensor passes through the block with documented shapes and valid gradients.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Day 18 artifacts and references

# Day 21: Architecture review: what dominates streaming latency

> **Week 3 • Day 7 of 7**  
> **Navigation:** [← Day 20](day_20.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 22 →](day_22.md)

> **v3 STATUS: CORE** The review now asks a systems question rather than a memorization question.
---

### Compute Target
`Local CPU`

---

### 1. Learn
- Review attention, convolution, feed-forward, normalization, and residual paths.
- Which operations scale with sequence length, and which are constant per chunk.

---

### 2. Build in MendSpeech
- Write `reports/week3_conformer.md` naming the operations that dominate streaming cost, using the shape trace as evidence.
- No separate inspector UI; the report is the artifact.

---

### 3. Experiment and Measure
- Give a ten-minute whiteboard explanation from waveform features through one block to a latency claim.

---

### 4. Required Output Artifacts
['- `reports/week3_conformer.md`', '- `results/day19_shape_trace.md`']

---

### 5. Completion Check
> **Definition of Done for Day 21:**  
> You can explain which parts are local, which are global, and which become the bottleneck under streaming.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Gulati et al., Conformer
- NVIDIA NeMo Conformer implementation

# Day 53: Optimization and serving ablations

> **Week 8 • Day 4 of 7**  
> **Navigation:** [← Day 52](day_52.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 54 →](day_54.md)

> **v3 STATUS: CORE** Fixed inputs, one variable at a time, and Pareto frontiers rather than a single winner.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Pareto frontiers: when no configuration dominates.
- Holding inputs fixed so comparisons mean something.

---

### 2. Build in MendSpeech
- Run every optimization variant and serving configuration on the identical frozen subset in `src/bench/run_ablations.py`.

---

### 3. Experiment and Measure
- Plot WER against p99 latency and mark Pareto-efficient points in `results/day53_pareto.png`.
- Report serving configurations separately from model-level optimizations.
- Keep live measurements separate from any simulated estimate.

---

### 4. Required Output Artifacts
['- `src/bench/run_ablations.py`', '- `results/day53_ablations.csv`', '- `results/day53_pareto.png`']

---

### 5. Completion Check
> **Definition of Done for Day 53:**  
> You can say which configuration to ship and which trade-offs are unavoidable, with measured frontiers.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Multi-objective evaluation and Pareto analysis

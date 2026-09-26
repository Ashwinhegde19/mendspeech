# Day 52: Robustness matrix on the frozen set

> **Week 8 • Day 3 of 7**  
> **Navigation:** [← Day 51](day_51.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 53 →](day_53.md)

> **v3 STATUS: CORE** The full corruption x severity x decoder grid, on data nobody can now change.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Why a full matrix beats spot checks.
- Multiple-comparison discipline when slicing results.

---

### 2. Build in MendSpeech
- Run the full matrix through the frozen harness in `src/bench/run_matrix.py`.

---

### 3. Experiment and Measure
- Report WER/CER and confidence behaviour for every corruption, severity, and decoder combination.
- Identify the corruption/decoder pair with the worst risk-coverage behaviour.
- Repeat enough runs to estimate variance on a representative subset.

---

### 4. Required Output Artifacts
['- `src/bench/run_matrix.py`', '- `results/day52_robustness_matrix.csv`', '- `results/day52_robustness_matrix.png`']

---

### 5. Completion Check
> **Definition of Done for Day 52:**  
> The complete matrix is measured and the worst cell is identified, with variance estimated on a subset.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Evaluation methodology for sliced results

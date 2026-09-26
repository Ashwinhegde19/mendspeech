# Day 15: Attention for speech sequences

> **Week 3 • Day 1 of 7**  
> **Navigation:** [← Day 14](day_14.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 16 →](day_16.md)

> **v3 STATUS: CORE** Attention cost is why streaming needs a stateful encoder rather than a windowed one.
---

### Compute Target
`Local CPU, L4 optional for scaling`

---

### 1. Learn
- Query, key, value projections.
- Scaled dot product attention.
- Attention masks.
- Quadratic cost in sequence length and why long-form audio suffers.

---

### 2. Build in MendSpeech
- Implement single-head then multi-head attention in `src/models/attention.py`.
- Add shape assertions and gradient tests in `tests/test_attention.py`.

---

### 3. Experiment and Measure
- Change sequence length and measure forward time and peak memory.
- Plot the quadratic cost you predicted against the cost you measured.

---

### 4. Required Output Artifacts
['- `src/models/attention.py`', '- `tests/test_attention.py`', '- `results/day15_attention_cost.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 15:**  
> You can derive every major tensor shape from memory and explain the quadratic term streaming must avoid.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Vaswani et al., Attention Is All You Need
- Annotated transformer implementations

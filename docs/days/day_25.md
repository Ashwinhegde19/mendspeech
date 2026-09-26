# Day 25: Context and lookahead cost

> **Week 4 • Day 4 of 7**  
> **Navigation:** [← Day 24](day_24.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 26 →](day_26.md)

> **v3 STATUS: CORE** Lookahead is a latency knob; this session measures what it costs and buys.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Right context versus left context.
- Algorithmic latency versus accuracy.
- Why future context cannot be free.

---

### 2. Build in MendSpeech
- Implement fixed left/right context settings in `src/streaming/context.py`.
- Log the context configuration with every result.

---

### 3. Experiment and Measure
- Compare at least two supported context settings on the same subset.
- Plot WER against measured algorithmic latency and mark the Pareto-efficient points.
- Report where errors cluster near chunk boundaries.

---

### 4. Required Output Artifacts
['- `src/streaming/context.py`', '- `tests/test_context.py`', '- `results/day25_context_tradeoff.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 25:**  
> You can explain exactly why future context creates latency, with a measured curve rather than an assertion.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Stateful Conformer primary material
- NVIDIA NeMo streaming ASR documentation

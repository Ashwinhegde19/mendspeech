# Day 23: Temporal subsampling experiment

> **Week 4 • Day 2 of 7**  
> **Navigation:** [← Day 22](day_22.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 24 →](day_24.md)

> **v1 STATUS: CORE — absorbs Day 22.** Also cover Day 22's FastConformer-vs-Conformer comparison checklist and compute estimates in this session.

---

### Compute Target
`Modal L4 useful`

---

### 1. Learn
- Convolutional subsampling.
- Temporal resolution.
- Information loss versus compute reduction.

---

### 2. Build in MendSpeech
- Implement a small subsampling front end or isolate one from a framework.
- Track frames per second before and after each stage.
- Incorporate Day 22's comparison checklist and diagram of FastConformer efficiency choices.

---

### 3. Experiment and Measure
- Compare 2x, 4x, and 8x temporal reduction on tensor length, runtime, and rough output behavior.
- Estimate attention-matrix size before and after subsampling; record the absorbed Day 22 evidence.

---

### 4. Required Output Artifacts
- `src/models/subsampling.py`
- `results/day23_subsampling.csv`
- `docs/day22_fastconformer_notes.md` — absorbed Day 22 evidence
- `results/day22_compute_estimates.csv` — absorbed Day 22 evidence

---

### 5. Completion Check
> **Definition of Done for Day 23:**  
> You can quantify how subsampling changes sequence length and downstream
attention cost.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- FastConformer primary paper
- NVIDIA NeMo FastConformer model documentation

# Day 27: Profiling the streaming model

> **Week 4 • Day 6 of 7**  
> **Navigation:** [← Day 26](day_26.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 28 →](day_28.md)

> **v3 STATUS: CORE** Phase P4 begins. Measure before optimizing, or you optimize the wrong thing.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Where time actually goes in a streaming forward pass.
- CPU launch overhead versus GPU compute time.
- Kernel-level versus end-to-end timing.

---

### 2. Build in MendSpeech
- Add per-operator profiling to the Day 26 harness in `src/bench/profile_ops.py`.
- Produce a ranked operator table for one fixed configuration.

---

### 3. Experiment and Measure
- Rank operators by measured time and separate launch overhead from compute.
- Identify the top three candidates for optimization and state the expected ceiling for each.
- Write the baseline row into `results/day27_operator_profile.csv`.

---

### 4. Required Output Artifacts
['- `src/bench/profile_ops.py`', '- `results/day27_operator_profile.csv`', '- `docs/day27_optimization_targets.md`']

---

### 5. Completion Check
> **Definition of Done for Day 27:**  
> You can name the top three time consumers with measured evidence and an expected gain for each.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- PyTorch profiler
- NVIDIA Nsight Systems
- Kernel launch overhead references

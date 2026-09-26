# Day 45: Load test to saturation

> **Week 7 • Day 3 of 7**  
> **Navigation:** [← Day 44](day_44.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 46 →](day_46.md)

> **v3 STATUS: CORE** The knee, not the maximum, is the number that matters.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Load testing methodology: fixed hardware, fixed input, fixed configuration.
- Saturation behaviour and queue growth.
- Why throughput at saturation is not a user experience.

---

### 2. Build in MendSpeech
- Implement the load harness in `src/serve/loadtest.py` with configurable concurrency and fixed input.

---

### 3. Experiment and Measure
- Sweep concurrency until latency degrades; report the knee in `results/day45_load_curve.csv`.
- Report per-stream p50/p95/p99, queue depth, and dropped or delayed chunks at each level.
- Reproduce one controlled overload failure and one recovery in `docs/day45_failure_recovery.md`.

---

### 4. Required Output Artifacts
['- `src/serve/loadtest.py`', '- `results/day45_load_curve.csv`', '- `docs/day45_failure_recovery.md`', '- `reports/day45_serving.md`']

---

### 5. Completion Check
> **Definition of Done for Day 45:**  
> You can name the concurrency knee with measured evidence and show a reproduced failure and recovery.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Load testing methodology
- Queueing and saturation behaviour

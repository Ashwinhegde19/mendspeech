# Day 28: torch.compile and graph capture

> **Week 4 • Day 7 of 7**  
> **Navigation:** [← Day 27](day_27.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 29 →](day_29.md)

> **v3 STATUS: CORE** First optimization technique, measured against the Day 27 profile rather than assumed.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- torch.compile: graph capture, fusion, and recompilation triggers.
- Dynamic shapes and why recompilation is silent and expensive.
- CUDA graphs for static-shape workloads.

---

### 2. Build in MendSpeech
- Apply torch.compile to the hot path in `src/asr/optimized_runner.py`, pinning shapes to avoid recompilation.
- Add a CUDA-graph fast path only for static-shape inputs in `src/asr/optimized_runner.py`.

---

### 3. Experiment and Measure
- Measure WER, p50/p95/p99 latency, RTF, and memory against the Day 27 baseline on identical inputs.
- Record compile time and warmup separately from steady-state latency.
- Verify output parity against the unoptimized path; a speedup with different transcripts is not a speedup.

---

### 4. Required Output Artifacts
['- `src/asr/optimized_runner.py`', '- `tests/test_optimized_parity.py`', '- `results/day28_compile_speedup.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 28:**  
> You have a measured before/after for compilation with verified output parity, or the failure and its cause documented.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- PyTorch torch.compile documentation
- CUDA graph capture documentation

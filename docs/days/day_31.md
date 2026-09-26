# Day 31: Streaming fast path

> **Week 5 • Day 3 of 7**  
> **Navigation:** [← Day 30](day_30.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 32 →](day_32.md)

> **v3 STATUS: CORE** Optimized offline inference does not automatically make streaming fast; this session checks.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- State reuse versus recomputation across chunks.
- Where redundant computation remains in a cache-aware encoder.

---

### 2. Build in MendSpeech
- Add a streaming-specific fast path in `src/streaming/fast_path.py` reusing Day 30's best variant.
- Assert cached and uncached streaming produce equivalent transcripts.

---

### 3. Experiment and Measure
- Measure steady-state per-chunk latency after warmup, separately from the first chunk.
- Report the speedup of the fast path against the Day 30 baseline, or state that it did not help.

---

### 4. Required Output Artifacts
['- `src/streaming/fast_path.py`', '- `tests/test_fast_path_parity.py`', '- `results/day31_fast_path.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 31:**  
> You can state measured steady-state streaming latency and whether the fast path earned its complexity.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Stateful Conformer primary material
- NVIDIA NeMo streaming ASR documentation

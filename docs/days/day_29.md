# Day 29: Batching and throughput

> **Week 5 • Day 1 of 7**  
> **Navigation:** [← Day 28](day_28.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 30 →](day_30.md)

> **v3 STATUS: CORE** Throughput and latency are different axes; this session keeps them separate.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Static versus dynamic batching.
- Queueing delay versus service time.
- Why throughput gains can hurt single-stream latency.

---

### 2. Build in MendSpeech
- Implement batched inference in `src/serve/batching.py` with a configurable batch policy.
- Expose batch size and queue wait as separately logged quantities.

---

### 3. Experiment and Measure
- Sweep batch size and report throughput and per-request latency separately.
- Find the batch size where queueing delay starts to dominate.
- Report RTF at the best throughput point and the latency at the lowest-concurrency point.

---

### 4. Required Output Artifacts
['- `src/serve/batching.py`', '- `results/day29_batch_sweep.csv`', '- `docs/day29_queueing.md`']

---

### 5. Completion Check
> **Definition of Done for Day 29:**  
> You can state the throughput/latency knee with measured evidence and explain what happens past it.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Queueing theory for inference servers
- vLLM and TGI batching documentation

# Day 47: Per-stage latency budget decomposition

> **Week 7 • Day 5 of 7**  
> **Navigation:** [← Day 46](day_46.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 48 →](day_48.md)

> **v3 STATUS: CORE** The headline artifact of the whole project. The question is where the time actually goes, not what feels slow.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Separating queueing, model, decoding, network, and serialization time.
- Why a blended average hides the tail that users feel.

---

### 2. Build in MendSpeech
- Instrument every stage in `src/bench/budget.py` using the Day 26 harness conventions.

---

### 3. Experiment and Measure
- Decompose waveform-to-polished-text into VAD/endpointing, ASR, decode, LLM TTFT, LLM full response, and network/serialization.
- Report p50/p95/p99 per stage in `results/day47_latency_budget.csv`.
- Name the single stage that owns the p99 and state the largest available optimization target in `docs/day47_latency_budget.md`.

---

### 4. Required Output Artifacts
['- `src/bench/budget.py`', '- `results/day47_latency_budget.csv`', '- `docs/day47_latency_budget.md`', '- `results/day47_latency_budget.png`']

---

### 5. Completion Check
> **Definition of Done for Day 47:**  
> You can point at the stage that owns the tail with per-stage percentiles, and the claim is reproducible from one command.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Latency attribution methodology
- Tail latency in distributed systems

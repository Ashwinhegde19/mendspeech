# Day 44: Async streaming service

> **Week 7 • Day 2 of 7**  
> **Navigation:** [← Day 43](day_43.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 45 →](day_45.md)

> **v3 STATUS: CORE** One provider, one endpoint. The service must be measurable, not merely working.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- FastAPI and async WebSocket handling.
- Per-stream state isolation.
- Correct cancellation when a client disconnects mid-utterance.

---

### 2. Build in MendSpeech
- Implement the service in `src/serve/app.py` around the optimized Day 32 configuration.
- Containerize reproducibly in `infra/serve/`.

---

### 3. Experiment and Measure
- Verify concurrent streams do not share or corrupt cache state.
- Confirm a mid-utterance disconnect leaves no orphaned GPU work.
- Report cold start separately from warm latency.

---

### 4. Required Output Artifacts
['- `src/serve/app.py`', '- `tests/test_serve_isolation.py`', '- `infra/serve/Dockerfile`', '- `infra/serve/README.md`']

---

### 5. Completion Check
> **Definition of Done for Day 44:**  
> Concurrent streams are isolated, disconnects are clean, and cold start is reported separately from warm latency.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- FastAPI WebSocket documentation
- Async Python concurrency patterns

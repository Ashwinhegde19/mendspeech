# Day 43: Serving contract and message schema

> **Week 7 • Day 1 of 7**  
> **Navigation:** [← Day 42](day_42.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 44 →](day_44.md)

> **v3 STATUS: CORE** Phase P6 begins. Design the contract before implementing, or latency semantics get baked in wrong.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- WebSocket message schemas for streaming audio and incremental transcripts.
- What belongs in a partial result versus a final result.
- Backpressure semantics at the protocol level.

---

### 2. Build in MendSpeech
- Define the WebSocket message schema in `src/serve/schema.py`: audio chunks in, partial and final transcripts with confidence and latency out.
- Define timeout, disconnect, and cancellation behaviour in `src/serve/schema.py`.

---

### 3. Experiment and Measure
- Write the contract as a testable specification in `docs/day43_serving_contract.md`.
- Verify the schema round-trips in `tests/test_serve_schema.py`.

---

### 4. Required Output Artifacts
['- `src/serve/schema.py`', '- `tests/test_serve_schema.py`', '- `docs/day43_serving_contract.md`']

---

### 5. Completion Check
> **Definition of Done for Day 43:**  
> The message contract is explicit about partial versus final results, latency fields, and failure semantics.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- WebSocket protocol design
- Streaming API design patterns

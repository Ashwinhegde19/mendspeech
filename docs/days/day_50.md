# Day 50: Freeze the evaluation protocol

> **Week 8 • Day 1 of 7**  
> **Navigation:** [← Day 49](day_49.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 51 →](day_51.md)

> **v3 STATUS: CORE** Phase P7 begins. Freeze before measuring, or the measurement decides the protocol.
---

### Compute Target
`Local CPU with Modal L4 dry run`

---

### 1. Learn
- What makes an evaluation protocol reproducible.
- Pre-registering claims so results cannot be reinterpreted afterwards.

---

### 2. Build in MendSpeech
- Freeze code, model, and data revisions, hardware, corruption configs, and metrics in `configs/frozen.yaml`.
- Define baselines and claims you will NOT make in `experiments/protocol.md`.
- Define null outcomes and failure criteria in advance.

---

### 3. Experiment and Measure
- Run a dry run to confirm every required field has a measurement or an explicit status.
- Scope every claim to the benchmark scale and state the statistical caveat.

---

### 4. Required Output Artifacts
['- `configs/frozen.yaml`', '- `experiments/protocol.md`', '- `docs/day50_protocol.md`']

---

### 5. Completion Check
> **Definition of Done for Day 50:**  
> Another engineer can reproduce the supported comparisons and knows exactly which claims are out of scope.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Experimental design and pre-registration references

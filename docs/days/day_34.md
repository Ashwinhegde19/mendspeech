# Day 34: Bounded adaptive-context comparison

> **Week 5 • Day 6 of 7**  
> **Navigation:** [← Day 33](day_33.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 35 →](day_35.md)

> **v3 STATUS: CORE** , capability-bounded. Live switching only if the checkpoint supports it; otherwise report the deferral honestly.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Policy-driven context selection.
- Confidence smoothing.
- Latency budget.
- Stability versus oscillation.

---

### 2. Build in MendSpeech
- Implement a capability-guarded policy in `src/controller/adaptive_context.py` that classifies chunks as easy or uncertain.
- Compare at most two supported right-context settings from Day 25.
- Return an explicit unavailable status when fewer than two settings are supported.

---

### 3. Experiment and Measure
- Compare fixed-fast, fixed-accurate, and the bounded adaptive policy with an explicit `live`, `simulated`, or `unavailable` status.
- Keep simulated estimates separate from measured latency; do not count cached reuse as a runtime gain.

---

### 4. Required Output Artifacts
['- `src/controller/adaptive_context.py`', '- `results/day34_adaptive_context.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 34:**  
> You have a measured live comparison, a clearly limited simulated comparison, or an evidence-backed unavailable result.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Stateful Conformer primary material
- NVIDIA NeMo streaming ASR documentation

# Day 34: Bounded adaptive-context comparison

> **Week 5 • Day 6 of 7**  
> **Navigation:** [← Day 33](day_33.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 35 →](day_35.md)

> **v2 STATUS: CORE — capability-bounded comparison.** Use only Day 24/32 supported settings. Live adaptation is conditional; unavailable support defers the adaptive claim, not the remaining streaming, endpointing, cache, or serving requirements.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Policy driven context selection.
- Confidence smoothing.
- Latency budget.
- Stability versus oscillation.

---

### 2. Build in MendSpeech
- Implement a small capability-guarded policy that classifies chunks as easy or uncertain using the existing uncertainty signal and fixed decision rules.
- Compare at most two supported right-context settings from Day 32. Use live switching only when the checkpoint supports it; otherwise simulate policy choices from separate fixed-setting runs on the same controlled subset.
- If fewer than two settings are supported, return an explicit unavailable status and reason. No custom serving infrastructure, new model, or architecture change to force adaptation.

---

### 3. Experiment and Measure
- Compare the two fixed policies and the bounded adaptive policy where supported, with an explicit `live`, `simulated`, or `unavailable` status for the adaptive comparison.
- A simulated comparison is offline policy evidence, not live adaptive latency. Keep measured fixed-run timings separate; leave adaptive latency missing unless measured on a real live switching run. Do not fabricate measurements or claim a benefit from a simulation alone.

---

### 4. Required Output Artifacts
- `src/controller/adaptive_context.py` — bounded policy and capability guard; reports unavailable when unsupported
- `results/day34_adaptive_context.csv` — retain this path even when unavailable; include status, supported settings, evidence/reason, and missing values for unmeasured metrics

---

### 5. Completion Check
> **Definition of Done for Day 34:**  
> You have a measured live comparison, an explicitly limited simulated policy
> comparison, or an evidence-backed unavailable result. Unsupported behavior
> honestly defers the adaptive claim without waiving the rest of Gate 4.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Stateful or cache aware Conformer primary material
- NVIDIA NeMo streaming ASR documentation and examples

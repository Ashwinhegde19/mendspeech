# Day 52: Run recognition and context ablations

> **Week 8 • Day 3 of 7**  
> **Navigation:** [← Day 51](day_51.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 53 →](day_53.md)

> **v2 STATUS: CORE — absorbs [Day 53](day_53.md).** Run recognition/context and repair/seam ablations together on the frozen harness; evidence, not elapsed sessions, closes the gate.

---

### Compute Target
`Modal L4, keep hardware fixed`

---

### 1. Learn
- Fixed lookahead comparison.
- Adaptive context policy.
- WER, latency, RTF, memory, confidence behavior.

---

### 2. Build in MendSpeech
- Run every streaming condition on the exact same benchmark subset.
- Repeat timing runs enough to estimate variance.
- Record GPU type and environment automatically through the Modal runner.
- Record `execution_mode=live|simulated` for every context policy. A live
  adaptive claim requires runtime context changes in the recognizer, not
  cached-output selection, offline scheduling, or an oracle decision rule.
- Keep simulated-policy cost estimates separate from measured latency/RTF;
  do not count cached reuse or a hypothetical context reduction as runtime gains.

---

### 3. Experiment and Measure
- Plot WER versus measured latency and mark Pareto efficient live points.
  Label simulated analyses separately with their assumptions; do not mix
  estimates into a measured frontier. Null or worse adaptive outcomes are valid.

---

### 4. Required Output Artifacts
- `results/capstone_streaming.csv`
- `results/streaming_pareto.png`

---

### 5. Completion Check
> **Definition of Done for Day 52:**  
> You can say whether implemented adaptive context helped, hurt, or made no
> meaningful difference. If only simulation is available, state that limitation
> and defer live runtime claims rather than fabricate speedups.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Your frozen protocol and prior results
- The frozen context-policy protocol and live/simulated execution labels
- Primary papers only when needed to interpret a result

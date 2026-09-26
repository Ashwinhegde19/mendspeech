# Day 13: Confidence thresholds and error triage policy

> **Week 2 • Day 6 of 7**  
> **Navigation:** [← Day 12](day_12.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 14 →](day_14.md)

> **v3 STATUS: CORE — triage, not repair.** Replaces the old repair-policy session. A downstream consumer needs to know accept, low-confidence, or reject; it does not need a synthesis policy.
---

### Compute Target
`Local CPU`

---

### 1. Learn
- Choosing a threshold from validation data rather than test data.
- Risk-coverage: what fraction of traffic a threshold accepts and at what error rate.
- Why a single threshold is a policy decision, not a modelling result.

---

### 2. Build in MendSpeech
- Implement three triage policies — accept, low-confidence, reject — in `src/controller/triage.py`.
- Each policy maps confidence to an action with a reason code in `src/controller/triage.py`.

---

### 3. Experiment and Measure
- Sweep thresholds on the validation split and plot risk against coverage.
- Report the accepted fraction and the error rate inside the accepted set per corruption type.
- Freeze the chosen thresholds in `configs/triage_thresholds.yaml` using validation only.

---

### 4. Required Output Artifacts
['- `src/controller/triage.py`', '- `tests/test_triage.py`', '- `results/day13_risk_coverage.csv`', '- `configs/triage_thresholds.yaml`']

---

### 5. Completion Check
> **Definition of Done for Day 13:**  
> Thresholds are chosen from held-out validation evidence and you can state the error rate you accept in exchange for the coverage you keep.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- - Risk-coverage and selective prediction
- - Calibration threshold selection

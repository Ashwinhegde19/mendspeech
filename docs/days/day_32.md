# Day 32: Optimization scorecard

> **Week 5 • Day 4 of 7**  
> **Navigation:** [← Day 31](day_31.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 33 →](day_33.md)

> **v3 STATUS: CORE** One table decides what ships. Techniques that did not help are reported with the same prominence.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Presenting negative results without overclaiming.
- Why an optimization table is a design document, not a log.

---

### 2. Build in MendSpeech
- Build the scorecard generator in `src/bench/scorecard.py`.
- Produce the scorecard in `results/day32_optimization_scorecard.csv`.

---

### 3. Experiment and Measure
- Tabulate WER, p50/p95/p99, RTF, and memory for baseline and every technique tried.
- Add a `what_did_not_help` section to `docs/day32_optimization_report.md`.
- Select the shipping configuration from the scorecard and justify it on measured grounds.

---

### 4. Required Output Artifacts
['- `src/bench/scorecard.py`', '- `results/day32_optimization_scorecard.csv`', '- `docs/day32_optimization_report.md`']

---

### 5. Completion Check
> **Definition of Done for Day 32:**  
> You can defend the shipping configuration from a table, including the techniques that failed.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Inference optimization case studies
- ONNX Runtime performance tuning

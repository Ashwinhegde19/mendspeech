# Day 32: Supported fixed-context lookahead ablation

> **Week 5 • Day 4 of 7**  
> **Navigation:** [← Day 31](day_31.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 33 →](day_33.md)

> **v2 STATUS: CORE — fixed context only.** Use only configurations verified for the Day 24 checkpoint. This session does not require live context switching.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Right context.
- Lookahead.
- Commit delay.
- WER and latency as competing objectives.

---

### 2. Build in MendSpeech
- Run the supported fixed-context configurations recorded in Day 24, using a separate run per setting and holding model, audio, hardware, batching, and decoding fixed.
- Store per-utterance and aggregate metrics, exact context values and units, and capability status. Do not coerce unsupported values or build a new streaming path.

---

### 3. Experiment and Measure
- Plot measured WER versus measured L4 latency and identify dominated operating points only when multiple supported settings exist.
- If only one setting is supported, retain its measured point and label the comparison unavailable; if none is runnable, record the reason and leave metrics missing. Keep the CSV and figure paths, with an explicitly annotated unavailable comparison rather than fabricated points or latency.

---

### 4. Required Output Artifacts
- `experiments/lookahead_ablation.py`
- `results/day32_lookahead.csv`
- `results/day32_pareto.png`

---

### 5. Completion Check
> **Definition of Done for Day 32:**  
> You can defend a supported fixed operating point using measured data, or show
> why the comparison is unavailable. A single point is not a Pareto frontier;
> unsupported context variation defers that claim, not the rest of Gate 4.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Stateful or cache aware Conformer primary material
- NVIDIA NeMo streaming ASR documentation and examples

# Day 25: Fit and validate calibrated confidence and triage

> **Week 4 • Day 4 of 7**
> **Navigation:** [← Day 24](day_24.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 26 →](day_26.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 11](day_11.md), [Day 14](day_14.md), [Day 18](day_18.md), [Day 24](day_24.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Temperature scaling, word/utterance correctness labels, reliability, Brier/ECE, risk-coverage.

---

### 2. Build in MendSpeech
- Implement src/asr/calibration.py with an explicit score-to-correctness unit, calibration-only fitting and separate validation thresholds.
- Map scores to actual decoded tokens/words; if beam scores lack valid aligned features, retain verified greedy triage rather than calling fused scores probabilities.
- Bind calibration to checkpoint/head/tokenizer/decoder/precision and context mode; build accept/uncertain/reject policy that preserves raw text and bypasses editor on uncertainty.

---

### 3. Experiment and Measure
- Compare raw/calibrated reliability and risk-coverage on disjoint validation, with bins/counts and corruption slices.
- Test empty/blank/all-correct/all-wrong and serialization/refit; non-improvement is valid but unsupported calibrated claims are not.

---

### 4. Required Output Artifacts
- `src/asr/calibration.py`
- `src/controller/triage.py`
- `tests/test_calibration.py`
- `tests/test_triage.py`
- `configs/calibration.yaml`
- `configs/triage_thresholds.yaml`
- `results/day25_reliability.csv`
- `results/day25_reliability.png`
- `results/day25_risk_coverage.csv`

---

### 5. Completion Check
> **Definition of Done for Day 25:**
> Calibration is actually fitted and independently evaluated, distinct from threshold selection, for the exact shipping-candidate configuration.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.

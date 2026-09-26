# Day 42: Week 6 robustness milestone

> **Week 6 • Day 7 of 7**  
> **Navigation:** [← Day 41](day_41.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 43 →](day_43.md)

> **v2 STATUS: CORE — Gate 5 is evidence-based.** Extend the one app; preserve
> explicit blocked precision status rather than claiming unmeasured optimization.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Review fine tuning, augmentation, RNNT, and calibration.

---

### 2. Build in MendSpeech
- Extend `app/audio_lab.py` to switch between base and adapted recognizer;
  retain the existing ASR/streaming/policy controls rather than create another app.
- Show clean WER, damaged WER, confidence calibration, and repair percentage.
- Show model/policy versions, raw versus calibrated confidence, validation-selected
  thresholds, safe action/reason codes and Day 40 measured/blocked precision
  status. Do not offer an unavailable export as if it were implemented.

---

### 3. Experiment and Measure
- Run one fixed benchmark suite and freeze results for Week 8 comparisons.
- Keep matched clean/raw-damaged controls and separate optional Indic extension
  results from the immutable core test set. Report regression, negative outcomes,
  inspect/abstain behavior, adaptation provenance and actual L4 configurations.

---

### 4. Required Output Artifacts
- `app/audio_lab.py`
- `results/week6_frozen_baseline.csv`
- `reports/week6_training.md`

---

### 5. Completion Check
> **Definition of Done for Day 42:**  
> The one app and report demonstrate measured adaptation and calibration results,
including clean regression or a negative result. Precision tradeoffs are supported
by controlled L4 measurements or explicitly blocked, never falsely completed.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- NVIDIA NeMo ASR training documentation
- RNNT primary references
- Calibration and reliability diagram references

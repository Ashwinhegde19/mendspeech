# Day 41: Confidence calibration for repair decisions

> **Week 6 • Day 6 of 7**  
> **Navigation:** [← Day 40](day_40.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 42 →](day_42.md)

> **v2 STATUS: CORE — calibration follows the selected model, decoder, and precision.** External LM search scores do not automatically become repair confidence.

---

### Compute Target
`Modal L4 for logits, local CPU for
analysis`

---

### 1. Learn
- Reliability diagrams.
- Expected calibration error intuition.
- Threshold selection from validation data.
- Decoder-dependent hypotheses, alignment, and acoustic versus LM score meaning.

---

### 2. Build in MendSpeech
- Build a simple calibration analysis for confidence versus correctness.
- Choose policy thresholds on validation, not test.
- Record model/head/tokenizer, precision, decoder configuration and LM revision
  from Days 24/26/40. Define and test how confidence attaches to the actual
  decoded words/spans; a fused beam score is not a probability, and greedy
  thresholds cannot silently transfer to LM-altered hypotheses. Validate token/
  timestamp alignment or retain the verified greedy path for repair.
- Refit/check calibration when model, precision or decoder changes. Preserve
  the exact score definition and fitting split in `configs/repair_modes_calibrated.yaml`.

---

### 3. Experiment and Measure
- Compare raw and calibrated confidence if a simple method is feasible.
- Evaluate correctness and reliability for the chosen configuration on held-out
  clean/damaged cases, including LM-induced errors if LM output enters repair.
  Keep validation-selected thresholds fixed; report failed calibration honestly.

---

### 4. Required Output Artifacts
- `src/asr/calibration.py`
- `results/day41_reliability.png`
- `configs/repair_modes_calibrated.yaml`

---

### 5. Completion Check
> **Definition of Done for Day 41:**  
> Repair thresholds are justified from held-out evidence for the actual
> model/decoder/precision, with tested hypothesis alignment and an explicit
> confidence definition. Search scores are not relabeled calibrated confidence.

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

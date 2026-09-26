# Day 11: Token confidence and where it fails

> **Week 2 • Day 4 of 7**  
> **Navigation:** [← Day 10](day_10.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 12 →](day_12.md)

> **v3 STATUS: CORE — confidence is a signal, not a truth.** This session exists to find the cases where confidence is confidently wrong.
---

### Compute Target
`Local CPU`

---

### 1. Learn
- Frame softmax probability versus token confidence.
- Why mean confidence hides per-token failures.
- Confident-but-wrong: the failure mode that breaks a confidence-gated system.

---

### 2. Build in MendSpeech
- Extract per-token confidence and align it to emitted tokens in `src/asr/confidence.py`.
- Build a word-level confidence timeline aligned to the transcript in `src/asr/confidence.py`.

---

### 3. Experiment and Measure
- Compare confidence across clean, noisy, clipped, and dropout audio.
- Collect at least ten confident-but-wrong examples and write them up in `docs/day11_confident_wrong.md`.
- Record the rate at which a fixed confidence threshold would have accepted a wrong token.

---

### 4. Required Output Artifacts
['- `src/asr/confidence.py`', '- `tests/test_confidence.py`', '- `results/day11_confidence_by_damage.csv`', '- `docs/day11_confident_wrong.md`']

---

### 5. Completion Check
> **Definition of Done for Day 11:**  
> You can state when low confidence is informative, and you have documented concrete cases where high confidence was wrong.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- - Confidence calibration background
- - Selective prediction and risk-coverage curves
- - CTC decoding confidence literature

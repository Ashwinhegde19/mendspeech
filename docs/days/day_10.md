# Day 10: WER, CER, and error taxonomy

> **Week 2 • Day 3 of 7**  
> **Navigation:** [← Day 09](day_09.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 11 →](day_11.md)

> **v3 STATUS: CORE — recognition quality.** This is the accuracy axis every later trade-off is measured against.
---

### Compute Target
`Local CPU`

---

### 1. Learn
- Word error rate: substitutions, deletions, insertions.
- Character error rate and when it helps.
- Why WER alone hides error severity.
- Names and numbers as a separate error class.

---

### 2. Build in MendSpeech
- Implement or verify WER and CER calculations in `src/metrics/wer.py`.
- Add an error analyzer labelling substitution, deletion, and insertion spans in `src/metrics/wer.py`.
- Add a names-and-numbers extractor so entity errors are counted separately in `src/metrics/wer.py`.

---

### 3. Experiment and Measure
- Score clean audio versus every SpeechDamageBench severity.
- Find which corruption type drives deletion errors fastest.
- Write tests for empty references, identical strings, and empty hypotheses in `tests/test_wer.py`.

---

### 4. Required Output Artifacts
['- `src/metrics/wer.py`', '- `tests/test_wer.py`', '- `results/day10_wer_by_damage.csv`', '- `results/day10_error_types.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 10:**  
> You can compute WER by hand for a short example and explain each error class, and entity errors are reported separately from the blended rate.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- - NIST SHERE scoring conventions
- - jiwer or equivalent reference behaviour
- - LibriSpeech reference transcript normalization

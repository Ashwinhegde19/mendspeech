# Day 12: Time alignment and word timestamps

> **Week 2 • Day 5 of 7**  
> **Navigation:** [← Day 11](day_11.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 13 →](day_13.md)

> **v3 STATUS: CORE — timestamps for latency attribution and interface feedback.** Word timing is what lets a streaming UI and a latency report say where time went.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Frame index to wall-clock mapping.
- Token timestamps versus forced alignment.
- Why timestamps must be validated before they are trusted downstream.

---

### 2. Build in MendSpeech
- Map emitted tokens to audio time spans in `src/asr/timestamps.py`.
- Validate timestamps against a synthetic event at a known offset.

---

### 3. Experiment and Measure
- Inject dropouts at known offsets and check that surrounding token boundaries stay stable.
- Report timestamp error in milliseconds per corruption type in `results/day12_timestamp_error.csv`.
- Confirm that a decoding change does not silently shift timestamps.

---

### 4. Required Output Artifacts
['- `src/asr/timestamps.py`', '- `tests/test_timestamps.py`', '- `results/day12_timestamp_error.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 12:**  
> Token timestamps are accurate to a stated millisecond tolerance and survive a decoding change, or the failure is documented.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- - Torchaudio forced alignment utilities
- - CTC timestamp estimation references

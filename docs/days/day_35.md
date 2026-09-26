# Day 35: Endpointing and the VAD baseline

> **Week 5 • Day 7 of 7**  
> **Navigation:** [← Day 34](day_34.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 36 →](day_36.md)

> **v3 STATUS: CORE** Add-on A, absorbed here. Endpointing decides when a final answer is sent, so its errors are latency errors.
---

### Compute Target
`Local CPU plus Modal L4`

---

### 1. Learn
- Energy versus spectral VAD.
- Onset and offset error in milliseconds.
- False alarms versus missed speech in a streaming setting.

---

### 2. Build in MendSpeech
- Implement a deterministic frame-level VAD in `src/vad/baseline.py` with framing, timestamp, and silence tests.
- Compare it with one local reference VAD on identical clean and damaged inputs.

---

### 3. Experiment and Measure
- Measure precision, recall, F1, false alarms, missed speech, and onset/offset error in ms.
- Report CPU RTF separately from GPU measurements.
- Carry the measured choice into Day 41 and record it in `results/day35_vad_benchmark.csv`.

---

### 4. Required Output Artifacts
['- `src/vad/baseline.py`', '- `tests/test_vad.py`', '- `results/day35_vad_benchmark.csv`', '- `docs/day35_vad_notes.md`']

---

### 5. Completion Check
> **Definition of Done for Day 35:**  
> Endpointing error is quantified in milliseconds and the chosen detector's failure modes are documented.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- WebRTC VAD
- Energy and spectral VAD references

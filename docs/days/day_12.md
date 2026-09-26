# Day 12: Time alignment and uncertain spans

> **Week 2 • Day 5 of 7**  
> **Navigation:** [← Day 11](day_11.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 13 →](day_13.md)

> **v2 STATUS: CORE.** Reuse the uncertainty overlay inside the single application.

---

### Compute Target
`Modal L4 recommended`

---

### 1. Learn
- Frame time conversion.
- Token timestamps and word timestamps.
- Alignment boundaries around corrupted regions.

---

### 2. Build in MendSpeech
- Map low confidence tokens back to audio time spans.
- Overlay uncertain intervals on waveform and spectrogram.
- Keep `app/uncertainty_overlay.py` as a reusable visualization module imported
  by `app/audio_lab.py`, not a separately maintained runnable application.

---

### 3. Experiment and Measure
- Inject known 100 ms and 250 ms dropouts and test whether uncertainty overlaps them.

---

### 4. Required Output Artifacts
- `src/asr/alignment.py`
- `app/uncertainty_overlay.py` (reusable module for `app/audio_lab.py`)
- `results/day12_overlap_metrics.csv`

---

### 5. Completion Check
> **Definition of Done for Day 12:**  
> The shared UI can highlight an uncertain audio interval and show the associated
word or token without creating another application.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- CTC primary paper or a reliable derivation
- Framework ASR documentation for logits, timestamps, and confidence

# Day 44: Selected-stack duration and prosody

> **Week 7 • Day 2 of 7**  
> **Navigation:** [← Day 43](day_43.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 45 →](day_45.md)

> **v2 STATUS: CORE — reuse Day 43's selected stack.** FastSpeech 2 is theory only, not a second installation.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Duration prediction.
- Pitch and energy predictors.
- Parallel generation intuition.

---

### 2. Build in MendSpeech
- Contrast FastSpeech 2 duration/pitch/energy predictors with the selected
  stack's generation path in `docs/day44_fastspeech2.md` (theory artifact).
- Reuse `src/tts/baseline.py`; inspect only controls actually exposed by the
  pinned revision. Do not invent native duration or pitch controls.
- Measure duration, pitch/energy summaries, and punctuation effects on fixed
  sentences and fixed consented speaker embeddings. A bounded post-synthesis
  duration adjustment must be labeled DSP, not learned prosody control.

---

### 3. Experiment and Measure
- Compare generated length against target intervals. If native rate control
  is unavailable, record `unsupported` and measure punctuation or DSP effects
  instead. Store waveforms ignored; commit only sample manifests/measurements.

---

### 4. Required Output Artifacts
- `docs/day44_fastspeech2.md`
- `results/day44_prosody_samples/`

---

### 5. Completion Check
> **Definition of Done for Day 44:**  
> You can explain short-span timing constraints using measured selected-stack
> behavior, distinguish native controls from DSP, and label unsupported controls.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- FastSpeech 2 paper
- HiFi GAN paper
- VITS paper
- DSP references for energy matching and equal power crossfades

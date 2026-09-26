# Day 45: Vocoder realism and acoustic boundary diagnostics

> **Week 7 • Day 3 of 7**  
> **Navigation:** [← Day 44](day_44.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 46 →](day_46.md)

> **v2 STATUS: CORE — boundary diagnostics and the selected stack's vocoder only.** No separate vocoder installation or training.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Mel to waveform generation.
- HiFi GAN style generator and discriminator intuition.
- Phase, bandwidth, and vocoder artifacts.
- Short time energy, local loudness, spectral balance, and room tone as boundary signals.

---

### 2. Build in MendSpeech
- Reuse only Day 43's selected stack's matched pretrained vocoder; keep
  weights frozen. GAN anatomy is theory, not a separate training experiment.
- Add boundary diagnostics that measure short time energy and simple spectral statistics before and after a candidate repair span.
- Save a local room tone estimate where possible.

---

### 3. Experiment and Measure
- Measure inference speed and real time factor on L4 with fixed batch size,
  warm-up, and sample rate. Distinguish isolated vocoder from end-to-end time;
  mark isolated timing unavailable if the interface does not expose it.
- Create intentionally mismatched generated spans and verify that the boundary diagnostics flag obvious loudness or spectral discontinuities.
- Include unchanged/identity stitch controls and tests for sample-count and
  outside-span preservation. Record both flagged and missed seam artifacts.

---

### 4. Required Output Artifacts
- `results/day45_vocoder_benchmark.csv`
- `src/repair/boundary_metrics.py`
- `docs/vocoder_and_boundary_notes.md`

---

### 5. Completion Check
> **Definition of Done for Day 45:**  
> You can separate acoustic model errors from vocoder artifacts and quantify at least
two causes of an audible seam.

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

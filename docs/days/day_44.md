# Day 44: Selected-stack duration and prosody

> **Week 7 • Day 2 of 7**  
> **Navigation:** [← Day 43](day_43.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 45 →](day_45.md)

> **v2 STATUS: CORE — repair-focused prosody in Day 43's selected stack.** Evaluate both language slices; FastSpeech 2 is theory only, not a second installation or an emotion-control claim.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Duration prediction.
- Pitch and energy predictors.
- Parallel generation intuition.
- Voiced/unvoiced pitch masking, duration error in seconds, energy continuity
  at a repair boundary, and listener assessment versus objective proxies.

---

### 2. Build in MendSpeech
- Contrast FastSpeech 2 duration/pitch/energy predictors with the selected
  stack's generation path in `docs/day44_fastspeech2.md` (theory artifact).
- Reuse `src/tts/baseline.py`; inspect only controls actually exposed by the
  pinned revision. Do not invent native duration or pitch controls.
- Measure duration, pitch/energy summaries, and punctuation effects on fixed
  sentences and fixed consented speaker embeddings. A bounded post-synthesis
  duration adjustment must be labeled DSP, not learned prosody control.
- Use Day 43's held-out manifest in both languages. Compare baseline to one
  supported native control, or one bounded DSP adjustment if native control is
  unavailable, holding text, speaker reference, seed and output format fixed.
  Use only permitted surrounding context for timing/style targets, not the
  missing clean target span. Any privileged target access is a separate oracle.
- Freeze control limits on validation material, never the evaluation sentences.
  Test duration/sample-count and finite-metric handling; report pitch only for
  valid voiced frames, with coverage and unvoiced/failed estimates marked
  missing rather than forced to zero. Add these checks alongside the public
  implementation when built, in `tests/test_tts_prosody.py`.

---

### 3. Experiment and Measure
- Compare generated length against target intervals. If native rate control
  is unavailable, record `unsupported` and measure punctuation or DSP effects
  instead. Store waveforms ignored; commit only sample manifests/measurements.
- In `results/day44_prosody_metrics.csv`, record per-language duration error,
  voiced pitch mismatch, energy discontinuity at context boundaries, and
  pronunciation/intelligibility observations for base and controlled conditions.
  Keep text/punctuation changes in separate rows from fixed-text control tests.
- Run a small randomized/blinded base-versus-controlled listening check with
  competent language review. Record item/rater counts, order seed, naturalness
  and intelligibility judgments, and single-rater limits where applicable in
  `results/day44_listening_sheet.md`. Improved seams are not guaranteed; reject
  a nicer-sounding timing change that materially damages intelligibility.

---

### 4. Required Output Artifacts
- `docs/day44_fastspeech2.md`
- `results/day44_prosody_samples/`
- `results/day44_prosody_metrics.csv`
- `results/day44_listening_sheet.md`
- `tests/test_tts_prosody.py`

---

### 5. Completion Check
> **Definition of Done for Day 44:**  
> You can explain short-span timing constraints using measured selected-stack
> behavior, distinguish native controls from DSP, and label unsupported controls.
> Both language slices have fixed-condition prosody and listening evidence with
> units, voiced coverage, target provenance and limitations. Missing native pitch
> control is not learned emotion control; no positive result is required.

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

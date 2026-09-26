# Day 56: Final product, demo, and clean reproduction

> **Week 8 • Day 7 of 7**  
> **Navigation:** [← Day 55](day_55.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Summary →](../INDEX.md)

> **v2 STATUS: CORE — absorbs Day 55.** Gate 7 closes on report, artifact, and reproduction evidence, not a date or guaranteed session count.

---

### Compute Target
`Modal L4 for inference, local CPU for
interface and analysis`

---

### 1. Learn
- Review the complete path from waveform and controlled corruption to streaming encoder, uncertainty, repair policy, cascaded reconstruction, direct audio baseline, and evaluation.

---

### 2. Build in MendSpeech
- Extend only `app/audio_lab.py` with upload or consented microphone input,
  controlled damage, transcript, uncertainty heatmap, Preserve / Inspect /
  Repair / Abstain, before/after playback, and measured metrics. Reuse Day 49
  abstention; do not create a separate final or voice-agent app.
- Label live input/control, prerecorded benchmark playback, simulated context,
  and oracle diagnostics distinctly. Expose only supported external conditions
  in benchmark playback; show unavailable comparator/inpainting as deferred.
- Show measured changed/preserved samples, including crossfade margins. Do not
  claim a full-waveform restoration model preserved everything outside a mask.
- Reproduce one frozen benchmark from a fresh environment and tag a stable release.

---

### 3. Experiment and Measure
- Record a concise demo and create a final architecture diagram.
- Reproduce one benchmark end to end from the documented command.
- Verify that every public chart can be regenerated from saved result files.

---

### 4. Required Output Artifacts
- `app/audio_lab.py`
- `README.md`
- `demos/final_demo.mp4`
- `docs/architecture.png`
- `release_notes.md`
- `results/reproduction_check.txt`

---

### 5. Completion Check
> **Definition of Done for Day 56:**  
> A new user can reproduce MendSpeech and SpeechDamageBench, evaluate the
> internal baselines and any supported external comparison, and distinguish
> measured results from unsupported/deferred capabilities. The one app and
> technical report agree on abstention, consent, and live/simulated labels.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Your frozen protocol and prior results
- The one comparator's verified support record or explicit deferral
- Primary papers only when needed to interpret a result

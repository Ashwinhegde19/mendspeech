# Day 49: Week 7 MendSpeech V1 cascaded repair milestone

> **Week 7 • Day 7 of 7**  
> **Navigation:** [← Day 48](day_48.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 50 →](day_50.md)

> **v2 STATUS: CORE — Gate 6 is evidence-based.** Implement abstention now and extend the single `app/audio_lab.py`; no separate voice-agent project.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Review TTS, duration, vocoder behavior, speaker conditioning, boundary matching, and information lost through the text bottleneck.
- Treat the cascaded path as a measured baseline, not a guaranteed real-time
  system. Label live versus simulated streaming/context control explicitly.

---

### 2. Build in MendSpeech
- Pipeline: damaged audio to streaming ASR to uncertain span to policy decision to speaker conditioned reconstruction to boundary matched waveform.
- Implement `src/controller/abstain.py` before this milestone, not on Day 54.
  Abstain when content evidence is insufficient, speaker use is unauthorized,
  or duration/boundary constraints cannot be met; preserve original audio and
  return a reason code. Use validation-set thresholds, never test-tuned ones.
- Extend only `app/audio_lab.py` for Preserve / Inspect / Repair / Abstain
  decisions, predicted-text reconstruction, and consent/capability status.
- Show preserved and reconstructed intervals with distinct visualization.
- Add a V1 label in results so the Week 8 direct audio repair comparison is explicit.

---

### 3. Experiment and Measure
- Run at least ten cases, including deliberate false repair, missed repair, seam artifacts, and one case where the policy abstains.
- Compare naive stitching and boundary matched stitching on the same repaired spans.
- Test low-evidence and permission-blocked abstention, clean no-repair cases,
  and unchanged samples outside declared edit/crossfade bounds. Keep oracle
  diagnostics separate and accept measured null/worse seam outcomes.

---

### 4. Required Output Artifacts
- `app/audio_lab.py`
- `src/controller/abstain.py`
- `demos/week7_before_after/`
- `results/week7_stitching_ablation.csv`
- `reports/week7_cascaded_repair.md`

---

### 5. Completion Check
> **Definition of Done for Day 49:**  
> MendSpeech V1 has tested abstention in the one app and measured predicted-text
> repair/seam evidence. Strengths, failures, deferred adaptation, and live versus
> simulated execution are documented; there is no deadline-based completion.

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

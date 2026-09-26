# Day 47: Speaker representation and preservation

> **Week 7 • Day 5 of 7**  
> **Navigation:** [← Day 46](day_46.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 48 →](day_48.md)

> **v2 STATUS: CORE — consented conditioning within the selected stack.** No second TTS installation.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Speaker embeddings.
- Reference conditioned synthesis.
- Speaker similarity as a measurable but imperfect proxy.
- Consent and voice identity boundaries.

---

### 2. Build in MendSpeech
- Reuse the selected stack's verified speaker-conditioning path and Day 43 provenance
  checks. Use only owned or explicitly consented references, separate from
  held-out target recordings; do not derive conditioning from a clean test
  reference unavailable at inference time.
- Compute speaker embeddings before and after synthesis if supported by the
  pinned tooling. Mark unavailable proxies `unsupported`; do not add another
  synthesis stack or infer identity preservation from naturalness alone.
- Record permitted voice uses, conditioning access, and limitations in
  `docs/voice_use_policy.md`; abstain when consent or required conditioning
  is missing. Use the base checkpoint if adaptation was deferred.

---

### 3. Experiment and Measure
- Compare full resynthesis with short span reconstruction for speaker similarity.

---

### 4. Required Output Artifacts
- `src/tts/speaker_conditioning.py`
- `results/day47_speaker_similarity.csv`
- `docs/voice_use_policy.md`

---

### 5. Completion Check
> **Definition of Done for Day 47:**  
> You can discuss speaker similarity measurements and their limitations without
claiming identity preservation from listening alone.

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

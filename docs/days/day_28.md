# Day 28: Week 4 integration

> **Week 4 • Day 7 of 7**  
> **Navigation:** [← Day 27](day_27.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 29 →](day_29.md)

> **v2 STATUS: CORE — absorbs Day 27.** Integration plus the top-3 failure casebook in one session, using the shared `app/audio_lab.py` entrypoint.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Review efficiency choices and baseline results.

---

### 2. Build in MendSpeech
- Replace the generic ASR runner in MendSpeech with the reproducible FastConformer path.
- Extend `app/audio_lab.py`, the single app entrypoint, to expose latency, RTF, WER when reference text exists, and GPU memory. Do not create a versioned demo app.
- Capture Day 27's top three repeatable failure patterns with transcript, confidence, and damage metadata in the retained casebook.
- Link Day 26's decoding comparison with head/tokenizer/LM/config provenance
  in `reports/week4_fastconformer.md`. Show offline beam/LM output only as
  offline evidence. Retain a verified streaming-compatible decoder (greedy if
  needed); do not replace it with an offline-only backend or treat LM-fused
  scores as confidence. Changes to text/timestamps require alignment checks
  and Day 41 decoder-specific calibration before calibrated repair claims.

---

### 3. Experiment and Measure
- Run the same ten reference clips through the full Week 2 uncertainty policy using FastConformer.
- Record the actual decoder for every run. Include helpful/harmful Day 26
  examples; preserve the original greedy results instead of overwriting them.

---

### 4. Required Output Artifacts
- `app/audio_lab.py`
- `results/fastconformer_failure_casebook.md` — absorbed Day 27 evidence
- `reports/week4_fastconformer.md`

---

### 5. Completion Check
> **Definition of Done for Day 28:**  
> The shared audio lab exposes a measured, inspectable FastConformer recognition
> core, and the report links the top-three failure casebook and completed
> three-way decoding evidence. Offline and streaming capabilities, score
> semantics and pending calibration are explicit; blocked decoding needs scope
> review rather than a completed Gate 3 label.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- FastConformer primary paper
- NVIDIA NeMo FastConformer model documentation

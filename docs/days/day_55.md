# Day 55: Write the research report and reproducibility guide

> **Week 8 • Day 6 of 7**  
> **Navigation:** [← Day 54](day_54.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 56 →](day_56.md)

> **v2 STATUS: MERGED into [Day 56](day_56.md).** Write the technical report alongside the single `app/audio_lab.py` demo; completion is evidence-based.

---

### Compute Target
`Local CPU`

---

### 1. Learn
- Abstract, motivation, hypotheses, method, baselines, metrics, results, limitations, ethics, and future work.
- Difference between observation and causal claim.
- How to report a negative or mixed architectural comparison honestly.
- Benchmark scale and its statistical limits: never claim population-level generalization from a ~5-speaker lab set.

---

### 2. Build in MendSpeech
- Write the complete report.
- Add exact reproduction commands and environment capture.
- Include a dedicated internal-versus-external comparison section with the
  one comparator's selected, supported, measured, unsupported, and deferred
  conditions. If it could not run, report internal comparisons and explicit
  external/inpainting deferral, not an invented architectural result.
- Document seam limitations, prosody loss, consent, abstention, and supported
  conditions where either method is stronger, unchanged, or worse.
- State Day 46's base/adapted evidence or training deferral, gold-text/oracle
  exclusions, and live versus simulated context labels. Simulation cannot
  establish measured runtime gains; blocked training is not measured adaptation.
- Trace greedy/beam/LM claims to Day 26, including harmful changes and separate
  decoder-only/fresh end-to-end timings. Trace each TTS language and prosody
  claim to Days 43–46, with held-out counts, reviewer/listener limits, and
  regression results rather than a pooled score that hides one language.
- State whether TTS is native streaming, phrase-chunked, or full-waveform
  delivery, whether full text is required up front, and what must finish before
  repaired audio can play. Native streaming, code-mixed synthesis, and emotion
  control are not implied by multilingual output or network chunking. Include
  Day 49 complete-repair timing separately from Day 45 synthesis latency.
- Reproduce the existing `app/audio_lab.py`; do not introduce a second app.
- Include plots with captions that state what changed and what stayed fixed.

---

### 3. Experiment and Measure
- Audit every major claim against a concrete table, figure, or experiment result.
- Remove or soften any conclusion that is not directly supported by frozen evidence.
- Verify that the report distinguishes measured facts from hypotheses and future work.

---

### 4. Required Output Artifacts
- `REPORT.md`
- `REPRODUCE.md`
- `results/final_figures/`
- `docs/limitations_and_claims.md`

---

### 5. Completion Check
> **Definition of Done for Day 55:**  
> A technical reader can understand the contribution, the architectural tradeoff, and the
limitations without opening the source code first.

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

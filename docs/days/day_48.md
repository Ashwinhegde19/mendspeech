# Day 48: Selective reconstruction with boundary matched stitching

> **Week 7 • Day 6 of 7**  
> **Navigation:** [← Day 47](day_47.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 49 →](day_49.md)

> **v2 STATUS: CORE — predicted-text selective repair with measured seam outcomes.** Oracle text is a separate diagnostic, never the normal path.

---

### Compute Target
`Modal L4 plus local CPU for stitching`

---

### 1. Learn
- Repair span text selection.
- Timing constraints and duration control.
- Boundary padding and silence handling.
- Short time energy matching and local loudness matching.
- Linear versus equal power crossfades.
- Spectral and room tone mismatch.
- Why ASR to text to TTS can lose pitch, emotion, breathing, and coarticulation.

---

### 2. Build in MendSpeech
- For normal runs, use the controller-selected interval and predicted ASR
  text, not the gold/reference transcript. Reuse the selected TTS stack;
  reject unsafe spans when inferred content or speaker permissions are weak.
- Gold text or known damage boundaries may be used only in separately labeled
  `oracle_text` / `oracle_span` diagnostics. Record text source and span source
  independently and exclude oracle rows from end-to-end performance claims.
- Match generated duration to the target interval without changing untouched speech.
- Match local energy before stitching and implement both linear and equal power crossfades.
- Add optional room tone under the regenerated span when the original context supports it.
- Log preserved samples, reconstructed samples, boundary length, and all matching parameters.
- Test identity/no-repair behavior, exact sample counts, and unchanged samples
  outside the target interval plus explicitly declared crossfade margins.

---

### 3. Experiment and Measure
- Compare full utterance TTS, naive selective repair, and boundary matched selective repair.
- Measure preservation percentage, latency, energy discontinuity, and speaker similarity proxy.
- Run a small blinded seam audibility check with randomized sample order.
- Hold predicted text and intervals fixed for stitching comparisons. Report
  smoother, unchanged, or worse seams; do not select cases to force an improvement.

---

### 4. Required Output Artifacts
- `src/repair/reconstruct.py`
- `src/repair/stitch.py`
- `src/repair/boundary_metrics.py`
- `results/day48_selective_samples/`
- `results/day48_seam_ablation.csv`

---

### 5. Completion Check
> **Definition of Done for Day 48:**  
> Predicted-text runs preserve samples outside declared repair/crossfade bounds,
> and seam metrics plus blinded checks compare matched and naive stitching on
> identical spans. Measured non-improvement is valid; oracle-only performance
> cannot satisfy the normal end-to-end check.

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

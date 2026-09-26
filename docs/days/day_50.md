# Day 50: Freeze research questions and baselines

> **Week 8 • Day 1 of 7**  
> **Navigation:** [← Day 49](day_49.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 51 →](day_51.md)

> **v2 STATUS: CORE — freeze evidence and capability limits, not a calendar.** One external restoration comparator at most; consume Week 2's bounded feasibility decision.

---

### Compute Target
`Local CPU for planning, Modal L4 for
dry run`

---

### 1. Learn
- Primary question: can selective semantic repair improve intelligibility while preserving more original speech than full resynthesis?
- Secondary question: can uncertainty guided context allocation improve the latency versus accuracy operating point?
- Architecture question: on supported conditions, how does cascaded ASR plus
  TTS compare with the one selected pretrained direct restoration comparator?
  Denoising/enhancement is not evidence of mask-aware missing-span inpainting.
- Scope every claim to the frozen benchmark scale (≥30 utterances, ≥5
  speakers, typically ~5 at this lab) and state the statistical caveat
  explicitly — do not claim population-level generalization.
- Define null outcomes, failure criteria, and claims you will not make.

---

### 2. Build in MendSpeech
- Freeze code revision, model revisions, datasets, hardware, corruption configs, and metrics.
- Freeze raw damaged audio, full resynthesis, naive selective repair, and
  boundary-matched selective repair, with predicted text as the normal path.
  Hold text/spans fixed for stitching comparisons; segregate oracle rows.
- Reuse `docs/baseline_install_notes.md` from Week 2: record selected checkpoint,
  revision/license, feasible/deferred state, supported corruptions, mask
  capability, resampling, and preservation semantics. Implement only one
  adapter, `src/baselines/direct_audio_restore.py`, if feasible; selection
  alone is not tested support. Do not assume mask input or inpainting ability.
- If unavailable, freeze the four internal comparisons above and explicitly
  defer external restoration/inpainting. No model hunting, second comparator,
  scratch-restoration fallback, or claim that the external method was tested.
- Freeze fixed/adaptive context conditions with `execution_mode=live` or
  `simulated`. Only implemented live control with same-L4 measurements can
  support runtime-gain claims; cached/oracle scheduling is not deployed speedup.
- Keep Day 49 abstention active, and record whether the TTS checkpoint is base
  or adapted plus Day 46's measured/deferred status. No required positive result.

---

### 3. Experiment and Measure
- Run a tiny dry run to verify every required field has a measurement or
  explicit status/reason. Unsupported/deferred conditions have missing metrics,
  not fabricated zeros; they are excluded from measured rankings and plots.

---

### 4. Required Output Artifacts
- `experiments/capstone_protocol.md`
- `configs/capstone_frozen.yaml`
- `docs/baseline_definitions.md`

---

### 5. Completion Check
> **Definition of Done for Day 50:**  
> Another engineer can reproduce the supported comparisons and distinguish
> selected from tested support, external/inpainting deferral, oracle diagnostics,
> and live versus simulated context results without inventing missing evidence.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Your frozen protocol and prior results
- The one Week 2 restoration comparator's capability and feasibility record
- Primary papers only when needed to interpret a result

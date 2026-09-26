# Day 46: Bounded TTS adaptation — base versus adapted

> **Week 7 • Day 4 of 7**  
> **Navigation:** [← Day 45](day_45.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 47 →](day_47.md)

> **v2 STATUS: CORE — conditional adaptation of the one selected TTS stack.** Training is permitted only after Day 43's feasibility gate; blocked training is `deferred`, not measured adaptation.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Small-data adaptation, frozen versus trainable parameters, overfitting,
  and held-out sentence evaluation with consented speaker conditioning.
- VITS latent variables, flows, and adversarial waveform synthesis may be
  compared theoretically; do not install or run a second TTS model.

---

### 2. Build in MendSpeech
- Consume `docs/tts_pipeline.md` without reopening stack selection. If its
  adaptation gate is `feasible`, implement one bounded selected-stack
  fine-tune in `training/tts_finetune.py` using `configs/tts_finetune.yaml`.
  Use only the supported adaptation method frozen at Day 43; assert trainable
  names/counts, frozen components, and finite gradients match its feasibility
  record. The existing pretrained vocoder remains frozen.
- Freeze seed, base revision, optimizer, batch size, precision, learning rate,
  data/split hashes, maximum steps, wall time, and L4 spend before the run.
  Stop at the first limit; no sweep, scratch training, or second project.
- Use only legally permitted paired data and consented speaker references.
  Hold out sentences and source recordings before training, check duplicate
  text/audio and speaker leakage, and keep frozen benchmark speakers/audio/
  transcripts out of training and tuning. Match speaker conditions across
  base/adapted outputs; do not claim unseen-speaker transfer from same-speaker
  held-out sentences.
- Keep checkpoints, generated audio, and run logs in ignored storage. Track
  only code/config, provenance hashes, measured summaries, and the report.
- If the gate or run is blocked, record `deferred` with the reason in
  `docs/tts_pipeline.md` and the comparison/listening artifacts. Do not create
  placeholder training artifacts or claim adaptation was executed. Retain
  the usable base model for repair; if base inference is blocked, defer it too.

---

### 3. Experiment and Measure
- Compare the frozen base and one adapted checkpoint on identical held-out
  sentences, speaker embeddings, generation settings, and L4 hardware.
  Measure intelligibility proxy, duration error, speaker proxy when supported,
  inference latency/RTF, trainable count, training time, memory, and actual cost.
- Randomize base/adapted sample order for a small listening check; report the
  number of raters/items and limitations. Keep test results out of selection.
- Record improvement, no meaningful change, or degradation as measured
  outcomes. Non-improvement is valid; incomplete or blocked training is not
  a negative result and must remain `deferred` with missing metrics, not zeros.

---

### 4. Required Output Artifacts
- Feasible branch only: `training/tts_finetune.py`
- Feasible branch only: `configs/tts_finetune.yaml`
- Feasible branch only: `reports/day46_tts_adaptation.md` (including failures
  after starting; never claim a completed comparison if the run was blocked)
- `results/day46_tts_comparison.csv`
- `results/day46_listening_sheet.md`
- Update `docs/tts_pipeline.md` with the final measured/deferred status. On
  the deferred branch, the retained comparison/listening paths contain only
  available base evidence and explicit unavailable adapted-condition status.

---

### 5. Completion Check
> **Definition of Done for Day 46:**  
> Either one bounded base-versus-adapted experiment has reproducible held-out
> evidence (including a valid null or worse result), or training is explicitly
> deferred with its blocking evidence. Feasibility-only work does not satisfy
> training completion and cannot be described as measured adaptation.

---

### 6. Study Method & Protocol
Use the predeclared feasibility and run limits, not an assumed single-session
training budget. Stop and label incomplete work honestly; do not expand the
experiment to protect a calendar target.

---

### 7. References & Resources
- FastSpeech 2 paper
- HiFi GAN paper
- VITS paper
- DSP references for energy matching and equal power crossfades

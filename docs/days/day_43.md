# Day 43: TTS system anatomy

> **Week 7 • Day 1 of 7**  
> **Navigation:** [← Day 42](day_42.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 44 →](day_44.md)

> **v2 STATUS: CORE — one TTS stack and a bounded adaptation feasibility gate.** No second synthesis installation; completion follows evidence, not a date.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Text or phoneme representation.
- Acoustic model.
- Mel spectrogram or latent representation.
- Vocoder.
- Speaker conditioning.
- Prosody.
- Content, speaker, and style representations; why useful factorization is not
  proof of perfect disentanglement.

---

### 2. Build in MendSpeech
- Select exactly one feasible, permitted TTS stack at this gate and reuse it
  throughout Week 7, including its existing pretrained vocoder. Check the
  planned stack's speaker/language, data, adaptation, and compute requirements
  before selection; do not prescribe an unverified new model/framework or
  install alternatives. A documented recipe is not measured L4 feasibility.
- In `docs/tts_pipeline.md`, pin checkpoint and processor/tokenizer revisions,
  library versions, sample rate, text normalization and token coverage, and
  speaker-embedding shape/provenance. Use only owned or explicitly consented
  speaker references; public availability alone is not consent.
- Record legal paired training-data provenance and permitted uses, duration,
  transcript quality, speaker/reference IDs, and disjoint train/validation/
  held-out sentence splits. Exclude frozen evaluation audio, transcripts, and
  speakers from training and tuning; no duplicate text/audio leakage.
- Choose and justify one supported bounded adaptation method for that stack;
  record its exact trainable parameter names/counts and frozen components.
  Keep the vocoder frozen and verify finite gradient flow on the selected
  revision instead of assuming an adapter API exists.
- Before Day 46, declare step, wall-time, data-duration, and spend ceilings
  from remaining resources. Run at most one small L4 forward/backward pilot;
  record batch size, precision, peak memory, seconds/step, current L4 price,
  estimated capped cost, and stop reason. A failed install, permissions/data
  gap, invalid gradients, or budget overrun means `adaptation_status=deferred`.
  Only a supported pilot within the declared bounds means `feasible`.
- Record inference feasibility separately. Do not model-hunt, train from
  scratch, add a second project, or promise a session/compute budget; if
  inference is blocked, dependent synthesis work remains deferred.
- Save generated waveforms locally in ignored storage and exposed intermediate
  representations; tracked sample directories contain only manifests/notes.
- Record where the selected system injects linguistic content, speaker
  identity, and style or prosody conditioning.

---

### 3. Experiment and Measure
- Compare several sentences with punctuation and pacing changes.
- FastSpeech 2 and VITS are short theoretical contrasts, not additional models
  to install or benchmark. Record unresolved capabilities explicitly.

---

### 4. Required Output Artifacts
- `src/tts/baseline.py`
- `results/day43_tts_samples/`
- `docs/tts_pipeline.md`

---

### 5. Completion Check
> **Definition of Done for Day 43:**  
> You can explain the selected text-to-waveform path and speaker conditioning,
> and `docs/tts_pipeline.md` records checked licenses, data/split provenance,
> exact trainable parameters, L4 pilot evidence or a blocking reason, cost
> bounds, and a feasible or deferred adaptation decision. No training success
> is claimed by this gate; unresolved fields remain explicitly unverified.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- The selected stack's model card, license, and supported adaptation recipe
- FastSpeech 2 paper
- HiFi GAN paper
- VITS paper
- DSP references for energy matching and equal power crossfades

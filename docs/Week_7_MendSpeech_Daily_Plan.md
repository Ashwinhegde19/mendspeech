# Week 7: TTS, Speaker Preservation, and Boundary Matched Reconstruction

> **Days 43 to 49**  
> **Navigation:** [← Week 6](Week_6_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Week 8 →](Week_8_MendSpeech_Daily_Plan.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Build MendSpeech V1 as a cascaded selective repair baseline with explicit seam diagnostics.
>
> **v2 evidence gate:** Gate 6 requires a measured predicted-text repair path,
> tested abstention, and honest seam/feasibility outcomes in one `app/audio_lab.py`.
> Use one feasible permitted TTS stack; adaptation is conditional, with no promised training
> session or compute budget. No separate voice-agent project.

---

## Week Map

| Day | Focus | Minimum Evidence / Artifact | Compute | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 43** | One TTS stack and feasibility | `docs/tts_pipeline.md`: checkpoint/license/tokenizer/speaker/data/parameter/L4 cost gate, feasible or deferred | `Modal L4` | [Open Day 43](days/day_43.md) |
| **Day 44** | Selected-stack duration and prosody | Measured selected-stack timing; native versus DSP controls labeled; FastSpeech theory only | `Modal L4` | [Open Day 44](days/day_44.md) |
| **Day 45** | Selected vocoder and boundary diagnostics | Frozen matched vocoder, measured seams, preservation tests; no vocoder training | `Modal L4` | [Open Day 45](days/day_45.md) |
| **Day 46** | Bounded TTS adaptation — base versus adapted | One feasible-branch experiment or explicit training deferral; null/worse outcomes valid | `Modal L4` | [Open Day 46](days/day_46.md) |
| **Day 47** | Speaker representation and preservation | Consented conditioning in the same stack; proxy limits and unsupported status | `Modal L4` | [Open Day 47](days/day_47.md) |
| **Day 48** | Selective reconstruction and stitching | Predicted-text repair, separate oracle rows, measured seams without forced improvement | `Modal L4 plus local CPU` | [Open Day 48](days/day_48.md) |
| **Day 49** | Cascaded repair evidence gate | `app/audio_lab.py` and tested `src/controller/abstain.py`; measured failures and limitations | `Modal L4` | [Open Day 49](days/day_49.md) |

## v2 Scope / Compression Map

| Day | v2 Status | Bound |
| :--- | :--- | :--- |
| **Day 43** | CORE | Select one permitted stack at the bounded feasibility gate; no model hunting |
| **Day 44** | CORE | Selected-stack prosody; FastSpeech 2 theory only |
| **Day 45** | CORE | Selected pretrained vocoder and seam tests; no separate training |
| **Day 46** | CORE — conditional training | Base-versus-adapted only if feasible; otherwise training deferred |
| **Day 47** | CORE | Same stack, explicit consent and conditioning provenance |
| **Day 48** | CORE | Predicted text; oracle diagnostics separate; null outcomes valid |
| **Day 49** | CORE | Abstention and one app required before Gate 6 |

CORE is planned scope, not a claim of completion. A deferred training branch
does not become measured adaptation; blocked base inference also
defers dependent synthesis work.

---

## Reference Spine
- The selected stack's model card, license, and supported adaptation recipe
- FastSpeech 2 and VITS papers as bounded theoretical contrasts only
- DSP references for energy matching and equal power crossfades

---

## Daily Detailed Operating Plans

### DAY 43: TTS system anatomy
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_43.md`](days/day_43.md)

> **v2 STATUS: CORE — one TTS stack and a bounded adaptation feasibility gate.** No second synthesis installation; completion follows evidence, not a date.

#### Learn
- Text or phoneme representation.
- Acoustic model.
- Mel spectrogram or latent representation.
- Vocoder.
- Speaker conditioning.
- Prosody.
- Content, speaker, and style representations; why useful factorization is not
  proof of perfect disentanglement.

#### Build in MendSpeech
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

#### Experiment and Measure
- Compare several sentences with punctuation and pacing changes.
- FastSpeech 2 and VITS are short theoretical contrasts, not additional models
  to install or benchmark. Record unresolved capabilities explicitly.

#### Required Output
- `src/tts/baseline.py`
- `results/day43_tts_samples/`
- `docs/tts_pipeline.md`

#### Completion Check
> You can explain the selected text-to-waveform path and speaker conditioning,
> and `docs/tts_pipeline.md` records checked licenses, data/split provenance,
> exact trainable parameters, L4 pilot evidence or a blocking reason, cost
> bounds, and a feasible or deferred adaptation decision. No training success
> is claimed by this gate; unresolved fields remain explicitly unverified.

---

### DAY 44: Selected-stack duration and prosody
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_44.md`](days/day_44.md)

> **v2 STATUS: CORE — reuse Day 43's selected stack.** FastSpeech 2 is theory only, not a second installation.

#### Learn
- Duration prediction.
- Pitch and energy predictors.
- Parallel generation intuition.

#### Build in MendSpeech
- Contrast FastSpeech 2 duration/pitch/energy predictors with the selected
  stack's generation path in `docs/day44_fastspeech2.md` (theory artifact).
- Reuse `src/tts/baseline.py`; inspect only controls actually exposed by the
  pinned revision. Do not invent native duration or pitch controls.
- Measure duration, pitch/energy summaries, and punctuation effects on fixed
  sentences and fixed consented speaker embeddings. A bounded post-synthesis
  duration adjustment must be labeled DSP, not learned prosody control.

#### Experiment and Measure
- Compare generated length against target intervals. If native rate control
  is unavailable, record `unsupported` and measure punctuation or DSP effects
  instead. Store waveforms ignored; commit only sample manifests/measurements.

#### Required Output
- `docs/day44_fastspeech2.md`
- `results/day44_prosody_samples/`

#### Completion Check
> You can explain short-span timing constraints using measured selected-stack
> behavior, distinguish native controls from DSP, and label unsupported controls.

---

### DAY 45: Vocoder realism and acoustic boundary diagnostics
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_45.md`](days/day_45.md)

> **v2 STATUS: CORE — boundary diagnostics and the selected stack's vocoder only.** No separate vocoder installation or training.

#### Learn
- Mel to waveform generation.
- HiFi GAN style generator and discriminator intuition.
- Phase, bandwidth, and vocoder artifacts.
- Short time energy, local loudness, spectral balance, and room tone as boundary signals.

#### Build in MendSpeech
- Reuse only Day 43's selected stack's matched pretrained vocoder; keep
  weights frozen. GAN anatomy is theory, not a separate training experiment.
- Add boundary diagnostics that measure short time energy and simple spectral statistics before and after a candidate repair span.
- Save a local room tone estimate where possible.

#### Experiment and Measure
- Measure inference speed and real time factor on L4 with fixed batch size,
  warm-up, and sample rate. Distinguish isolated vocoder from end-to-end time;
  mark isolated timing unavailable if the interface does not expose it.
- Create intentionally mismatched generated spans and verify that the boundary diagnostics flag obvious loudness or spectral discontinuities.
- Include unchanged/identity stitch controls and tests for sample-count and
  outside-span preservation. Record both flagged and missed seam artifacts.

#### Required Output
- `results/day45_vocoder_benchmark.csv`
- `src/repair/boundary_metrics.py`
- `docs/vocoder_and_boundary_notes.md`

#### Completion Check
> You can separate acoustic model errors from vocoder artifacts and quantify at least
two causes of an audible seam.

---

### DAY 46: Bounded TTS adaptation — base versus adapted
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_46.md`](days/day_46.md)

> **v2 STATUS: CORE — conditional adaptation of the one selected TTS stack.** Training is permitted only after Day 43's feasibility gate; blocked training is `deferred`, not measured adaptation.

#### Learn
- Small-data adaptation, frozen versus trainable parameters, overfitting,
  and held-out sentence evaluation with consented speaker conditioning.
- VITS latent variables, flows, and adversarial waveform synthesis may be
  compared theoretically; do not install or run a second TTS model.

#### Build in MendSpeech
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

#### Experiment and Measure
- Compare the frozen base and one adapted checkpoint on identical held-out
  sentences, speaker embeddings, generation settings, and L4 hardware.
  Measure intelligibility proxy, duration error, speaker proxy when supported,
  inference latency/RTF, trainable count, training time, memory, and actual cost.
- Randomize base/adapted sample order for a small listening check; report the
  number of raters/items and limitations. Keep test results out of selection.
- Record improvement, no meaningful change, or degradation as measured
  outcomes. Non-improvement is valid; incomplete or blocked training is not
  a negative result and must remain `deferred` with missing metrics, not zeros.

#### Required Output
- Feasible branch only: `training/tts_finetune.py`
- Feasible branch only: `configs/tts_finetune.yaml`
- Feasible branch only: `reports/day46_tts_adaptation.md` (including failures
  after starting; never claim a completed comparison if the run was blocked)
- `results/day46_tts_comparison.csv`
- `results/day46_listening_sheet.md`
- Update `docs/tts_pipeline.md` with the final measured/deferred status. On
  the deferred branch, the retained comparison/listening paths contain only
  available base evidence and explicit unavailable adapted-condition status.

#### Completion Check
> Either one bounded base-versus-adapted experiment has reproducible held-out
> evidence (including a valid null or worse result), or training is explicitly
> deferred with its blocking evidence. Feasibility-only work does not satisfy
> training completion and cannot be described as measured adaptation.

---

### DAY 47: Speaker representation and preservation
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_47.md`](days/day_47.md)

> **v2 STATUS: CORE — consented conditioning within the selected stack.** No second TTS installation.

#### Learn
- Speaker embeddings.
- Reference conditioned synthesis.
- Speaker similarity as a measurable but imperfect proxy.
- Consent and voice identity boundaries.

#### Build in MendSpeech
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

#### Experiment and Measure
- Compare full resynthesis with short span reconstruction for speaker similarity.

#### Required Output
- `src/tts/speaker_conditioning.py`
- `results/day47_speaker_similarity.csv`
- `docs/voice_use_policy.md`

#### Completion Check
> You can discuss speaker similarity measurements and their limitations without
claiming identity preservation from listening alone.

---

### DAY 48: Selective reconstruction with boundary matched stitching
- **Compute:** `Modal L4 plus local CPU for stitching`
- **Dedicated Daily File:** [`docs/days/day_48.md`](days/day_48.md)

> **v2 STATUS: CORE — predicted-text selective repair with measured seam outcomes.** Oracle text is a separate diagnostic, never the normal path.

#### Learn
- Repair span text selection.
- Timing constraints and duration control.
- Boundary padding and silence handling.
- Short time energy matching and local loudness matching.
- Linear versus equal power crossfades.
- Spectral and room tone mismatch.
- Why ASR to text to TTS can lose pitch, emotion, breathing, and coarticulation.

#### Build in MendSpeech
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

#### Experiment and Measure
- Compare full utterance TTS, naive selective repair, and boundary matched selective repair.
- Measure preservation percentage, latency, energy discontinuity, and speaker similarity proxy.
- Run a small blinded seam audibility check with randomized sample order.
- Hold predicted text and intervals fixed for stitching comparisons. Report
  smoother, unchanged, or worse seams; do not select cases to force an improvement.

#### Required Output
- `src/repair/reconstruct.py`
- `src/repair/stitch.py`
- `src/repair/boundary_metrics.py`
- `results/day48_selective_samples/`
- `results/day48_seam_ablation.csv`

#### Completion Check
> Predicted-text runs preserve samples outside declared repair/crossfade bounds,
> and seam metrics plus blinded checks compare matched and naive stitching on
> identical spans. Measured non-improvement is valid; oracle-only performance
> cannot satisfy the normal end-to-end check.

---

### DAY 49: Week 7 MendSpeech V1 cascaded repair milestone
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_49.md`](days/day_49.md)

> **v2 STATUS: CORE — Gate 6 is evidence-based.** Implement abstention now and extend the single `app/audio_lab.py`; no separate voice-agent project.

#### Learn
- Review TTS, duration, vocoder behavior, speaker conditioning, boundary matching, and information lost through the text bottleneck.
- Treat the cascaded path as a measured baseline, not a guaranteed real-time
  system. Label live versus simulated streaming/context control explicitly.

#### Build in MendSpeech
- Pipeline: damaged audio to streaming ASR to uncertain span to policy decision to speaker conditioned reconstruction to boundary matched waveform.
- Implement `src/controller/abstain.py` before this milestone, not on Day 54.
  Abstain when content evidence is insufficient, speaker use is unauthorized,
  or duration/boundary constraints cannot be met; preserve original audio and
  return a reason code. Use validation-set thresholds, never test-tuned ones.
- Extend only `app/audio_lab.py` for Preserve / Inspect / Repair / Abstain
  decisions, predicted-text reconstruction, and consent/capability status.
- Show preserved and reconstructed intervals with distinct visualization.
- Add a V1 label in results so the Week 8 direct audio repair comparison is explicit.

#### Experiment and Measure
- Run at least ten cases, including deliberate false repair, missed repair, seam artifacts, and one case where the policy abstains.
- Compare naive stitching and boundary matched stitching on the same repaired spans.
- Test low-evidence and permission-blocked abstention, clean no-repair cases,
  and unchanged samples outside declared edit/crossfade bounds. Keep oracle
  diagnostics separate and accept measured null/worse seam outcomes.

#### Required Output
- `app/audio_lab.py`
- `src/controller/abstain.py`
- `demos/week7_before_after/`
- `results/week7_stitching_ablation.csv`
- `reports/week7_cascaded_repair.md`

#### Completion Check
> MendSpeech V1 has tested abstention in the one app and measured predicted-text
> repair/seam evidence. Strengths, failures, deferred adaptation, and live versus
> simulated execution are documented; there is no deadline-based completion.

---

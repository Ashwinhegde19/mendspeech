# MendSpeech Complete 56-Day Plan

> **v2 current operational reference.** Compiled from the individual day specs.
> Read the [execution plan](REVISED_EXECUTION_PLAN.md) for scope, gates, and
> session accounting. Status banners override bodies: merged and dropped days
> do not create extra sessions or artifact obligations. Day numbers are stable
> identifiers, not calendar deadlines. Each link leads to the full source spec,
> including study protocol and references. Archived PDFs are unchanged.

---

## Week 1: Audio, Degradation, and Measurement Foundations

[Week 1 guide](Week_1_MendSpeech_Daily_Plan.md)

### Day 01: Waveforms, sampling, and the MendSpeech baseline

[Full Day 01 spec](days/day_01.md)

**Compute:** `Local CPU`

> **Still open after Day 01:** `data/benchmark/` and the `transcript` column
> were required here and were not finished. Do not re-run Day 01. Carry that
> labeled-corpus work into [Day 04](days/day_04.md). Use relative paths only.

#### Learn
- Waveform amplitude and time axes.
- Sampling rate, Nyquist intuition, bit depth, mono versus stereo.
- Why speech systems often standardize to 16 kHz.
- Duration, peak amplitude, RMS energy, and clipping.

#### Build in MendSpeech
- Create the repository and a minimal audio loader.
- Record or collect five clean speech clips with consent.
- Acquire a reference-transcripted subset (e.g., a LibriSpeech dev-clean slice) into `data/benchmark/`; the frozen benchmark will run on labeled clips, not only waveforms.
- Normalize all clips to a consistent sample rate and mono format.

#### Experiment and Measure
- Compare 8 kHz, 16 kHz, 24 kHz, and 48 kHz versions by listening and plotting.
- Measure duration, RMS energy, and file size for each version.

#### Required Output
- `notebooks/day01_waveform.ipynb`
- `data/clean_manifest.csv` (now includes a `transcript` column)
- `data/benchmark/` (reference-transcripted corpus slice)
- `docs/audio_baseline_notes.md`

#### Completion Check
> **Definition of Done for Day 01:**
> You can explain what is lost when sample rate is reduced and can reproduce the
same preprocessing from code.

### Day 02: Fourier intuition and STFT

[Full Day 02 spec](days/day_02.md)

**Compute:** `Local CPU`

#### Learn
- Frequency, phase, harmonics, and spectral energy.
- Fourier transform intuition without memorizing derivations.
- STFT frames, window length, hop length, overlap.
- Tradeoff between time resolution and frequency resolution.

#### Build in MendSpeech
- Implement STFT visualization with PyTorch or TorchAudio.
- Plot the same utterance with several window and hop settings.

#### Experiment and Measure
- Hold audio constant and change one STFT setting at a time.
- Write what phonetic or transient detail becomes easier or harder to see.

#### Required Output
- `notebooks/day02_stft.ipynb`
- `results/day02_stft_parameter_grid.png`

#### Completion Check
> **Definition of Done for Day 02:**
> You can choose a reasonable frame and hop configuration and explain why.

### Day 03: Mel scale and log Mel features

[Full Day 03 spec](days/day_03.md)

**Compute:** `Local CPU`

#### Learn
- Human frequency perception and the Mel scale.
- Mel filterbanks and log compression.
- Number of Mel bins and dynamic range.
- Normalization of acoustic features.

#### Build in MendSpeech
- Implement or inspect a log Mel feature pipeline.
- Build a function that returns features plus metadata needed for reproducibility.

#### Experiment and Measure
- Change Mel bin count and compare visual structure and compute size.
- Verify consistent feature shapes for different utterance lengths.

#### Required Output
- `src/audio/features.py`
- `notebooks/day03_logmel.ipynb`
- `tests/test_features.py`

#### Completion Check
> **Definition of Done for Day 03:**
> You can trace waveform to STFT to Mel filterbank to log Mel tensor.

### Day 04: Build SpeechDamageBench as a standalone package

[Full Day 04 spec](days/day_04.md)

**Compute:** `Local CPU`

> **Carry-forward from Day 01:** if `data/benchmark/` is still missing a
> labeled set, start it in this session. Day 07 only *freezes* the set; it
> does not collect it. Target: public transcripted speech (e.g. LibriSpeech
> `dev-clean`), relative paths, a `transcript` column, ≥5 speaker IDs.
> Reaching the full ≥30 utterances can finish on Day 05 — do not wait until
> Day 07.

#### Learn
- Deterministic corruption design and seed control.
- Additive noise, clipping, bandwidth limitation, dropouts, and reverberation.
- Why a benchmark should be reusable outside the main application.
- Versioned severity presets and manifest metadata.

#### Build in MendSpeech
- Create SpeechDamageBench as a **nested standalone package**. Do **not**
  overwrite the repo-root `pyproject.toml` (that file belongs to `mendspeech`).
- Implement noise, clipping, bandwidth reduction, dropout, and simple reverberation modules.
- Add a seed controlled configuration object and a small command line entry point.
- Record corruption name, severity, seed, parameters, and clean source id for every output.
- Presets live in `speechdamagebench/speechdamagebench/presets.py`. If you
  also want YAML, keep it inside the package (`presets/damage_levels.yaml`)
  so there is one source of truth.

#### Experiment and Measure
- Generate mild, medium, and severe examples from the same clean sentence.
- Reproduce the exact same damaged waveform from the same seed.
- Change only the seed and verify that the corruption changes while all configured parameters remain fixed.

#### Required Output
- `speechdamagebench/pyproject.toml` (package name `speechdamagebench`; do not touch repo-root `pyproject.toml`)
- `speechdamagebench/speechdamagebench/audio_damage.py`
- `speechdamagebench/speechdamagebench/presets.py`
- `speechdamagebench/speechdamagebench/cli.py`
- `speechdamagebench/tests/test_determinism.py`
- `data/benchmark/` begun if still missing (manifest with relative paths + transcripts)

#### Completion Check
> **Definition of Done for Day 04:**
> Another project can install SpeechDamageBench and regenerate the same damaged
clip from a manifest entry.

### Day 05: Objective audio measurements

[Full Day 05 spec](days/day_05.md)

**Compute:** `Local CPU`

#### Learn
- RMS and peak level.
- Simple SNR calculation when the clean reference is known.
- Spectral distance intuition.
- Why perceptual speech quality is not fully captured by one scalar metric.

#### Build in MendSpeech
- Add baseline metrics for clean versus corrupted pairs.
- Store results in a tidy CSV schema with clip id, corruption, severity, seed, and measurements.

#### Experiment and Measure
- Run all corruption levels on at least ten **labeled** clips from
  `data/benchmark/` (or `data/benchmark/` plus `data/clean_manifest.csv` if
  those rows have transcripts). If fewer than ten labeled clips exist, finish
  the corpus first — do not invent metrics on five unlabeled files.
- Look for cases where a metric disagrees with your listening judgment.

#### Required Output
- `src/metrics/audio_metrics.py`
- `results/week1_damage_metrics.csv`
- `docs/metric_limitations.md`

#### Completion Check
> **Definition of Done for Day 05:**
> You can explain what each metric says and what it fails to say.

### Day 06: Build the first MendSpeech audio console

[Full Day 06 spec](days/day_06.md)

**Compute:** `Local CPU`

#### Learn
- Audio playback in a lightweight interface.
- Waveform and spectrogram synchronization.
- Before and after comparison design.

#### Build in MendSpeech
- Build a local page or notebook dashboard with clean and damaged playback.
- Add corruption controls and immediately regenerate the damaged clip.
- Display waveform, spectrogram, and basic measurements.

#### Experiment and Measure
- Test with three speakers and several corruption types.
- Write down usability problems that would block later live debugging.

#### Required Output
- `app/audio_lab.py`
- `results/week1_audio_console.png`

#### Completion Check
> **Definition of Done for Day 06:**
> Another person can open the tool, damage an utterance, and understand the visual
change without reading your code.

### Day 07: Review, explain, and freeze Week 1

[Full Day 07 spec](days/day_07.md)

**Compute:** `Local CPU`

#### Learn
- Review waveform, STFT, Mel features, SNR, clipping, dropouts, and reverberation.

#### Build in MendSpeech
- Clean repository structure.
- **Freeze only.** Do not start collecting a new corpus on this day. The
  labeled set must already exist from Days 04–05. Freeze SpeechDamageBench
  v0.1 severity presets and a benchmark set of **≥30 utterances across ≥5
  speakers with reference transcripts** (lab scale is typically ~5 speakers;
  ≥5 is the floor, not a race to dozens), written as speaker-separated
  train/val/test splits in one manifest.
- Tag the benchmark package schema and add a minimal usage example independent of MendSpeech.

#### Experiment and Measure
- From a blank notebook, recreate one corruption and one log Mel plot without copying previous cells.

#### Required Output
- `reports/week1_audio_foundations.md`
- `data/benchmark_manifest.csv` (relative paths, `transcript`, `speaker_id`, and a `split` column: train/val/test)
- `speechdamagebench/README.md`
- `speechdamagebench/VERSION`

#### Completion Check
> **Definition of Done for Day 07:**
> You can teach the complete path from clean waveform to controlled corruption and
feature tensor.

---

## Week 2: ASR, CTC, Confidence, and Repair Localization

[Week 2 guide](Week_2_MendSpeech_Daily_Plan.md)

### Day 08: Frame sequence to transcript

[Full Day 08 spec](days/day_08.md)

**Compute:** `Modal L4 optional, CPU acceptable for
small runs`

> **v2 STATUS: CORE.** The external comparator requires a bounded feasibility
> record, not successful neural masked inpainting. This revision does not alter
> historical result evidence or assert that a new check has passed.

#### Learn
- Why acoustic frames outnumber output tokens.
- Encoder outputs, vocabulary logits, and decoding.
- CTC versus transducer versus attention decoder at a high level.

#### Build in MendSpeech
- Run a pretrained ASR model on clean and damaged SpeechDamageBench clips.
- Store transcript, token outputs if available, and timing metadata.
- Add a reusable Modal entry point so the same command can run ASR experiments on an L4 without editing deployment code each day.
- Check the single general-restoration candidate in `docs/baseline_install_notes.md`.
  VoiceFixer's documented interface is not evidence of mask-aware inpainting;
  label only verified capabilities. Use one setup session plus at most one
  focused compatibility retry, then stop. No model search or scratch fallback.
- Record license/permitted use, code/package/checkpoint revisions, invocation,
  supported conditions, native input/output format, mask support, and whether
  processing changes audio outside a target interval. Verify sample-rate and
  length conversion explicitly; unknown behavior remains unverified.
- Attempt one clean and one damaged smoke case. Record final feasibility
  `feasible` or `deferred`, attempt outcomes and blockers in the notes. If
  feasible, the only planned adapter is `src/baselines/direct_audio_restore.py`;
  do not create a second mask-specific adapter. Day 50/54 consume this record.

#### Experiment and Measure
- Compare clean and corrupted transcripts on the exact same utterances.
- Keep comparator smoke evidence separate from ASR results. Record source IDs,
  corruption parameters/seed, repeatability and any unsupported condition; leave
  unavailable metrics blank with a reason. Comparable GPU timing/memory uses L4.

#### Required Output
- `src/asr/baseline.py`
- `infra/modal_asr.py`
- `results/day08_baseline_transcripts.csv`
- `docs/baseline_install_notes.md` (one candidate, capability/provenance record,
  setup/retry evidence, `feasible` or `deferred`; no overwritten historical results)

#### Completion Check
> **Definition of Done for Day 08:**
> You can draw the path from features to encoder states to token probabilities to text,
and launch the same baseline locally or on Modal with a documented command.
The bounded comparator check has an evidence-backed status; a documented deferral
is sufficient for this external branch, but is not successful inpainting.

### Day 09: CTC from first principles

[Full Day 09 spec](days/day_09.md)

**Compute:** `Local CPU`

#### Learn
- CTC blank symbol.
- Repeated labels and collapse operation.
- Why many frame paths map to one transcript.
- Conditional independence assumption and its consequence.

#### Build in MendSpeech
- Implement CTC collapse yourself without a library decoder.
- Create hand written alignment examples and unit tests.

#### Experiment and Measure
- Enumerate several legal paths for a tiny target word.
- Break your decoder deliberately with repeated letters and fix it.

#### Required Output
- `src/asr/ctc_decode.py`
- `tests/test_ctc_decode.py`
- `docs/ctc_explained.md`

#### Completion Check
> **Definition of Done for Day 09:**
> You can explain why a blank is needed and correctly decode repeated characters.

### Day 10: WER, CER, and error taxonomy

[Full Day 10 spec](days/day_10.md)

**Compute:** `Local CPU`

#### Learn
- Word error rate: substitutions, deletions, insertions.
- Character error rate and when it helps.
- Why WER alone hides error severity.

#### Build in MendSpeech
- Implement or verify WER and CER calculations.
- Add an error analyzer that labels substitution, deletion, and insertion spans.

#### Experiment and Measure
- Score clean versus every SpeechDamageBench severity.
- Find which corruption type causes deletion errors fastest.

#### Required Output
- `src/metrics/wer.py`
- `results/day10_wer_by_damage.csv`
- `results/day10_error_types.csv`

#### Completion Check
> **Definition of Done for Day 10:**
> You can calculate WER by hand for a short example and explain each error.

### Day 11: Token confidence and uncertainty

[Full Day 11 spec](days/day_11.md)

**Compute:** `Modal L4 recommended`

#### Learn
- Softmax confidence and why it can be miscalibrated.
- Frame confidence versus token confidence versus word confidence.
- Entropy as an uncertainty signal.
- Confidence calibration intuition.

#### Build in MendSpeech
- Extract confidence or approximate it from model outputs.
- Create a word level confidence timeline aligned to the transcript.

#### Experiment and Measure
- Compare confidence on clean, noisy, clipped, and dropout audio.
- Find confident but wrong examples and document them.

#### Required Output
- `src/asr/confidence.py`
- `results/day11_confidence_cases.csv`
- `docs/confidence_failure_modes.md`

#### Completion Check
> **Definition of Done for Day 11:**
> You understand why low confidence can be useful but cannot be treated as truth.

### Day 12: Time alignment and uncertain spans

[Full Day 12 spec](days/day_12.md)

**Compute:** `Modal L4 recommended`

> **v2 STATUS: CORE.** Reuse the uncertainty overlay inside the single application.

#### Learn
- Frame time conversion.
- Token timestamps and word timestamps.
- Alignment boundaries around corrupted regions.

#### Build in MendSpeech
- Map low confidence tokens back to audio time spans.
- Overlay uncertain intervals on waveform and spectrogram.
- Keep `app/uncertainty_overlay.py` as a reusable visualization module imported
  by `app/audio_lab.py`, not a separately maintained runnable application.

#### Experiment and Measure
- Inject known 100 ms and 250 ms dropouts and test whether uncertainty overlaps them.

#### Required Output
- `src/asr/alignment.py`
- `app/uncertainty_overlay.py` (reusable module for `app/audio_lab.py`)
- `results/day12_overlap_metrics.csv`

#### Completion Check
> **Definition of Done for Day 12:**
> The shared UI can highlight an uncertain audio interval and show the associated
word or token without creating another application.

### Day 13: Define selective repair policy v0

[Full Day 13 spec](days/day_13.md)

**Compute:** `Local CPU after ASR outputs are
cached`

> **v2 STATUS: CORE.** Define safe action semantics now; exercise synthesis-time
> abstention by Day 49, not first at the final comparison.

#### Learn
- Threshold policies.
- Hysteresis to avoid rapid toggling.
- Minimum repair span and padding.
- False repair versus missed repair tradeoff.
- Uncertainty indicates a need for evidence, not permission to invent content.

#### Build in MendSpeech
- Keep Preserve, Balanced, and Rescue as sensitivity presets, not action labels.
  Each returns timed decisions with an action and reason code:
  - `preserve`: reliable audio remains unchanged.
  - `inspect`: flag uncertain content for review; do not synthesize it.
  - `repair`: propose a bounded edit only when content evidence, speaker-use
    permission, alignment and supported synthesis constraints are sufficient.
  - `abstain`: an unsafe or unsupported repair is refused; retain original audio
    and disclose why no reconstruction was produced.
- Missing evidence or unavailable synthesis support cannot silently become a
  repair. Week 2 tests decisions without claiming generated audio. Carry these
  semantics into `src/controller/abstain.py` and exercise them on Day 49.

#### Experiment and Measure
- Sweep thresholds on speaker-separated validation data only; log selected
  values in `configs/repair_modes.yaml` and freeze them before test scoring.
- Measure proposed repair coverage and overlap with known damage, false repair
  on clean speech, missed repair, and inspect/abstain rates. Ground-truth damage
  masks score decisions; they are not policy inputs in normal evaluation.
- Include reliable clean audio, uncertain text, missing permission, missing
  capability and invalid alignment cases; verify all non-repair actions leave
  audio unchanged. Raw confidence remains provisional until Day 41 calibration.

#### Required Output
- `src/controller/policy.py`
- `configs/repair_modes.yaml`
- `results/day13_policy_sweep.csv`

#### Completion Check
> **Definition of Done for Day 13:**
> You can explain and demonstrate preserve/inspect/repair/abstain decisions,
including refusal to synthesize unsupported content. Thresholds come only from
validation; false repairs and abstentions are visible rather than hidden.

### Day 14: Week 2 integration and review

[Full Day 14 spec](days/day_14.md)

**Compute:** `Modal L4 recommended`

> **v2 STATUS: CORE — Gate 2 advances on evidence, not a date.** Extend
> `app/audio_lab.py`; external-comparator deferral does not block the core ASR work.

#### Learn
- Review CTC, WER, confidence, timestamp alignment, and repair decisions.

#### Build in MendSpeech
- Extend `app/audio_lab.py`: damaged audio to transcript to confidence to timed
  preserve/inspect/repair/abstain proposals. Reuse the Day 12 overlay; do not
  create a separate milestone app or claim synthesis before it exists.
- Add clean JSON output for every run: source/corruption/seed, model revision,
  policy preset/version, thresholds, intervals, actions and reason codes.
- Verify the Modal wrapper records model revision, GPU type, software versions, and run id automatically.
- Carry forward `docs/baseline_install_notes.md`: one comparator's `feasible`
  or `deferred` status and verified capabilities. A blocker report is enough
  for this conditional branch; it must not be labeled masked-inpainting success.

#### Experiment and Measure
- Run at least twenty corrupted utterances with matched clean/raw-damaged
  controls and fixed validation-selected thresholds; inspect false repair,
  missed repair, inspect and abstain cases. Preserve model-version provenance.
- Report ASR WER/CER, uncertainty overlap, proposed repair coverage and clean
  false repairs; do not imply generated-audio improvement. Keep L4 comparisons
  separate from functional CPU smoke runs.

#### Required Output
- `app/audio_lab.py`
- `infra/modal_asr.py`
- `results/week2_casebook.md`
- `reports/week2_asr_uncertainty.md`

#### Completion Check
> **Definition of Done for Day 14:**
> The one app shows what the ASR heard and the exact proposed actions, with
unchanged audio for inspect/abstain. The casebook/report retain controls, model
and policy versions, errors and comparator feasibility status. Gate 2 does not
require a successful external neural restoration model.

---

## Week 3: Conformer From First Principles

[Week 3 guide](Week_3_MendSpeech_Daily_Plan.md)

### Day 15: Attention for speech sequences

[Full Day 15 spec](days/day_15.md)

**Compute:** `Local CPU, L4 optional for scaling`

#### Learn
- Query, key, value projections.
- Scaled dot product attention.
- Attention masks.
- Sequence length cost.

#### Build in MendSpeech
- Implement single head attention and then multi head attention in PyTorch.
- Add shape assertions and gradient tests.

#### Experiment and Measure
- Change sequence length and measure forward time and memory.

#### Required Output
- `src/models/attention.py`
- `tests/test_attention.py`
- `results/day15_attention_scaling.csv`

#### Completion Check
> **Definition of Done for Day 15:**
> You can derive every major tensor shape and explain quadratic sequence cost.

### Day 16: Conformer convolution module

[Full Day 16 spec](days/day_16.md)

**Compute:** `Local CPU`

#### Learn
- Pointwise convolution.
- GLU gating.
- Depthwise convolution.
- Batch normalization and activation.
- Why local patterns matter in speech.

#### Build in MendSpeech
- Implement a Conformer style convolution module.
- Test causality assumptions and receptive field.

#### Experiment and Measure
- Feed synthetic impulses and inspect how local information spreads.

#### Required Output
- `src/models/conformer_conv.py`
- `tests/test_conformer_conv.py`
- `notebooks/day16_receptive_field.ipynb`

#### Completion Check
> **Definition of Done for Day 16:**
> You can explain why depthwise convolution is computationally attractive and what
local context it captures.

### Day 17: Macaron feed forward and residual scaling

[Full Day 17 spec](days/day_17.md)

**Compute:** `Local CPU`

> **v1 STATUS: LEARN-ONLY — no build session.** Complete the Learn block during theory time; the macaron build work moves into [Day 18](days/day_18.md).

#### Learn
- Feed forward expansion.
- Swish or SiLU activation.
- Dropout.
- Half step residual weighting in Conformer.

#### Build in MendSpeech
- Implement the feed forward module and residual wrapper.
- Add numerical tests for shape and gradient flow.

#### Experiment and Measure
- Compare output statistics with and without residual scaling.

#### Required Output
- `src/models/conformer_ffn.py`
- `tests/test_conformer_ffn.py`

#### Completion Check
> **Definition of Done for Day 17:**
> You can explain the ordering of the Conformer block without memorizing a diagram.

### Day 18: Assemble one Conformer block

[Full Day 18 spec](days/day_18.md)

**Compute:** `Local CPU`

> **v2 STATUS: CORE — absorbs Days 17 and 19.** Implement the macaron feed-forward, assemble one tested Conformer block, and pass real log-Mel features through that same block with input projection and mask propagation. No separate encoder or depth sweep.

#### Learn
- Macaron structure.
- Layer normalization placement.
- Attention plus convolution interaction.
- Input projection, padding masks, and temporal dimensions for real log-Mel features.

#### Build in MendSpeech
- Implement the macaron feed-forward and half-step residual wrapper absorbed from Day 17, with shape and gradient tests.
- Assemble feed forward, the scratch attention and convolution modules, second feed forward, and final normalization into one block.
- Add input projection from real log-Mel features and propagate padding masks through this same block; assert input, output, and valid-length shapes.

#### Experiment and Measure
- Run forward and backward tests on several sequence lengths.
- Trace a real log-Mel tensor through projection and every stage; test mask handling and finite, nonzero gradients on valid inputs.
- Compare output statistics with and without half-step residual scaling in the same tests. A second toy-training ablation is not a release requirement.

#### Required Output
- `src/models/conformer_ffn.py` — Day 17's absorbed macaron implementation
- `tests/test_conformer_ffn.py`
- `src/models/conformer_block.py`
- `tests/test_conformer_block.py`
- `docs/conformer_block_walkthrough.md`
- `results/day19_shape_trace.md` — retained Day 19 path, produced here

#### Completion Check
> **Definition of Done for Day 18:**
> You can explain every operation in one tested block. A real log-Mel tensor
> passes through its projection and mask handling with documented shapes and
> valid gradients; the retained Day 19 shape trace records the evidence.

### Day 19: Real-log-Mel block validation (merged into Day 18)

[Full Day 19 spec](days/day_19.md)

**Compute:** `Local CPU — within Day 18`

> **v2 STATUS: MERGED into Day 18.** No standalone session. Validate input projection, mask propagation, shapes, and gradients on Day 18's same Conformer block; do not build a separate encoder or run a depth sweep.

#### Learn
- Input projection.
- Mask propagation.
- Temporal dimensions.

#### Build in MendSpeech
- In Day 18, connect real log-Mel features through input projection to the same tested Conformer block.
- Propagate padding masks and validate temporal dimensions; no separate implementation.

#### Experiment and Measure
- In Day 18, trace shapes through every stage on real speech and test masks and finite, nonzero gradients on valid inputs.

#### Required Output
- `results/day19_shape_trace.md` — produced in Day 18; no standalone code artifact

#### Completion Check
> **Definition of Done for Day 19:**
> Absorbed into Day 18: a real log-Mel tensor passes through the same block's
> projection and mask handling with a documented shape trace and valid gradients.

### Day 20: Compare your block with a production

[Full Day 20 spec](days/day_20.md)

**Compute:** `Local CPU`

> **v1 STATUS: DROPPED — no session.** Optional spare-time reading only: orient in production Conformer code. Do not schedule an evening session for this day.

#### Learn
- Read the original Conformer paper sections relevant to block design.
- Inspect a mature implementation such as NeMo.
- Identify differences caused by engineering and efficiency.

#### Build in MendSpeech
- Create an annotated comparison table: your component, paper definition, production implementation.

#### Experiment and Measure
- Choose one difference and reproduce its effect on a small benchmark if feasible (optional — Week 3 is a learning artifact; the production encoder is NeMo FastConformer from Week 4).

#### Required Output
- `docs/day20_implementation_comparison.md`

#### Completion Check
> **Definition of Done for Day 20:**
> You can read production Conformer code and orient yourself without treating it as
magic.

### Day 21: Week 3 architecture review

[Full Day 21 spec](days/day_21.md)

**Compute:** `Local CPU`

> **v2 STATUS: CORE — static architecture review.** Keep the review milestone and shape evidence; no architecture-inspector UI.

#### Learn
- Review attention, convolution, feed forward, normalization, residual paths, and sequence cost.

#### Build in MendSpeech
- Write a static architecture report showing the single block's stage shapes, mask propagation, and context assumptions, using `results/day19_shape_trace.md` from Day 18.

#### Experiment and Measure
- Give yourself a ten minute whiteboard explanation from waveform features through one Conformer block.

#### Required Output
- `reports/week3_conformer.md`

#### Completion Check
> **Definition of Done for Day 21:**
> You can explain which parts are local, which are global, and which become
> problematic for streaming, with the static report linked to the tested shape trace.

---

## Week 4: FastConformer and Efficient Encoder Behavior

[Week 4 guide](Week_4_MendSpeech_Daily_Plan.md)

### Day 22: Why FastConformer exists

[Full Day 22 spec](days/day_22.md)

**Compute:** `Local CPU`

> **v1 STATUS: LEARN-ONLY — merged into [Day 23](days/day_23.md).** Paper notes and compute estimates only; no separate session.

#### Learn
- Sequence length as an attention cost driver.
- Subsampling before expensive encoder blocks.
- Depthwise separable convolution.
- Local and limited context attention.

#### Build in MendSpeech
- Read the FastConformer paper with a comparison checklist.
- Write a diagram showing what changes relative to Conformer.

#### Experiment and Measure
- Estimate attention matrix size before and after aggressive temporal subsampling.

#### Required Output
- `docs/day22_fastconformer_notes.md`
- `results/day22_compute_estimates.csv`

#### Completion Check
> **Definition of Done for Day 22:**
> You can explain FastConformer as a set of concrete efficiency choices, not just a
faster model name.

### Day 23: Temporal subsampling experiment

[Full Day 23 spec](days/day_23.md)

**Compute:** `Modal L4 useful`

> **v1 STATUS: CORE — absorbs Day 22.** Also cover Day 22's FastConformer-vs-Conformer comparison checklist and compute estimates in this session.

#### Learn
- Convolutional subsampling.
- Temporal resolution.
- Information loss versus compute reduction.

#### Build in MendSpeech
- Implement a small subsampling front end or isolate one from a framework.
- Track frames per second before and after each stage.
- Incorporate Day 22's comparison checklist and diagram of FastConformer efficiency choices.

#### Experiment and Measure
- Compare 2x, 4x, and 8x temporal reduction on tensor length, runtime, and rough output behavior.
- Estimate attention-matrix size before and after subsampling; record the absorbed Day 22 evidence.

#### Required Output
- `src/models/subsampling.py`
- `results/day23_subsampling.csv`
- `docs/day22_fastconformer_notes.md` — absorbed Day 22 evidence
- `results/day22_compute_estimates.csv` — absorbed Day 22 evidence

#### Completion Check
> **Definition of Done for Day 23:**
> You can quantify how subsampling changes sequence length and downstream
attention cost.

### Day 24: Pretrained FastConformer baseline

[Full Day 24 spec](days/day_24.md)

**Compute:** `Modal L4`

> **v2 STATUS: CORE — baseline and early capability check.** Verify the selected checkpoint's supported behavior before downstream streaming and export work; unsupported capabilities are documented, not replaced by a model search or new architecture.

#### Learn
- Model checkpoint loading.
- Tokenizer and decoder configuration.
- Batch versus single utterance inference.
- Checkpoint-specific cache-aware inference, right-context controls, export support, and tokenizer language coverage.

#### Build in MendSpeech
- Run a current NeMo FastConformer checkpoint on your clean and damaged sets.
- Record model revision and all inference settings.
- In `configs/model_baseline.yaml`, record capability status and evidence for cache-aware inference, supported right-context values and units, runtime context switching, the intended export path, and tokenizer language support for the planned evaluation languages. Distinguish verified, unsupported, and unverified behavior.
- Check the pinned model/framework documentation and use minimal supported smoke checks where feasible. Record failures and limitations early; do not start a model hunt, retrain an encoder, or add custom infrastructure to manufacture support.

#### Experiment and Measure
- Benchmark WER, latency, and GPU memory by damage type.
- Carry the capability record into Days 25, 31–35 and later export work. Unsupported adaptive switching defers the adaptive claim, not unrelated gate evidence; unsupported cache-aware inference or export remains an explicit dependency issue, not a completed requirement.

#### Required Output
- `src/asr/fastconformer_runner.py`
- `results/day24_fastconformer_baseline.csv`
- `configs/model_baseline.yaml`

#### Completion Check
> **Definition of Done for Day 24:**
> You have a reproducible baseline with model, data, hardware, and settings fixed,
> plus an evidence-backed capability record covering cache-aware inference,
> right context, export, and tokenizer language support. Unsupported or unverified
> capabilities and their downstream implications are explicit.

### Day 25: Context and attention limits

[Full Day 25 spec](days/day_25.md)

**Compute:** `Modal L4`

#### Learn
- Full context attention.
- Limited context attention.
- Left and right context.
- Accuracy versus latency intuition.

#### Build in MendSpeech
- Inspect context settings in the model configuration.
- Create a visual timeline explaining visible past and future context.

#### Experiment and Measure
- If supported, compare at least two context settings on the same subset.

#### Required Output
- `docs/day25_context_timeline.md`
- `results/day25_context_compare.csv`

#### Completion Check
> **Definition of Done for Day 25:**
> You can explain exactly why future context creates algorithmic latency.

### Day 26: Efficiency benchmark harness

[Full Day 26 spec](days/day_26.md)

**Compute:** `Modal L4`

#### Learn
- Warmup runs.
- Synchronized GPU timing.
- Median and percentile latency.
- Real time factor.
- Peak memory.

#### Build in MendSpeech
- Create one benchmark function used by every later experiment.
- Log environment and model metadata automatically.

#### Experiment and Measure
- Run repeated inference and calculate variance.
- Detect and discard obviously invalid cold start comparisons.

#### Required Output
- `src/bench/benchmark_asr.py`
- `src/bench/environment.py`
- `results/day26_repeatability.csv`

#### Completion Check
> **Definition of Done for Day 26:**
> Repeated runs produce stable enough numbers to support comparisons.

### Day 27: FastConformer failure casebook

[Full Day 27 spec](days/day_27.md)

**Compute:** `Modal L4`

> **v1 STATUS: MERGED into [Day 28](days/day_28.md).** Keep only the top-3 failure patterns; no separate session.

#### Learn
- Error slicing by corruption type and severity.
- Short versus long utterance effects.
- Confidence versus error.

#### Build in MendSpeech
- Build a casebook of at least fifteen interesting failures.
- Link each case to audio, transcript, confidence, and damage metadata.

#### Experiment and Measure
- Look for systematic error patterns rather than isolated anecdotes.

#### Required Output
- `results/fastconformer_failure_casebook.md`

#### Completion Check
> **Definition of Done for Day 27:**
> You can name at least three repeatable failure patterns and propose a testable
reason for each.

### Day 28: Week 4 integration

[Full Day 28 spec](days/day_28.md)

**Compute:** `Modal L4`

> **v2 STATUS: CORE — absorbs Day 27.** Integration plus the top-3 failure casebook in one session, using the shared `app/audio_lab.py` entrypoint.

#### Learn
- Review efficiency choices and baseline results.

#### Build in MendSpeech
- Replace the generic ASR runner in MendSpeech with the reproducible FastConformer path.
- Extend `app/audio_lab.py`, the single app entrypoint, to expose latency, RTF, WER when reference text exists, and GPU memory. Do not create a versioned demo app.
- Capture Day 27's top three repeatable failure patterns with transcript, confidence, and damage metadata in the retained casebook.

#### Experiment and Measure
- Run the same ten reference clips through the full Week 2 uncertainty policy using FastConformer.

#### Required Output
- `app/audio_lab.py`
- `results/fastconformer_failure_casebook.md` — absorbed Day 27 evidence
- `reports/week4_fastconformer.md`

#### Completion Check
> **Definition of Done for Day 28:**
> The shared audio lab exposes a measured, inspectable FastConformer recognition
> core, and the report links the top-three failure casebook.

---

## Week 5: Streaming, Cache Aware Inference, and Adaptive Context

[Week 5 guide](Week_5_MendSpeech_Daily_Plan.md)

### Day 29: Offline versus streaming ASR

[Full Day 29 spec](days/day_29.md)

**Compute:** `Modal L4`

#### Learn
- Audio chunks.
- Algorithmic latency.
- Partial hypotheses.
- Endpointing and finalization.

#### Build in MendSpeech
- Create a chunk simulator that feeds audio incrementally.
- Log when each chunk becomes available and when text changes.

#### Experiment and Measure
- Compare offline transcript with naive chunk by chunk transcription.

#### Required Output
- `src/streaming/chunker.py`
- `results/day29_offline_vs_naive.csv`

#### Completion Check
> **Definition of Done for Day 29:**
> You can explain why naive chunking creates boundary errors and redundant compute.

### Day 30: Buffered streaming

[Full Day 30 spec](days/day_30.md)

**Compute:** `Modal L4`

#### Learn
- Overlapping windows.
- Buffer size.
- Stride.
- Repeated computation.

#### Build in MendSpeech
- Implement or run buffered streaming with configurable overlap.
- Measure how much audio is recomputed.

#### Experiment and Measure
- Sweep buffer and stride settings.
- Measure WER and latency tradeoffs.

#### Required Output
- `src/streaming/buffered.py`
- `results/day30_buffered_sweep.csv`

#### Completion Check
> **Definition of Done for Day 30:**
> You can quantify the compute waste caused by overlapping history.

### Day 31: Cache aware streaming internals

[Full Day 31 spec](days/day_31.md)

**Compute:** `Modal L4`

#### Learn
- Cached activations.
- Past context state.
- Streaming masks.
- Right context and lookahead.

#### Build in MendSpeech
- Use NeMo cache aware streaming inference on a supported FastConformer checkpoint.
- Log cache related configuration and chunk boundaries.
- If cache-aware inference is unsupported for the chosen checkpoint, document the limitation and fall back to buffered streaming; the buffered vs cache comparison still runs.

#### Experiment and Measure
- Compare buffered and cache aware inference on the same audio and same hardware.

#### Required Output
- `src/streaming/cache_aware_runner.py`
- `results/day31_buffered_vs_cache.csv`

#### Completion Check
> **Definition of Done for Day 31:**
> You can explain what is cached, what is recomputed, and why cache aware inference
can be more efficient.

### Day 32: Supported fixed-context lookahead ablation

[Full Day 32 spec](days/day_32.md)

**Compute:** `Modal L4`

> **v2 STATUS: CORE — fixed context only.** Use only configurations verified for the Day 24 checkpoint. This session does not require live context switching.

#### Learn
- Right context.
- Lookahead.
- Commit delay.
- WER and latency as competing objectives.

#### Build in MendSpeech
- Run the supported fixed-context configurations recorded in Day 24, using a separate run per setting and holding model, audio, hardware, batching, and decoding fixed.
- Store per-utterance and aggregate metrics, exact context values and units, and capability status. Do not coerce unsupported values or build a new streaming path.

#### Experiment and Measure
- Plot measured WER versus measured L4 latency and identify dominated operating points only when multiple supported settings exist.
- If only one setting is supported, retain its measured point and label the comparison unavailable; if none is runnable, record the reason and leave metrics missing. Keep the CSV and figure paths, with an explicitly annotated unavailable comparison rather than fabricated points or latency.

#### Required Output
- `experiments/lookahead_ablation.py`
- `results/day32_lookahead.csv`
- `results/day32_pareto.png`

#### Completion Check
> **Definition of Done for Day 32:**
> You can defend a supported fixed operating point using measured data, or show
> why the comparison is unavailable. A single point is not a Pareto frontier;
> unsupported context variation defers that claim, not the rest of Gate 4.

### Day 33: Break the cache on purpose

[Full Day 33 spec](days/day_33.md)

**Compute:** `Modal L4`

#### Learn
- State continuity.
- Chunk boundary dependencies.
- Cache reset and truncation.

#### Build in MendSpeech
- Add controlled experiments that reset or shorten cache at selected boundaries.

#### Experiment and Measure
- Measure WER changes around the reset point.
- Inspect whether errors cluster near boundaries or propagate later.

#### Required Output
- `experiments/cache_break_test.py`
- `results/day33_cache_failures.md`

#### Completion Check
> **Definition of Done for Day 33:**
> You can explain a concrete failure caused by incorrect state handling.

### Day 34: Bounded adaptive-context comparison

[Full Day 34 spec](days/day_34.md)

**Compute:** `Modal L4`

> **v2 STATUS: CORE — capability-bounded comparison.** Use only Day 24/32 supported settings. Live adaptation is conditional; unavailable support defers the adaptive claim, not the remaining streaming, endpointing, cache, or serving requirements.

#### Learn
- Policy driven context selection.
- Confidence smoothing.
- Latency budget.
- Stability versus oscillation.

#### Build in MendSpeech
- Implement a small capability-guarded policy that classifies chunks as easy or uncertain using the existing uncertainty signal and fixed decision rules.
- Compare at most two supported right-context settings from Day 32. Use live switching only when the checkpoint supports it; otherwise simulate policy choices from separate fixed-setting runs on the same controlled subset.
- If fewer than two settings are supported, return an explicit unavailable status and reason. No custom serving infrastructure, new model, or architecture change to force adaptation.

#### Experiment and Measure
- Compare the two fixed policies and the bounded adaptive policy where supported, with an explicit `live`, `simulated`, or `unavailable` status for the adaptive comparison.
- A simulated comparison is offline policy evidence, not live adaptive latency. Keep measured fixed-run timings separate; leave adaptive latency missing unless measured on a real live switching run. Do not fabricate measurements or claim a benefit from a simulation alone.

#### Required Output
- `src/controller/adaptive_context.py` — bounded policy and capability guard; reports unavailable when unsupported
- `results/day34_adaptive_context.csv` — retain this path even when unavailable; include status, supported settings, evidence/reason, and missing values for unmeasured metrics

#### Completion Check
> **Definition of Done for Day 34:**
> You have a measured live comparison, an explicitly limited simulated policy
> comparison, or an evidence-backed unavailable result. Unsupported behavior
> honestly defers the adaptive claim without waiving the rest of Gate 4.

### Day 35: Week 5 live streaming milestone

[Full Day 35 spec](days/day_35.md)

**Compute:** `Modal L4`

> **v2 STATUS: CORE — shared audio lab streaming milestone.** Streaming, VAD/endpointing, cache-state handling, and Gate 4 Add-on B serving/load evidence remain required. Adaptive context is displayed only where supported.

#### Learn
- Review buffered streaming, cache aware inference, lookahead, cache failures,
  adaptive context, VAD-driven endpointing, and partial-versus-final latency.

#### Build in MendSpeech
- Extend the existing `app/audio_lab.py` entrypoint to connect microphone or simulated live audio to the streaming recognizer; do not create another app.
- Reuse Add-on A VAD for endpointing and log speech start, speech end, and
  finalization timestamps.
- Show partial text, confidence timeline, VAD/endpointing state, cache state, queue depth, and measured latency. Show current context mode only when the runner exposes it; distinguish fixed from live adaptive behavior and never present a simulated policy as live.
- Keep Gate 4 Add-on B async serving, per-stream isolation, backpressure, and load/failure evidence required. Link those artifacts rather than building context-specific serving infrastructure.

#### Experiment and Measure
- Record a short demo with clean and damaged speech.
- Measure time to first partial transcript, endpoint delay, false starts, and
  missed endpoints on the same cases.
- Document remaining technical limitations honestly.
- Link Day 31/33 cache evidence, Day 34's live/simulated/unavailable status, and Add-on B serving/load results. Unsupported adaptive context does not waive endpointing, cache correctness, or serving checks.

#### Required Output
- `app/audio_lab.py`
- `demos/week5_streaming_demo.mp4`
- `reports/week5_streaming.md`

#### Completion Check
> **Definition of Done for Day 35:**
> A person can speak and watch MendSpeech transcribe incrementally while exposing
> VAD, endpointing, cache, and uncertainty state, with context mode shown only
> where available. The report links required serving/load and cache evidence;
> any adaptive deferral is explicit and does not substitute for those checks.

---

## Week 6: Robustness, Fine Tuning, RNNT, and Calibration

[Week 6 guide](Week_6_MendSpeech_Daily_Plan.md)

### Day 36: Training pipeline anatomy

[Full Day 36 spec](days/day_36.md)

**Compute:** `Modal L4`

#### Learn
- Manifest format.
- Batching variable duration audio.
- Loss curves.
- Learning rate.
- Validation split.
- Checkpointing.

#### Build in MendSpeech
- Create a tiny reproducible training configuration.
- Run a short smoke training job and verify loss decreases.

#### Experiment and Measure
- Deliberately use a bad learning rate and record the failure signature.

#### Required Output
- `configs/train_smoke.yaml`
- `results/day36_training_smoke.csv`
- `docs/training_debug_notes.md`

#### Completion Check
> **Definition of Done for Day 36:**
> You can diagnose whether a run is learning, diverging, or overfitting from basic
evidence.

### Day 37: Build a robust fine tuning dataset

[Full Day 37 spec](days/day_37.md)

**Compute:** `Local CPU`

> **v2 STATUS: CORE.** Prepare one ASR adaptation dataset. The frozen core
> evaluation set is immutable; any Indic extension stays separate.

#### Learn
- Train, validation, test separation.
- Speaker leakage.
- Synthetic corruption sampling.
- Balanced severity distribution.

#### Build in MendSpeech
- Prepare train/validation manifests pairing verified transcripts with clean
  and corrupted audio for one adaptation experiment. Record source/speaker IDs,
  language, license/consent, transcript verification and normalization, corruption,
  severity, seed, parameters and package version; retain clean examples.
- Preserve the already frozen test membership and speaker-separated splits.
  All clean/corrupted copies of a source stay in one split. A test manifest is
  a reference to the frozen evaluation set, never a new sample or rewritten set.
- Optionally reuse the separate `data/indic_codemix_manifest.csv` prepared after
  Gate 2 for the same adaptation experiment only if the checkpoint/tokenizer
  supports the language and a competent transcript verifier is available.
  Keep extension train/validation/test speakers disjoint from each other and
  the core evaluation speakers. Otherwise retain it as evaluation-only and
  record the limitation; do not add a second training track.

#### Experiment and Measure
- Audit source-level duplicates, speaker leakage and corruption provenance.
  Report split counts, clean/severity balance and immutable core test membership.
- Record extension language support/verification evidence and inclusion or
  evaluation-only status. Add-on C streaming metrics remain for later evaluation,
  not invented data-preparation results.

#### Required Output
- `data/train_manifest.jsonl`
- `data/val_manifest.jsonl`
- `data/test_manifest.jsonl` (frozen-set reference; preserve existing contents)
- `reports/data_audit.md`

#### Completion Check
> **Definition of Done for Day 37:**
> The audit demonstrates no source or speaker leakage into training/validation,
preserves the frozen core evaluation set and documents provenance. Any optional
extension is separately identified and does not create another training track.

### Day 38: Fine tune for damaged speech robustness

[Full Day 38 spec](days/day_38.md)

**Compute:** `Modal L4; use a manageable training configuration within budget`

> **v2 STATUS: CORE.** One bounded ASR adaptation experiment with a clean-speech
> regression control; negative results count as evidence, failed runs do not.

#### Learn
- Transfer learning.
- Frozen versus trainable layers.
- Mixed precision.
- Gradient accumulation.

#### Build in MendSpeech
- Adapt the existing compatible ASR checkpoint once using Day 37's audited data.
  Fix trainable layers, learning rate, steps, seed, clean/damaged sampling,
  precision, gradient accumulation and budget before running; choose the best
  checkpoint using validation only, never frozen test outcomes.
- If the optional Indic data passes Day 37's support and verification checks,
  include it in this same experiment and report its slice separately. Otherwise
  retain evaluation-only coverage; do not install/train a second recognizer.
- Save the reproducible configuration and provenance: base code/weight revisions,
  manifest hashes, source permissions, software/hardware and selection rule.
  Keep checkpoints and raw training logs local/gitignored. Day 39's augmentation
  ablation reuses this recipe; it is not a separate adaptation track.

#### Experiment and Measure
- Compare base and adapted checkpoints on identical frozen clean/damaged cases
  with the same decoder and normalization. Report WER/CER by condition, clean
  regression, confidence shifts and failures, not only aggregate improvement.
- Report any extension results separately from the frozen core benchmark. All
  comparable latency/RTF/peak-memory measurements use the same L4 configuration.
  Record actual cost and a failed run honestly instead of claiming adaptation.

#### Required Output
- `training/finetune.py`
- `configs/finetune.yaml`
- `checkpoints/week6_best/` (local/gitignored weights and raw logs)
- `results/day38_base_vs_adapted.csv`
- `reports/day38_adaptation.md` (provenance, selection rule, cost and limitations)

#### Completion Check
> **Definition of Done for Day 38:**
> The single base-versus-adapted experiment is reproducible and states what
improved, what did not and whether clean speech regressed. Test data never
selects the checkpoint, and optional language coverage is labeled separately.

### Day 39: SpecAugment and augmentation ablation

[Full Day 39 spec](days/day_39.md)

**Compute:** `Modal L4`

#### Learn
- Time masking.
- Frequency masking.
- Data augmentation as invariance training.

#### Build in MendSpeech
- Add one augmentation intervention to a controlled short run.

#### Experiment and Measure
- Compare no augmentation versus selected augmentation with the same seed and training budget.

#### Required Output
- `experiments/specaugment_ablation.py`
- `results/day39_augmentation.csv`

#### Completion Check
> **Definition of Done for Day 39:**
> You can separate the effect of augmentation from the effect of extra training time.

### Day 40: RNN-T concepts and quantization lab

[Full Day 40 spec](days/day_40.md)

**Compute:** `Modal L4`

> **v2 STATUS: CORE.** Retain the quantization lab behind a compatibility check.
> Unsupported precision is a documented blocker, not a completed optimization.

#### Learn
- Encoder.
- Prediction network.
- Joint network.
- Blank handling.
- Streaming emission behavior.
- Difference from CTC independence.
- Post-training quantization: dynamic vs static INT8, and why static needs a calibration set.
- What quantization can and cannot preserve in an ASR model (logit sharpness, confidence behavior).

#### Build in MendSpeech
- Smoke-check the selected ASR model's export and precision support on one
  compatible L4 backend. Pin model/backend revisions and verify supported
  operators, dynamic lengths, decoding and actual device/kernel placement.
  Do not add a provider sweep or silently compare CPU INT8 with GPU inference.
- Export the Day 38 checkpoint only through the supported path. Check the
  original model versus exported model at the same precision on validation
  clips before quantization: output/logit tolerances where exposed, decoded
  text, lengths and decoding settings. Record and resolve parity failures first.
- Compare FP16 and INT8 only where the same backend supports their intended
  execution on L4. For static INT8, select and record a representative calibration
  slice from validation, disjoint by source/speaker from test; never calibrate
  using the frozen test benchmark. State if the backend does not need calibration.
- If export, parity or precision support blocks measurement, retain working
  inference and record the exact failed check, error and unsupported precision.
  Do not change models/backends repeatedly to manufacture an INT8 result.

#### Experiment and Measure
- After parity passes, measure original/exported baseline and supported FP16/INT8
  variants on identical frozen cases, decoder, batch size, timing boundaries,
  warm-up and L4 hardware. Report actual WER/CER, latency/RTF and peak memory;
  INT8 may be slower or less accurate and need not be selected for deployment.
- Record logit/confidence shifts for Day 41 validation-based calibration. Report
  precision coverage and CPU fallbacks explicitly; a mixed-device run is not
  a controlled GPU speed comparison.
- Give each variant `measured` or `blocked` status with reasons and blank
  unavailable metrics. Gate 5 can carry an explicit blocked optimization status,
  but cannot claim successful quantization or a speedup without measurements.

#### Required Output
- `docs/rnnt_walkthrough.md` (theory summary from the Learn block)
- `docs/day40_quantization_notes.md` (compatibility, parity, calibration source,
  backend/precision settings and blockers)
- `results/day40_quantization_tradeoffs.csv` (actual metrics or blocked rows)

#### Completion Check
> **Definition of Done for Day 40:**
> You can explain RNN-T streaming behavior and demonstrate original/export
parity plus measured supported-precision tradeoffs on L4, or identify the exact
compatibility/parity blocker. A blocked branch stays explicitly unimplemented;
no unsupported INT8, speedup or calibration claim is presented as complete.

### Day 41: Confidence calibration for repair decisions

[Full Day 41 spec](days/day_41.md)

**Compute:** `Modal L4 for logits, local CPU for
analysis`

#### Learn
- Reliability diagrams.
- Expected calibration error intuition.
- Threshold selection from validation data.

#### Build in MendSpeech
- Build a simple calibration analysis for confidence versus correctness.
- Choose policy thresholds on validation, not test.

#### Experiment and Measure
- Compare raw and calibrated confidence if a simple method is feasible.

#### Required Output
- `src/asr/calibration.py`
- `results/day41_reliability.png`
- `configs/repair_modes_calibrated.yaml`

#### Completion Check
> **Definition of Done for Day 41:**
> Repair thresholds are now justified from held out evidence rather than guessed.

### Day 42: Week 6 robustness milestone

[Full Day 42 spec](days/day_42.md)

**Compute:** `Modal L4`

> **v2 STATUS: CORE — Gate 5 is evidence-based.** Extend the one app; preserve
> explicit blocked precision status rather than claiming unmeasured optimization.

#### Learn
- Review fine tuning, augmentation, RNNT, and calibration.

#### Build in MendSpeech
- Extend `app/audio_lab.py` to switch between base and adapted recognizer;
  retain the existing ASR/streaming/policy controls rather than create another app.
- Show clean WER, damaged WER, confidence calibration, and repair percentage.
- Show model/policy versions, raw versus calibrated confidence, validation-selected
  thresholds, safe action/reason codes and Day 40 measured/blocked precision
  status. Do not offer an unavailable export as if it were implemented.

#### Experiment and Measure
- Run one fixed benchmark suite and freeze results for Week 8 comparisons.
- Keep matched clean/raw-damaged controls and separate optional Indic extension
  results from the immutable core test set. Report regression, negative outcomes,
  inspect/abstain behavior, adaptation provenance and actual L4 configurations.

#### Required Output
- `app/audio_lab.py`
- `results/week6_frozen_baseline.csv`
- `reports/week6_training.md`

#### Completion Check
> **Definition of Done for Day 42:**
> The one app and report demonstrate measured adaptation and calibration results,
including clean regression or a negative result. Precision tradeoffs are supported
by controlled L4 measurements or explicitly blocked, never falsely completed.

---

## Week 7: TTS, Speaker Preservation, and Boundary Matched Reconstruction

[Week 7 guide](Week_7_MendSpeech_Daily_Plan.md)

### Day 43: TTS system anatomy

[Full Day 43 spec](days/day_43.md)

**Compute:** `Modal L4`

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
> **Definition of Done for Day 43:**
> You can explain the selected text-to-waveform path and speaker conditioning,
> and `docs/tts_pipeline.md` records checked licenses, data/split provenance,
> exact trainable parameters, L4 pilot evidence or a blocking reason, cost
> bounds, and a feasible or deferred adaptation decision. No training success
> is claimed by this gate; unresolved fields remain explicitly unverified.

### Day 44: Selected-stack duration and prosody

[Full Day 44 spec](days/day_44.md)

**Compute:** `Modal L4`

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
> **Definition of Done for Day 44:**
> You can explain short-span timing constraints using measured selected-stack
> behavior, distinguish native controls from DSP, and label unsupported controls.

### Day 45: Vocoder realism and acoustic boundary diagnostics

[Full Day 45 spec](days/day_45.md)

**Compute:** `Modal L4`

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
> **Definition of Done for Day 45:**
> You can separate acoustic model errors from vocoder artifacts and quantify at least
two causes of an audible seam.

### Day 46: Bounded TTS adaptation — base versus adapted

[Full Day 46 spec](days/day_46.md)

**Compute:** `Modal L4`

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
> **Definition of Done for Day 46:**
> Either one bounded base-versus-adapted experiment has reproducible held-out
> evidence (including a valid null or worse result), or training is explicitly
> deferred with its blocking evidence. Feasibility-only work does not satisfy
> training completion and cannot be described as measured adaptation.

### Day 47: Speaker representation and preservation

[Full Day 47 spec](days/day_47.md)

**Compute:** `Modal L4`

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
> **Definition of Done for Day 47:**
> You can discuss speaker similarity measurements and their limitations without
claiming identity preservation from listening alone.

### Day 48: Selective reconstruction with boundary matched stitching

[Full Day 48 spec](days/day_48.md)

**Compute:** `Modal L4 plus local CPU for stitching`

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
> **Definition of Done for Day 48:**
> Predicted-text runs preserve samples outside declared repair/crossfade bounds,
> and seam metrics plus blinded checks compare matched and naive stitching on
> identical spans. Measured non-improvement is valid; oracle-only performance
> cannot satisfy the normal end-to-end check.

### Day 49: Week 7 MendSpeech V1 cascaded repair milestone

[Full Day 49 spec](days/day_49.md)

**Compute:** `Modal L4`

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
> **Definition of Done for Day 49:**
> MendSpeech V1 has tested abstention in the one app and measured predicted-text
> repair/seam evidence. Strengths, failures, deferred adaptation, and live versus
> simulated execution are documented; there is no deadline-based completion.

---

## Week 8: Research Capstone: Controlled Repair Comparisons

[Week 8 guide](Week_8_MendSpeech_Daily_Plan.md)

### Day 50: Freeze research questions and baselines

[Full Day 50 spec](days/day_50.md)

**Compute:** `Local CPU for planning, Modal L4 for
dry run`

> **v2 STATUS: CORE — freeze evidence and capability limits, not a calendar.** One external restoration comparator at most; consume Week 2's bounded feasibility decision.

#### Learn
- Primary question: can selective semantic repair improve intelligibility while preserving more original speech than full resynthesis?
- Secondary question: can uncertainty guided context allocation improve the latency versus accuracy operating point?
- Architecture question: on supported conditions, how does cascaded ASR plus
  TTS compare with the one selected pretrained direct restoration comparator?
  Denoising/enhancement is not evidence of mask-aware missing-span inpainting.
- Scope every claim to the frozen benchmark scale (≥30 utterances, ≥5
  speakers, typically ~5 at this lab) and state the statistical caveat
  explicitly — do not claim population-level generalization.
- Define null outcomes, failure criteria, and claims you will not make.

#### Build in MendSpeech
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

#### Experiment and Measure
- Run a tiny dry run to verify every required field has a measurement or
  explicit status/reason. Unsupported/deferred conditions have missing metrics,
  not fabricated zeros; they are excluded from measured rankings and plots.

#### Required Output
- `experiments/capstone_protocol.md`
- `configs/capstone_frozen.yaml`
- `docs/baseline_definitions.md`

#### Completion Check
> **Definition of Done for Day 50:**
> Another engineer can reproduce the supported comparisons and distinguish
> selected from tested support, external/inpainting deferral, oracle diagnostics,
> and live versus simulated context results without inventing missing evidence.

### Day 51: Release SpeechDamageBench v1 and freeze evaluation

[Full Day 51 spec](days/day_51.md)

**Compute:** `Local CPU`

#### Learn
- Severity grids.
- Speaker separated evaluation.
- Seed control and deterministic manifests.
- Package versioning and reproducibility.
- Clean regression cases that must remain untouched.

#### Build in MendSpeech
- Finalize the independent SpeechDamageBench package with noise, clipping, bandwidth, dropout, and reverberation presets.
- Generate the frozen test matrix and lock manifest checksums.
- Add an installation command and a one command example that reproduces one benchmark item.

#### Experiment and Measure
- Reinstall the package in a clean environment.
- Regenerate a sample from the manifest and verify its checksum.
- Validate that clean references remain unchanged.

#### Required Output
- `speechdamagebench/`
- `speechdamagebench/README.md`
- `speechdamagebench/CHANGELOG.md`
- `benchmarks/speechdamagebench_manifest.csv`
- `benchmarks/README.md`

#### Completion Check
> **Definition of Done for Day 51:**
> SpeechDamageBench is independently installable, deterministic, versioned, and
usable without MendSpeech.

### Day 52: Run recognition and context ablations

[Full Day 52 spec](days/day_52.md)

**Compute:** `Modal L4, keep hardware fixed`

> **v2 STATUS: CORE — absorbs [Day 53](days/day_53.md).** Run recognition/context and repair/seam ablations together on the frozen harness; evidence, not elapsed sessions, closes the gate.

#### Learn
- Fixed lookahead comparison.
- Adaptive context policy.
- WER, latency, RTF, memory, confidence behavior.

#### Build in MendSpeech
- Run every streaming condition on the exact same benchmark subset.
- Repeat timing runs enough to estimate variance.
- Record GPU type and environment automatically through the Modal runner.
- Record `execution_mode=live|simulated` for every context policy. A live
  adaptive claim requires runtime context changes in the recognizer, not
  cached-output selection, offline scheduling, or an oracle decision rule.
- Keep simulated-policy cost estimates separate from measured latency/RTF;
  do not count cached reuse or a hypothetical context reduction as runtime gains.

#### Experiment and Measure
- Plot WER versus measured latency and mark Pareto efficient live points.
  Label simulated analyses separately with their assumptions; do not mix
  estimates into a measured frontier. Null or worse adaptive outcomes are valid.

#### Required Output
- `results/capstone_streaming.csv`
- `results/streaming_pareto.png`

#### Completion Check
> **Definition of Done for Day 52:**
> You can say whether implemented adaptive context helped, hurt, or made no
> meaningful difference. If only simulation is available, state that limitation
> and defer live runtime claims rather than fabricate speedups.

### Day 53: Run cascaded repair and seam ablations

[Full Day 53 spec](days/day_53.md)

**Compute:** `Modal L4`

> **v2 STATUS: MERGED into [Day 52](days/day_52.md) — single combined ablation session.** Produce all repair/seam controls and artifacts within Day 52; no standalone session.

#### Learn
- Repair threshold.
- Repair span padding.
- Preserve percentage.
- Full resynthesis baseline.
- Boundary energy matching, crossfade choice, and seam artifact rate.

#### Build in MendSpeech
- Run Preserve, Balanced, Rescue, full resynthesis, naive selective stitching, and boundary matched selective stitching.
- Record original waveform retained, repair percentage, end to end latency, speaker similarity proxy, and seam metrics.

#### Experiment and Measure
- Test whether repairing more audio always helps intelligibility.
- Test whether boundary matching reduces seam artifacts without materially increasing latency.
- Keep recognition outputs fixed for the stitching comparison so only the repair method changes.

#### Required Output
- `results/capstone_cascaded_repair.csv`
- `results/repair_tradeoff.png`
- `results/seam_ablation.png`

#### Completion Check
> **Definition of Done for Day 53:**
> You have a defensible result for the cascaded selective repair path and can separate
recognition, reconstruction, and stitching effects.

### Day 54: Capability-scoped direct restoration comparison

[Full Day 54 spec](days/day_54.md)

**Compute:** `Modal L4 for comparisons; local CPU for analysis`

> **v2 STATUS: CORE — one conditional external comparator, no model hunting.** Week 2 feasibility bounds apply; unavailable external restoration/inpainting is explicitly deferred.

#### Learn
- Why text is an information bottleneck for prosody and acoustic continuity.
- Direct audio inpainting in latent or codec token spaces at a conceptual level.
- Fair baseline design when systems have different latency and compute profiles.
- Failure taxonomy across semantic correctness, speaker similarity, prosody, seam quality, and compute.

#### Build in MendSpeech
- Consume the one selected comparator and bounded feasible/deferred decision
  in Week 2's `docs/baseline_install_notes.md`. Selection is not a claim of
  tested support. Do not search for substitutes or train restoration from scratch.
- If feasible, implement only `src/baselines/direct_audio_restore.py` behind
  the shared benchmark interface. Record checkpoint/revision/license, sample
  rate, supported damage conditions, mask support, and whether output changes
  samples outside a requested interval. Do not pass masks unless supported.
  This plan chooses one restoration adapter, not a separate inpainting
  adapter; an inpainting claim requires verified mask-aware
  missing-span reconstruction, not a suggestive filename or denoising output.
- Feed identical supported SpeechDamageBench cases, label out-of-scope
  conditions `unsupported`, and exclude them from aggregate comparisons.
  A model available only outside L4 is not an L4 efficiency comparison;
  defer it rather than silently changing hardware or extending the budget.
- If unavailable, still compare raw damaged audio, full resynthesis, naive
  selective repair, and boundary-matched selective repair. Record external
  restoration/inpainting as `deferred`, not tested; no second project/fallback.
- Reuse the tested `src/controller/abstain.py` from Day 49. Keep abstention
  active when inferred content, conditioning consent, or seam safety is weak.

#### Experiment and Measure
- Compare the four internal paths and only the supported external conditions
  on the same cases. Use predicted text for normal TTS paths; separately label
  oracle text/spans and live versus simulated context policies.
- Select at least ten worst or most revealing cases and inspect them manually.
- In `results/capstone_architecture_compare.csv`, include method/checkpoint,
  corruption, mask capability, text/span source, execution mode, condition
  status, and reason. Measured rows may show improvement, equality, or harm;
  `unsupported`/`deferred` rows have missing metrics, never invented numbers.
- Create a failure casebook and tradeoff plot from measured evidence only.
  If no supported external run exists, title the plot as internal comparisons
  and state the deferral; do not imply both architectures were evaluated.

#### Required Output
- Feasible branch only: `src/baselines/direct_audio_restore.py` (one adapter
  with capability metadata; no placeholder implementation if deferred)
- `results/capstone_architecture_compare.csv`
- `results/capstone_failure_casebook.md`
- `results/architecture_tradeoff.png`
- Reuse, do not postpone: `src/controller/abstain.py` (required by Day 49)

#### Completion Check
> **Definition of Done for Day 54:**
> Supported conditions have reproducible measured comparisons and explicit
> limitations; unsupported conditions are not fabricated. If the comparator
> is unavailable, the internal capstone plus external/inpainting deferral is
> complete, but an external or mask-aware comparison is not claimed as tested.

### Day 55: Write the research report and reproducibility guide

[Full Day 55 spec](days/day_55.md)

**Compute:** `Local CPU`

> **v2 STATUS: MERGED into [Day 56](days/day_56.md).** Write the technical report alongside the single `app/audio_lab.py` demo; completion is evidence-based.

#### Learn
- Abstract, motivation, hypotheses, method, baselines, metrics, results, limitations, ethics, and future work.
- Difference between observation and causal claim.
- How to report a negative or mixed architectural comparison honestly.
- Benchmark scale and its statistical limits: never claim population-level generalization from a ~5-speaker lab set.

#### Build in MendSpeech
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
- Reproduce the existing `app/audio_lab.py`; do not introduce a second app.
- Include plots with captions that state what changed and what stayed fixed.

#### Experiment and Measure
- Audit every major claim against a concrete table, figure, or experiment result.
- Remove or soften any conclusion that is not directly supported by frozen evidence.
- Verify that the report distinguishes measured facts from hypotheses and future work.

#### Required Output
- `REPORT.md`
- `REPRODUCE.md`
- `results/final_figures/`
- `docs/limitations_and_claims.md`

#### Completion Check
> **Definition of Done for Day 55:**
> A technical reader can understand the contribution, the architectural tradeoff, and the
limitations without opening the source code first.

### Day 56: Final product, demo, and clean reproduction

[Full Day 56 spec](days/day_56.md)

**Compute:** `Modal L4 for inference, local CPU for
interface and analysis`

> **v2 STATUS: CORE — absorbs Day 55.** Gate 7 closes on report, artifact, and reproduction evidence, not a date or guaranteed session count.

#### Learn
- Review the complete path from waveform and controlled corruption to streaming encoder, uncertainty, repair policy, cascaded reconstruction, direct audio baseline, and evaluation.

#### Build in MendSpeech
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

#### Experiment and Measure
- Record a concise demo and create a final architecture diagram.
- Reproduce one benchmark end to end from the documented command.
- Verify that every public chart can be regenerated from saved result files.

#### Required Output
- `app/audio_lab.py`
- `README.md`
- `demos/final_demo.mp4`
- `docs/architecture.png`
- `release_notes.md`
- `results/reproduction_check.txt`

#### Completion Check
> **Definition of Done for Day 56:**
> A new user can reproduce MendSpeech and SpeechDamageBench, evaluate the
> internal baselines and any supported external comparison, and distinguish
> measured results from unsupported/deferred capabilities. The one app and
> technical report agree on abstention, consent, and live/simulated labels.

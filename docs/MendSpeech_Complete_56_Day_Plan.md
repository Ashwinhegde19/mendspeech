# MendSpeech Complete 56-Day Plan

> **v3 operational reference.** Compiled from the individual day specs.
> Read the [execution plan](REVISED_EXECUTION_PLAN.md) for scope, phases, and gates.
> Status banners override bodies: merged and dropped days do not create
> sessions or artifact obligations. Day numbers are stable identifiers, not
> calendar deadlines. Archived PDFs are unchanged.

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

#### Learn
- Why acoustic frames outnumber output tokens.
- Encoder outputs, vocabulary logits, and decoding.
- CTC versus transducer versus attention decoder at a high level.

#### Build in MendSpeech
- Run a pretrained ASR model on clean and damaged SpeechDamageBench clips.
- Store transcript, token outputs if available, and timing metadata.
- Add a reusable Modal entry point so the same command can run ASR experiments on an L4 without editing deployment code each day.
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

> **v3 STATUS: CORE — recognition quality.** This is the accuracy axis every later trade-off is measured against.

#### Learn
- Word error rate: substitutions, deletions, insertions.
- Character error rate and when it helps.
- Why WER alone hides error severity.
- Names and numbers as a separate error class.

#### Build in MendSpeech
- Implement or verify WER and CER calculations in `src/metrics/wer.py`.
- Add an error analyzer labelling substitution, deletion, and insertion spans in `src/metrics/wer.py`.
- Add a names-and-numbers extractor so entity errors are counted separately in `src/metrics/wer.py`.

#### Experiment and Measure
- Score clean audio versus every SpeechDamageBench severity.
- Find which corruption type drives deletion errors fastest.
- Write tests for empty references, identical strings, and empty hypotheses in `tests/test_wer.py`.

#### Required Output
['- `src/metrics/wer.py`', '- `tests/test_wer.py`', '- `results/day10_wer_by_damage.csv`', '- `results/day10_error_types.csv`']

#### Completion Check
> **Definition of Done for Day 10:**
> You can compute WER by hand for a short example and explain each error class, and entity errors are reported separately from the blended rate.

### Day 11: Token confidence and where it fails

[Full Day 11 spec](days/day_11.md)

**Compute:** `Local CPU`

> **v3 STATUS: CORE — confidence is a signal, not a truth.** This session exists to find the cases where confidence is confidently wrong.

#### Learn
- Frame softmax probability versus token confidence.
- Why mean confidence hides per-token failures.
- Confident-but-wrong: the failure mode that breaks a confidence-gated system.

#### Build in MendSpeech
- Extract per-token confidence and align it to emitted tokens in `src/asr/confidence.py`.
- Build a word-level confidence timeline aligned to the transcript in `src/asr/confidence.py`.

#### Experiment and Measure
- Compare confidence across clean, noisy, clipped, and dropout audio.
- Collect at least ten confident-but-wrong examples and write them up in `docs/day11_confident_wrong.md`.
- Record the rate at which a fixed confidence threshold would have accepted a wrong token.

#### Required Output
['- `src/asr/confidence.py`', '- `tests/test_confidence.py`', '- `results/day11_confidence_by_damage.csv`', '- `docs/day11_confident_wrong.md`']

#### Completion Check
> **Definition of Done for Day 11:**
> You can state when low confidence is informative, and you have documented concrete cases where high confidence was wrong.

### Day 12: Time alignment and word timestamps

[Full Day 12 spec](days/day_12.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE — timestamps for latency attribution and interface feedback.** Word timing is what lets a streaming UI and a latency report say where time went.

#### Learn
- Frame index to wall-clock mapping.
- Token timestamps versus forced alignment.
- Why timestamps must be validated before they are trusted downstream.

#### Build in MendSpeech
- Map emitted tokens to audio time spans in `src/asr/timestamps.py`.
- Validate timestamps against a synthetic event at a known offset.

#### Experiment and Measure
- Inject dropouts at known offsets and check that surrounding token boundaries stay stable.
- Report timestamp error in milliseconds per corruption type in `results/day12_timestamp_error.csv`.
- Confirm that a decoding change does not silently shift timestamps.

#### Required Output
['- `src/asr/timestamps.py`', '- `tests/test_timestamps.py`', '- `results/day12_timestamp_error.csv`']

#### Completion Check
> **Definition of Done for Day 12:**
> Token timestamps are accurate to a stated millisecond tolerance and survive a decoding change, or the failure is documented.

### Day 13: Confidence thresholds and error triage policy

[Full Day 13 spec](days/day_13.md)

**Compute:** `Local CPU`

> **v3 STATUS: CORE — triage, not repair.** Replaces the old repair-policy session. A downstream consumer needs to know accept, low-confidence, or reject; it does not need a synthesis policy.

#### Learn
- Choosing a threshold from validation data rather than test data.
- Risk-coverage: what fraction of traffic a threshold accepts and at what error rate.
- Why a single threshold is a policy decision, not a modelling result.

#### Build in MendSpeech
- Implement three triage policies — accept, low-confidence, reject — in `src/controller/triage.py`.
- Each policy maps confidence to an action with a reason code in `src/controller/triage.py`.

#### Experiment and Measure
- Sweep thresholds on the validation split and plot risk against coverage.
- Report the accepted fraction and the error rate inside the accepted set per corruption type.
- Freeze the chosen thresholds in `configs/triage_thresholds.yaml` using validation only.

#### Required Output
['- `src/controller/triage.py`', '- `tests/test_triage.py`', '- `results/day13_risk_coverage.csv`', '- `configs/triage_thresholds.yaml`']

#### Completion Check
> **Definition of Done for Day 13:**
> Thresholds are chosen from held-out validation evidence and you can state the error rate you accept in exchange for the coverage you keep.

### Day 14: Decoding comparison: greedy, beam, and beam plus LM

[Full Day 14 spec](days/day_14.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Week 2 integration and the accuracy-versus-latency trade-off. A lower WER is not automatically better: a fluent but acoustically wrong transcript is the failure this project must catch.

#### Learn
- Greedy versus beam search: accuracy gained against search cost.
- External n-gram language models: why a plausible transcript can be acoustically wrong.
- Cache reuse for isolating decoder cost from acoustic cost.

#### Build in MendSpeech
- Extend the baseline runner to support greedy, beam, and beam plus one small n-gram LM in `src/asr/decoding.py`.
- Record LM text provenance, normalization, and split roles in `data/lm_text_manifest.csv`; exclude evaluation references and duplicates.
- Freeze one LM order and a small validation-only beam/LM-weight candidate list in `configs/decoding.yaml`.

#### Experiment and Measure
- Report WER/CER and names-and-numbers error for all three decoders on identical held-out cases.
- Collect cases where the LM helped and cases where it hurt; a lower WER does not prove safety.
- Measure decoder-only time on cached acoustic outputs separately from fresh audio-to-transcript latency.

#### Required Output
['- `src/asr/decoding.py`', '- `tests/test_asr_decoding.py`', '- `configs/decoding.yaml`', '- `data/lm_text_manifest.csv`', '- `results/day14_decoding_comparison.csv`', '- `docs/day14_harmful_lm_changes.md`', '- `app/audio_lab.py`']

#### Completion Check
> **Definition of Done for Day 14:**
> All three decoders are measured on the same held-out cases with separated decoder and end-to-end timing, and the LM's harmful changes are documented as first-class evidence.

---

## Week 3: Conformer From First Principles

[Week 3 guide](Week_3_MendSpeech_Daily_Plan.md)

### Day 15: Attention for speech sequences

[Full Day 15 spec](days/day_15.md)

**Compute:** `Local CPU, L4 optional for scaling`

> **v3 STATUS: CORE** Attention cost is why streaming needs a stateful encoder rather than a windowed one.

#### Learn
- Query, key, value projections.
- Scaled dot product attention.
- Attention masks.
- Quadratic cost in sequence length and why long-form audio suffers.

#### Build in MendSpeech
- Implement single-head then multi-head attention in `src/models/attention.py`.
- Add shape assertions and gradient tests in `tests/test_attention.py`.

#### Experiment and Measure
- Change sequence length and measure forward time and peak memory.
- Plot the quadratic cost you predicted against the cost you measured.

#### Required Output
['- `src/models/attention.py`', '- `tests/test_attention.py`', '- `results/day15_attention_cost.csv`']

#### Completion Check
> **Definition of Done for Day 15:**
> You can derive every major tensor shape from memory and explain the quadratic term streaming must avoid.

### Day 16: Conformer convolution module

[Full Day 16 spec](days/day_16.md)

**Compute:** `Local CPU`

> **v3 STATUS: CORE** Depthwise convolution is what makes a Conformer cheaper than attention alone at long sequence lengths.

#### Learn
- Depthwise separable convolution.
- Receptive field and locality.
- Causality assumptions for a streaming encoder.

#### Build in MendSpeech
- Implement a Conformer-style convolution module in `src/models/conv_module.py`.
- Test causality assumptions and receptive field growth in `tests/test_conv_module.py`.

#### Experiment and Measure
- Feed synthetic impulses and inspect how local information spreads.
- Measure receptive field against layer count and compare with the analytic prediction.

#### Required Output
['- `src/models/conv_module.py`', '- `tests/test_conv_module.py`', '- `results/day16_receptive_field.csv`']

#### Completion Check
> **Definition of Done for Day 16:**
> You can explain why depthwise convolution is cheap and what local context it captures relative to attention.

### Day 17: Macaron feed forward and residual scaling

[Full Day 17 spec](days/day_17.md)

**Compute:** `Local CPU`

> **v3 STATUS: LEARN-ONLY** No build session; the macaron build moves into Day 18.

#### Learn
- Macaron structure.
- Layer normalization placement.
- Residual scaling and why half-step helps deep stacks.

#### Build in MendSpeech
- No standalone build. Day 18 implements the macaron feed-forward.

#### Experiment and Measure
- No standalone measurement. Day 18 compares output statistics with and without residual scaling.

#### Required Output
['None for this learn-only session; artifacts are produced within Day 18.']

#### Completion Check
> **Definition of Done for Day 17:**
> You can explain the ordering of the Conformer block without memorizing a diagram.

### Day 18: Assemble one Conformer block

[Full Day 18 spec](days/day_18.md)

**Compute:** `Local CPU`

> **v3 STATUS: CORE** Absorbs Days 17 and 19: one tested block, with real log-Mel features passing through it.

#### Learn
- Macaron structure.
- Attention plus convolution interaction.
- Input projection, padding masks, and temporal dimensions for real log-Mel features.

#### Build in MendSpeech
- Implement the macaron feed-forward and half-step residual wrapper in `src/models/conformer_ffn.py`.
- Assemble feed-forward, attention, convolution, second feed-forward, and normalization in `src/models/conformer_block.py`.
- Add log-Mel input projection and propagate padding masks; assert input, output, and valid-length shapes.

#### Experiment and Measure
- Run forward and backward tests on several sequence lengths.
- Trace a real log-Mel tensor through projection and every stage; assert finite, nonzero gradients.
- Compare output statistics with and without half-step residual scaling.

#### Required Output
['- `src/models/conformer_ffn.py`', '- `tests/test_conformer_ffn.py`', '- `src/models/conformer_block.py`', '- `tests/test_conformer_block.py`', '- `docs/conformer_block_walkthrough.md`', '- `results/day19_shape_trace.md`']

#### Completion Check
> **Definition of Done for Day 18:**
> You can point to every operation in one tested block and say why it exists, and a real log-Mel tensor passes through it with valid gradients.

### Day 19: Real-log-Mel block validation

[Full Day 19 spec](days/day_19.md)

**Compute:** `Local CPU — within Day 18`

> **v3 STATUS: MERGED** into Day 18. No standalone session.

#### Learn
- Input projection and mask propagation.

#### Build in MendSpeech
- Validate projection, masks, shapes, and gradients on Day 18's same block; do not build a separate encoder.

#### Experiment and Measure
- No depth sweep. The shape trace in `results/day19_shape_trace.md` is produced within Day 18.

#### Required Output
['- `results/day19_shape_trace.md` — produced in Day 18']

#### Completion Check
> **Definition of Done for Day 19:**
> A real log-Mel tensor passes through the block with documented shapes and valid gradients.

### Day 20: Compare your block with a production implementation

[Full Day 20 spec](days/day_20.md)

**Compute:** `Local CPU`

> **v3 STATUS: DROPPED** Reading assignment only; do not schedule a session.

#### Learn
- Production Conformer code structure.

#### Build in MendSpeech
- No build. Optional reading time only.

#### Experiment and Measure
- No measurement.

#### Required Output
['None.']

#### Completion Check
> **Definition of Done for Day 20:**
> You have skimmed a production implementation and can name what your scratch block omits.

### Day 21: Architecture review: what dominates streaming latency

[Full Day 21 spec](days/day_21.md)

**Compute:** `Local CPU`

> **v3 STATUS: CORE** The review now asks a systems question rather than a memorization question.

#### Learn
- Review attention, convolution, feed-forward, normalization, and residual paths.
- Which operations scale with sequence length, and which are constant per chunk.

#### Build in MendSpeech
- Write `reports/week3_conformer.md` naming the operations that dominate streaming cost, using the shape trace as evidence.
- No separate inspector UI; the report is the artifact.

#### Experiment and Measure
- Give a ten-minute whiteboard explanation from waveform features through one block to a latency claim.

#### Required Output
['- `reports/week3_conformer.md`', '- `results/day19_shape_trace.md`']

#### Completion Check
> **Definition of Done for Day 21:**
> You can explain which parts are local, which are global, and which become the bottleneck under streaming.

---

## Week 4: FastConformer and Efficient Encoder Behavior

[Week 4 guide](Week_4_MendSpeech_Daily_Plan.md)

### Day 22: Why FastConformer exists

[Full Day 22 spec](days/day_22.md)

**Compute:** `Local CPU`

> **v3 STATUS: LEARN-ONLY** Merged into Day 23; paper notes and compute estimates only.

#### Learn
- Sequence length as an attention cost driver.
- Subsampling before expensive encoder blocks.
- Depthwise separable convolution.
- Local and limited context attention.

#### Build in MendSpeech
- No standalone build. Day 23 incorporates the comparison checklist and diagram.

#### Experiment and Measure
- Day 23 incorporates attention-matrix estimates before and after temporal subsampling.

#### Required Output
['None for this learn-only session; retained notes and estimate paths are produced within Day 23.']

#### Completion Check
> **Definition of Done for Day 22:**
> You can explain FastConformer as a set of concrete efficiency choices, not just a faster model name.

### Day 23: Temporal subsampling experiment

[Full Day 23 spec](days/day_23.md)

**Compute:** `Modal L4 useful`

> **v3 STATUS: CORE** Absorbs Day 22: quantify how much sequence length subsampling removes, and what that saves.

#### Learn
- Convolutional subsampling.
- Temporal resolution.
- Information loss versus compute reduction.

#### Build in MendSpeech
- Implement a small subsampling front end in `src/models/subsampling.py`.
- Track frames per second before and after each stage.
- Incorporate Day 22's comparison checklist into `docs/day22_fastconformer_notes.md`.

#### Experiment and Measure
- Compare 2x, 4x, and 8x temporal reduction on tensor length, runtime, and rough output behavior.
- Estimate attention-matrix size before and after subsampling.

#### Required Output
['- `src/models/subsampling.py`', '- `results/day23_subsampling.csv`', '- `docs/day22_fastconformer_notes.md`', '- `results/day22_compute_estimates.csv`']

#### Completion Check
> **Definition of Done for Day 23:**
> You can quantify how subsampling changes sequence length and downstream attention cost.

### Day 24: Pretrained streaming ASR baseline and capability check

[Full Day 24 spec](days/day_24.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Record what the selected checkpoint actually supports before later phases depend on it.

#### Learn
- Model checkpoint loading.
- Tokenizer and decoder configuration.
- Batch versus single utterance inference.
- Cache-aware inference, right-context controls, export support, and tokenizer language coverage.

#### Build in MendSpeech
- Run a current streaming-capable ASR checkpoint on clean and damaged sets in `src/asr/streaming_runner.py`.
- Record model revision and all inference settings.
- Record capability status for cache-aware inference, supported right-context values and units, runtime context switching, intended export path, and language coverage in `configs/model_baseline.yaml`.

#### Experiment and Measure
- Benchmark WER, latency, and GPU memory by damage type.
- Use minimal supported smoke checks where feasible and record failures early.
- Estimate later-phase effort from the capability record; do not start a model hunt to fill a gap.

#### Required Output
['- `src/asr/streaming_runner.py`', '- `results/day24_baseline.csv`', '- `configs/model_baseline.yaml`']

#### Completion Check
> **Definition of Done for Day 24:**
> You have a reproducible baseline with model, data, hardware, and settings fixed, plus an evidence-backed capability record.

### Day 25: Context and lookahead cost

[Full Day 25 spec](days/day_25.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Lookahead is a latency knob; this session measures what it costs and buys.

#### Learn
- Right context versus left context.
- Algorithmic latency versus accuracy.
- Why future context cannot be free.

#### Build in MendSpeech
- Implement fixed left/right context settings in `src/streaming/context.py`.
- Log the context configuration with every result.

#### Experiment and Measure
- Compare at least two supported context settings on the same subset.
- Plot WER against measured algorithmic latency and mark the Pareto-efficient points.
- Report where errors cluster near chunk boundaries.

#### Required Output
['- `src/streaming/context.py`', '- `tests/test_context.py`', '- `results/day25_context_tradeoff.csv`']

#### Completion Check
> **Definition of Done for Day 25:**
> You can explain exactly why future context creates latency, with a measured curve rather than an assertion.

### Day 26: Efficiency benchmark harness

[Full Day 26 spec](days/day_26.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** One harness serves every later measurement. A benchmark built twice is a benchmark you cannot trust.

#### Learn
- Warmup runs.
- Synchronized GPU timing.
- Median and percentile latency.
- Real time factor.
- Peak memory.

#### Build in MendSpeech
- Create one benchmark function used by every later experiment in `src/bench/benchmark_asr.py`.
- Log environment, model, batch, and precision metadata automatically in `src/bench/environment.py`.

#### Experiment and Measure
- Run repeated inference and calculate variance.
- Detect and discard obviously invalid cold start comparisons, reporting what was discarded and why.
- Record the timing boundary explicitly: where measurement starts and ends.

#### Required Output
['- `src/bench/benchmark_asr.py`', '- `src/bench/environment.py`', '- `results/day26_repeatability.csv`']

#### Completion Check
> **Definition of Done for Day 26:**
> Repeated runs produce stable enough numbers to support comparisons, and the timing boundary is documented.

### Day 27: Profiling the streaming model

[Full Day 27 spec](days/day_27.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Phase P4 begins. Measure before optimizing, or you optimize the wrong thing.

#### Learn
- Where time actually goes in a streaming forward pass.
- CPU launch overhead versus GPU compute time.
- Kernel-level versus end-to-end timing.

#### Build in MendSpeech
- Add per-operator profiling to the Day 26 harness in `src/bench/profile_ops.py`.
- Produce a ranked operator table for one fixed configuration.

#### Experiment and Measure
- Rank operators by measured time and separate launch overhead from compute.
- Identify the top three candidates for optimization and state the expected ceiling for each.
- Write the baseline row into `results/day27_operator_profile.csv`.

#### Required Output
['- `src/bench/profile_ops.py`', '- `results/day27_operator_profile.csv`', '- `docs/day27_optimization_targets.md`']

#### Completion Check
> **Definition of Done for Day 27:**
> You can name the top three time consumers with measured evidence and an expected gain for each.

### Day 28: torch.compile and graph capture

[Full Day 28 spec](days/day_28.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** First optimization technique, measured against the Day 27 profile rather than assumed.

#### Learn
- torch.compile: graph capture, fusion, and recompilation triggers.
- Dynamic shapes and why recompilation is silent and expensive.
- CUDA graphs for static-shape workloads.

#### Build in MendSpeech
- Apply torch.compile to the hot path in `src/asr/optimized_runner.py`, pinning shapes to avoid recompilation.
- Add a CUDA-graph fast path only for static-shape inputs in `src/asr/optimized_runner.py`.

#### Experiment and Measure
- Measure WER, p50/p95/p99 latency, RTF, and memory against the Day 27 baseline on identical inputs.
- Record compile time and warmup separately from steady-state latency.
- Verify output parity against the unoptimized path; a speedup with different transcripts is not a speedup.

#### Required Output
['- `src/asr/optimized_runner.py`', '- `tests/test_optimized_parity.py`', '- `results/day28_compile_speedup.csv`']

#### Completion Check
> **Definition of Done for Day 28:**
> You have a measured before/after for compilation with verified output parity, or the failure and its cause documented.

---

## Week 5: Streaming, Cache Aware Inference, and Adaptive Context

[Week 5 guide](Week_5_MendSpeech_Daily_Plan.md)

### Day 29: Batching and throughput

[Full Day 29 spec](days/day_29.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Throughput and latency are different axes; this session keeps them separate.

#### Learn
- Static versus dynamic batching.
- Queueing delay versus service time.
- Why throughput gains can hurt single-stream latency.

#### Build in MendSpeech
- Implement batched inference in `src/serve/batching.py` with a configurable batch policy.
- Expose batch size and queue wait as separately logged quantities.

#### Experiment and Measure
- Sweep batch size and report throughput and per-request latency separately.
- Find the batch size where queueing delay starts to dominate.
- Report RTF at the best throughput point and the latency at the lowest-concurrency point.

#### Required Output
['- `src/serve/batching.py`', '- `results/day29_batch_sweep.csv`', '- `docs/day29_queueing.md`']

#### Completion Check
> **Definition of Done for Day 29:**
> You can state the throughput/latency knee with measured evidence and explain what happens past it.

### Day 30: Quantization: INT8 and FP16

[Full Day 30 spec](days/day_30.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Quantization is the technique most likely to be misreported, so parity comes before speed.

#### Learn
- Dynamic versus static INT8.
- Why static quantization needs a calibration set.
- What quantization can and cannot preserve in a speech model.

#### Build in MendSpeech
- Smoke-check export and precision support on one compatible backend; pin revisions in `docs/day30_quant_notes.md`.
- Verify original-versus-exported parity at equal precision on validation clips before quantizing.
- Apply supported INT8/FP16 variants through `src/asr/quantized_runner.py`.

#### Experiment and Measure
- Measure WER/CER, p50/p95/p99, RTF, and peak memory for baseline and each supported precision.
- Record confidence and logit shifts caused by quantization.
- State explicitly whether a variant is faster. A slower variant is a valid finding.

#### Required Output
['- `src/asr/quantized_runner.py`', '- `docs/day30_quant_notes.md`', '- `results/day30_quantization_tradeoffs.csv`', '- `app/audio_lab.py`']

#### Completion Check
> **Definition of Done for Day 30:**
> You have measured accuracy, latency, and memory for every supported precision, or a documented compatibility blocker. No speedup is claimed without a measurement.

### Day 31: Streaming fast path

[Full Day 31 spec](days/day_31.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Optimized offline inference does not automatically make streaming fast; this session checks.

#### Learn
- State reuse versus recomputation across chunks.
- Where redundant computation remains in a cache-aware encoder.

#### Build in MendSpeech
- Add a streaming-specific fast path in `src/streaming/fast_path.py` reusing Day 30's best variant.
- Assert cached and uncached streaming produce equivalent transcripts.

#### Experiment and Measure
- Measure steady-state per-chunk latency after warmup, separately from the first chunk.
- Report the speedup of the fast path against the Day 30 baseline, or state that it did not help.

#### Required Output
['- `src/streaming/fast_path.py`', '- `tests/test_fast_path_parity.py`', '- `results/day31_fast_path.csv`']

#### Completion Check
> **Definition of Done for Day 31:**
> You can state measured steady-state streaming latency and whether the fast path earned its complexity.

### Day 32: Optimization scorecard

[Full Day 32 spec](days/day_32.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** One table decides what ships. Techniques that did not help are reported with the same prominence.

#### Learn
- Presenting negative results without overclaiming.
- Why an optimization table is a design document, not a log.

#### Build in MendSpeech
- Build the scorecard generator in `src/bench/scorecard.py`.
- Produce the scorecard in `results/day32_optimization_scorecard.csv`.

#### Experiment and Measure
- Tabulate WER, p50/p95/p99, RTF, and memory for baseline and every technique tried.
- Add a `what_did_not_help` section to `docs/day32_optimization_report.md`.
- Select the shipping configuration from the scorecard and justify it on measured grounds.

#### Required Output
['- `src/bench/scorecard.py`', '- `results/day32_optimization_scorecard.csv`', '- `docs/day32_optimization_report.md`']

#### Completion Check
> **Definition of Done for Day 32:**
> You can defend the shipping configuration from a table, including the techniques that failed.

### Day 33: Break the cache on purpose

[Full Day 33 spec](days/day_33.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** A streaming system that silently mishandles state is worse than a slow correct one.

#### Learn
- State continuity.
- Chunk boundary dependencies.
- Cache reset and truncation.

#### Build in MendSpeech
- Add controlled experiments that reset or shorten the cache at chosen boundaries in `src/streaming/cache_stress.py`.

#### Experiment and Measure
- Measure WER changes around the reset point.
- Determine whether errors cluster at boundaries or propagate, and write it up in `results/day33_cache_failures.md`.

#### Required Output
['- `src/streaming/cache_stress.py`', '- `tests/test_cache_reset.py`', '- `results/day33_cache_failures.md`']

#### Completion Check
> **Definition of Done for Day 33:**
> You can explain a concrete failure caused by incorrect state handling and where it appears.

### Day 34: Bounded adaptive-context comparison

[Full Day 34 spec](days/day_34.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** , capability-bounded. Live switching only if the checkpoint supports it; otherwise report the deferral honestly.

#### Learn
- Policy-driven context selection.
- Confidence smoothing.
- Latency budget.
- Stability versus oscillation.

#### Build in MendSpeech
- Implement a capability-guarded policy in `src/controller/adaptive_context.py` that classifies chunks as easy or uncertain.
- Compare at most two supported right-context settings from Day 25.
- Return an explicit unavailable status when fewer than two settings are supported.

#### Experiment and Measure
- Compare fixed-fast, fixed-accurate, and the bounded adaptive policy with an explicit `live`, `simulated`, or `unavailable` status.
- Keep simulated estimates separate from measured latency; do not count cached reuse as a runtime gain.

#### Required Output
['- `src/controller/adaptive_context.py`', '- `results/day34_adaptive_context.csv`']

#### Completion Check
> **Definition of Done for Day 34:**
> You have a measured live comparison, a clearly limited simulated comparison, or an evidence-backed unavailable result.

### Day 35: Endpointing and the VAD baseline

[Full Day 35 spec](days/day_35.md)

**Compute:** `Local CPU plus Modal L4`

> **v3 STATUS: CORE** Add-on A, absorbed here. Endpointing decides when a final answer is sent, so its errors are latency errors.

#### Learn
- Energy versus spectral VAD.
- Onset and offset error in milliseconds.
- False alarms versus missed speech in a streaming setting.

#### Build in MendSpeech
- Implement a deterministic frame-level VAD in `src/vad/baseline.py` with framing, timestamp, and silence tests.
- Compare it with one local reference VAD on identical clean and damaged inputs.

#### Experiment and Measure
- Measure precision, recall, F1, false alarms, missed speech, and onset/offset error in ms.
- Report CPU RTF separately from GPU measurements.
- Carry the measured choice into Day 41 and record it in `results/day35_vad_benchmark.csv`.

#### Required Output
['- `src/vad/baseline.py`', '- `tests/test_vad.py`', '- `results/day35_vad_benchmark.csv`', '- `docs/day35_vad_notes.md`']

#### Completion Check
> **Definition of Done for Day 35:**
> Endpointing error is quantified in milliseconds and the chosen detector's failure modes are documented.

---

## Week 6: Robustness, Fine Tuning, RNNT, and Calibration

[Week 6 guide](Week_6_MendSpeech_Daily_Plan.md)

### Day 36: Training pipeline anatomy

[Full Day 36 spec](days/day_36.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Phase P5 begins. You cannot fine-tune responsibly if you cannot read a loss curve.

#### Learn
- Manifest format.
- Batching variable-duration audio.
- Loss curves.
- Learning rate, validation split, checkpointing.

#### Build in MendSpeech
- Create one reproducible training configuration in `configs/train_smoke.yaml`.
- Implement the training loop in `training/train.py` with checkpointing and validation hooks.

#### Experiment and Measure
- Run a short smoke job and verify the loss decreases.
- Deliberately use a bad learning rate and record the failure signature in `results/day36_training_smoke.csv`.
- Verify checkpoints reload and reproduce the same validation number.

#### Required Output
['- `configs/train_smoke.yaml`', '- `training/train.py`', '- `tests/test_train_loop.py`', '- `results/day36_training_smoke.csv`']

#### Completion Check
> **Definition of Done for Day 36:**
> You can diagnose whether a run is learning, diverging, or overfitting from basic evidence, and a checkpoint reloads reproducibly.

### Day 37: Adaptation dataset and leakage audit

[Full Day 37 spec](days/day_37.md)

**Compute:** `Local CPU`

> **v3 STATUS: CORE** Leaked data makes every later number meaningless, so this session precedes training.

#### Learn
- Train, validation, and test separation.
- Speaker leakage.
- Synthetic corruption sampling.
- Balanced severity distribution.

#### Build in MendSpeech
- Build manifests pairing clean transcripts with corrupted audio for one adaptation experiment in `data/`.
- Preserve the already-frozen test membership and speaker-separated splits.
- Audit source duplicates and speaker leakage, and write the audit to `reports/data_audit.md`.

#### Experiment and Measure
- Prove no source or speaker appears in more than one split, including via corrupted copies.
- Report the severity distribution and correct any imbalance before training.

#### Required Output
['- `data/train_manifest.jsonl`', '- `data/val_manifest.jsonl`', '- `data/test_manifest.jsonl`', '- `reports/data_audit.md`']

#### Completion Check
> **Definition of Done for Day 37:**
> The evaluation set cannot appear in training through clean or corrupted duplicates, and the audit shows how you know.

### Day 38: Fine-tune for damaged-speech robustness

[Full Day 38 spec](days/day_38.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** The personalization experiment: base versus adapted, measured with a clean-speech regression check.

#### Learn
- Transfer learning.
- Frozen versus trainable layers.
- Mixed precision.
- Gradient accumulation.

#### Build in MendSpeech
- Fine-tune the Day 24 checkpoint on the Day 37 dataset using `training/finetune.py`.
- Freeze trainable layers, learning rate, steps, seed, and sampling in `configs/finetune.yaml`.

#### Experiment and Measure
- Compare base and adapted models on the frozen test set, reporting WER per corruption and severity.
- Measure clean-speech regression explicitly; an adaptation that helps damaged speech but harms clean speech is a documented tradeoff, not a win.
- Record actual training time and cost.

#### Required Output
['- `training/finetune.py`', '- `configs/finetune.yaml`', '- `reports/day38_adaptation.md`', '- `results/day38_base_vs_adapted.csv`']

#### Completion Check
> **Definition of Done for Day 38:**
> You can state exactly what improved, what did not, and whether clean speech regressed.

### Day 39: Augmentation ablation

[Full Day 39 spec](days/day_39.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Confound control: augmentation must be separated from extra training time.

#### Learn
- Time masking.
- Frequency masking.
- Data augmentation as invariance training.

#### Build in MendSpeech
- Add one augmentation intervention to a controlled short run in `experiments/specaugment_ablation.py`.

#### Experiment and Measure
- Compare no augmentation versus selected augmentation with the same seed and the same step budget.
- Report whether the gain survives when the extra steps are given to the unaugmented baseline.

#### Required Output
['- `experiments/specaugment_ablation.py`', '- `results/day39_augmentation.csv`']

#### Completion Check
> **Definition of Done for Day 39:**
> You can separate the effect of augmentation from the effect of extra training time.

### Day 40: RL reward design

[Full Day 40 spec](days/day_40.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** The reward must be falsifiable, or the run proves nothing. This is the most novel session in the plan.

#### Learn
- Policy-gradient and PPO intuition for sequence output.
- Reward hacking: what a model does when the reward is exploitable.
- Designing a reward that is falsifiable in advance.

#### Build in MendSpeech
- Define the reward in `src/rl/reward.py`: penalize fluent output that the acoustics do not support.
- Write the falsifiable prediction in `configs/rl.yaml` BEFORE running anything.
- Implement a minimal policy-gradient or PPO-style update in `src/rl/ppo.py`.

#### Experiment and Measure
- Show the reward can be gamed: construct at least one input where a naive reward rewards a wrong transcript.
- Verify the reward is computable offline from cached logits before spending GPU time.
- Unit-test reward components in `tests/test_reward.py`.

#### Required Output
['- `src/rl/reward.py`', '- `src/rl/ppo.py`', '- `configs/rl.yaml`', '- `tests/test_reward.py`', '- `docs/day40_reward_design.md`']

#### Completion Check
> **Definition of Done for Day 40:**
> You have a written falsifiable prediction, a reward shown to be gameable in at least one case, and a tested implementation.

### Day 41: RL post-training run

[Full Day 41 spec](days/day_41.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Base versus fine-tuned versus RL, on the same held-out data. A null result here is still a result.

#### Learn
- Reward/advantage computation.
- KL regularization against the reference model.
- Why RL can degrade a well-calibrated model.

#### Build in MendSpeech
- Run the bounded RL post-training from `training/rl_train.py` using the Day 38 checkpoint as reference.
- Track reward, KL, and held-out WER together; reward rising while WER worsens is the key diagnostic.

#### Experiment and Measure
- Compare base, fine-tuned, and RL variants on held-out data.
- Re-run the Day 13 risk-coverage analysis for the RL model; improved WER does not imply improved triage safety.
- Record total GPU cost against the declared budget in `results/day41_rl_vs_baseline.csv`.

#### Required Output
['- `training/rl_train.py`', '- `results/day41_rl_vs_baseline.csv`', '- `docs/day41_rl_findings.md`', '- `results/day41_risk_coverage.csv`']

#### Completion Check
> **Definition of Done for Day 41:**
> You can state whether RL helped, did nothing, or hurt, with evidence, and you checked triage safety rather than WER alone.

### Day 42: Personalization comparison and robustness milestone

[Full Day 42 spec](days/day_42.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** One table answers the personalization question and closes Phase P5.

#### Learn
- Separating adaptation effects from training-time effects.
- Reporting a null result without overclaiming.

#### Build in MendSpeech
- Produce the final base/fine-tuned/RL comparison in `reports/day42_personalization.md`.
- Extend `app/audio_lab.py` to switch between the base, fine-tuned, and RL checkpoints.

#### Experiment and Measure
- Report WER per corruption and severity for all three checkpoints, plus clean-speech regression.
- Report the risk-coverage curve for each checkpoint.
- State plainly which checkpoint ships and why the choice rests on measured evidence.

#### Required Output
['- `reports/day42_personalization.md`', '- `results/day42_personalization_matrix.csv`', '- `app/audio_lab.py`']

#### Completion Check
> **Definition of Done for Day 42:**
> The personalization question is answered with a table and a shipping recommendation, including any null results.

---

## Week 7: TTS, Speaker Preservation, and Boundary Matched Reconstruction

[Week 7 guide](Week_7_MendSpeech_Daily_Plan.md)

### Day 43: Serving contract and message schema

[Full Day 43 spec](days/day_43.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Phase P6 begins. Design the contract before implementing, or latency semantics get baked in wrong.

#### Learn
- WebSocket message schemas for streaming audio and incremental transcripts.
- What belongs in a partial result versus a final result.
- Backpressure semantics at the protocol level.

#### Build in MendSpeech
- Define the WebSocket message schema in `src/serve/schema.py`: audio chunks in, partial and final transcripts with confidence and latency out.
- Define timeout, disconnect, and cancellation behaviour in `src/serve/schema.py`.

#### Experiment and Measure
- Write the contract as a testable specification in `docs/day43_serving_contract.md`.
- Verify the schema round-trips in `tests/test_serve_schema.py`.

#### Required Output
['- `src/serve/schema.py`', '- `tests/test_serve_schema.py`', '- `docs/day43_serving_contract.md`']

#### Completion Check
> **Definition of Done for Day 43:**
> The message contract is explicit about partial versus final results, latency fields, and failure semantics.

### Day 44: Async streaming service

[Full Day 44 spec](days/day_44.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** One provider, one endpoint. The service must be measurable, not merely working.

#### Learn
- FastAPI and async WebSocket handling.
- Per-stream state isolation.
- Correct cancellation when a client disconnects mid-utterance.

#### Build in MendSpeech
- Implement the service in `src/serve/app.py` around the optimized Day 32 configuration.
- Containerize reproducibly in `infra/serve/`.

#### Experiment and Measure
- Verify concurrent streams do not share or corrupt cache state.
- Confirm a mid-utterance disconnect leaves no orphaned GPU work.
- Report cold start separately from warm latency.

#### Required Output
['- `src/serve/app.py`', '- `tests/test_serve_isolation.py`', '- `infra/serve/Dockerfile`', '- `infra/serve/README.md`']

#### Completion Check
> **Definition of Done for Day 44:**
> Concurrent streams are isolated, disconnects are clean, and cold start is reported separately from warm latency.

### Day 45: Load test to saturation

[Full Day 45 spec](days/day_45.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** The knee, not the maximum, is the number that matters.

#### Learn
- Load testing methodology: fixed hardware, fixed input, fixed configuration.
- Saturation behaviour and queue growth.
- Why throughput at saturation is not a user experience.

#### Build in MendSpeech
- Implement the load harness in `src/serve/loadtest.py` with configurable concurrency and fixed input.

#### Experiment and Measure
- Sweep concurrency until latency degrades; report the knee in `results/day45_load_curve.csv`.
- Report per-stream p50/p95/p99, queue depth, and dropped or delayed chunks at each level.
- Reproduce one controlled overload failure and one recovery in `docs/day45_failure_recovery.md`.

#### Required Output
['- `src/serve/loadtest.py`', '- `results/day45_load_curve.csv`', '- `docs/day45_failure_recovery.md`', '- `reports/day45_serving.md`']

#### Completion Check
> **Definition of Done for Day 45:**
> You can name the concurrency knee with measured evidence and show a reproduced failure and recovery.

### Day 46: LLM post-processing stage

[Full Day 46 spec](days/day_46.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** One small pinned model, behind an adapter, so the ASR result stays reproducible without it.

#### Learn
- Time-to-first-token versus full response.
- Streaming versus batched generation.
- Prefix caching and why repeated system context should be free.

#### Build in MendSpeech
- Add one small pinned LLM post-processing adapter in `src/llm/polish.py`; the core ASR path must run without it.
- Pin model, revision, quantization, and prompt template in `configs/llm.yaml`.

#### Experiment and Measure
- Measure TTFT and full-response latency separately.
- Measure prefix-cache hit rate across repeated requests and its effect on TTFT.
- Report quality change on the polished output, not only latency.

#### Required Output
['- `src/llm/polish.py`', '- `configs/llm.yaml`', '- `tests/test_llm_polish.py`', '- `results/day46_llm_latency.csv`']

#### Completion Check
> **Definition of Done for Day 46:**
> The LLM stage is measured for TTFT and full response, and the ASR result is still reproducible with it disabled.

### Day 47: Per-stage latency budget decomposition

[Full Day 47 spec](days/day_47.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** The headline artifact of the whole project. The question is where the time actually goes, not what feels slow.

#### Learn
- Separating queueing, model, decoding, network, and serialization time.
- Why a blended average hides the tail that users feel.

#### Build in MendSpeech
- Instrument every stage in `src/bench/budget.py` using the Day 26 harness conventions.

#### Experiment and Measure
- Decompose waveform-to-polished-text into VAD/endpointing, ASR, decode, LLM TTFT, LLM full response, and network/serialization.
- Report p50/p95/p99 per stage in `results/day47_latency_budget.csv`.
- Name the single stage that owns the p99 and state the largest available optimization target in `docs/day47_latency_budget.md`.

#### Required Output
['- `src/bench/budget.py`', '- `results/day47_latency_budget.csv`', '- `docs/day47_latency_budget.md`', '- `results/day47_latency_budget.png`']

#### Completion Check
> **Definition of Done for Day 47:**
> You can point at the stage that owns the tail with per-stage percentiles, and the claim is reproducible from one command.

### Day 48: End-to-end latency optimization round

[Full Day 48 spec](days/day_48.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** One targeted change, chosen by the Day 47 budget, then measured.

#### Learn
- Choosing one optimization from measured evidence rather than preference.
- Verifying that an end-to-end gain is real and not measurement drift.

#### Build in MendSpeech
- Apply the change the Day 47 budget identified as the largest target in `src/`.
- Re-run the full Day 47 decomposition after the change.

#### Experiment and Measure
- Report before/after p50/p95/p99 for the whole pipeline in `results/day48_e2e_optimization.csv`.
- Re-run enough repetitions to separate a real gain from noise.
- If the change did not help, say so and record the negative result.

#### Required Output
['- `results/day48_e2e_optimization.csv`', '- `docs/day48_optimization_outcome.md`', '- `app/audio_lab.py`']

#### Completion Check
> **Definition of Done for Day 48:**
> You have a measured end-to-end before/after, or a documented negative result with evidence.

### Day 49: Interim review

[Full Day 49 spec](days/day_49.md)

**Compute:** `Local CPU`

> **v3 STATUS: DROPPED** in v3. Content absorbed into Days 47 and 48; the latency budget and the optimization outcome now serve as the review.

#### Learn
- No new material; this slot was the old repair milestone.

#### Build in MendSpeech
- No build.

#### Experiment and Measure
- No measurement.

#### Required Output
['None.']

#### Completion Check
> **Definition of Done for Day 49:**
> This slot is intentionally unused; the review content lives in Days 47 and 48.

---

## Week 8: Research Capstone: Controlled Repair Comparisons

[Week 8 guide](Week_8_MendSpeech_Daily_Plan.md)

### Day 50: Freeze the evaluation protocol

[Full Day 50 spec](days/day_50.md)

**Compute:** `Local CPU with Modal L4 dry run`

> **v3 STATUS: CORE** Phase P7 begins. Freeze before measuring, or the measurement decides the protocol.

#### Learn
- What makes an evaluation protocol reproducible.
- Pre-registering claims so results cannot be reinterpreted afterwards.

#### Build in MendSpeech
- Freeze code, model, and data revisions, hardware, corruption configs, and metrics in `configs/frozen.yaml`.
- Define baselines and claims you will NOT make in `experiments/protocol.md`.
- Define null outcomes and failure criteria in advance.

#### Experiment and Measure
- Run a dry run to confirm every required field has a measurement or an explicit status.
- Scope every claim to the benchmark scale and state the statistical caveat.

#### Required Output
['- `configs/frozen.yaml`', '- `experiments/protocol.md`', '- `docs/day50_protocol.md`']

#### Completion Check
> **Definition of Done for Day 50:**
> Another engineer can reproduce the supported comparisons and knows exactly which claims are out of scope.

### Day 51: Release SpeechDamageBench v1 and freeze the evaluation set

[Full Day 51 spec](days/day_51.md)

**Compute:** `Local CPU`

> **v3 STATUS: CORE** The frozen set is the project's anchor; new experiments get new configs, never a new test set.

#### Learn
- Severity grids.
- Speaker-separated evaluation.
- Seed control and deterministic manifests.
- Package versioning and checksum verification.

#### Build in MendSpeech
- Finalize the standalone package and lock manifest checksums in `benchmarks/`.
- Document a one-command example that reproduces one benchmark item in `speechdamagebench/README.md`.

#### Experiment and Measure
- Reinstall the package in a clean environment.
- Regenerate a sample from the manifest and verify its checksum.
- Verify clean references are byte-identical after regeneration.

#### Required Output
['- `speechdamagebench/CHANGELOG.md`', '- `benchmarks/manifest.csv`', '- `benchmarks/README.md`']

#### Completion Check
> **Definition of Done for Day 51:**
> A clean environment reproduces a benchmark item from the manifest, and clean references are provably unchanged.

### Day 52: Robustness matrix on the frozen set

[Full Day 52 spec](days/day_52.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** The full corruption x severity x decoder grid, on data nobody can now change.

#### Learn
- Why a full matrix beats spot checks.
- Multiple-comparison discipline when slicing results.

#### Build in MendSpeech
- Run the full matrix through the frozen harness in `src/bench/run_matrix.py`.

#### Experiment and Measure
- Report WER/CER and confidence behaviour for every corruption, severity, and decoder combination.
- Identify the corruption/decoder pair with the worst risk-coverage behaviour.
- Repeat enough runs to estimate variance on a representative subset.

#### Required Output
['- `src/bench/run_matrix.py`', '- `results/day52_robustness_matrix.csv`', '- `results/day52_robustness_matrix.png`']

#### Completion Check
> **Definition of Done for Day 52:**
> The complete matrix is measured and the worst cell is identified, with variance estimated on a subset.

### Day 53: Optimization and serving ablations

[Full Day 53 spec](days/day_53.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Fixed inputs, one variable at a time, and Pareto frontiers rather than a single winner.

#### Learn
- Pareto frontiers: when no configuration dominates.
- Holding inputs fixed so comparisons mean something.

#### Build in MendSpeech
- Run every optimization variant and serving configuration on the identical frozen subset in `src/bench/run_ablations.py`.

#### Experiment and Measure
- Plot WER against p99 latency and mark Pareto-efficient points in `results/day53_pareto.png`.
- Report serving configurations separately from model-level optimizations.
- Keep live measurements separate from any simulated estimate.

#### Required Output
['- `src/bench/run_ablations.py`', '- `results/day53_ablations.csv`', '- `results/day53_pareto.png`']

#### Completion Check
> **Definition of Done for Day 53:**
> You can say which configuration to ship and which trade-offs are unavoidable, with measured frontiers.

### Day 54: Personalization comparison on the frozen harness

[Full Day 54 spec](days/day_54.md)

**Compute:** `Modal L4`

> **v3 STATUS: CORE** Base, fine-tuned, and RL, measured once on data frozen before any of them ran.

#### Learn
- Why the final comparison must use the frozen set, not a convenient one.
- Reporting regression as carefully as improvement.

#### Build in MendSpeech
- Run the three checkpoints through the frozen harness in `src/bench/run_personalization.py`.

#### Experiment and Measure
- Report WER, risk-coverage, and clean-speech regression for base, fine-tuned, and RL.
- Report what RL cost in GPU time against what it bought.
- State plainly whether personalization earned its place in the pipeline.

#### Required Output
['- `src/bench/run_personalization.py`', '- `results/day54_personalization_final.csv`', '- `reports/day54_personalization.md`']

#### Completion Check
> **Definition of Done for Day 54:**
> The personalization decision is made on frozen evidence, including the case where it did not pay off.

### Day 55: Technical report and reproduction guide

[Full Day 55 spec](days/day_55.md)

**Compute:** `Local CPU`

> **v3 STATUS: CORE** Every claim points at a table. Every omission appears as a limitation.

#### Learn
- Separating observation from causal claim.
- Reporting a mixed or negative result honestly.
- Why a report nobody can reproduce is not evidence.

#### Build in MendSpeech
- Write `REPORT.md` with exact reproduction commands and environment capture.
- Write `REPRODUCE.md` and verify it from a clean checkout.

#### Experiment and Measure
- Audit every major claim against a concrete table, figure, or experiment.
- Remove or soften any conclusion not directly supported by frozen evidence.
- Add a limitations section listing every blocked or deferred capability.

#### Required Output
['- `REPORT.md`', '- `REPRODUCE.md`', '- `results/final_figures/`', '- `docs/limitations_and_claims.md`']

#### Completion Check
> **Definition of Done for Day 55:**
> A technical reader understands the contribution, the trade-offs, and the limitations without opening the source.

### Day 56: Final demo, clean reproduction, and release

[Full Day 56 spec](days/day_56.md)

**Compute:** `Modal L4 plus local interface`

> **v3 STATUS: CORE** Release gate. The demo must show measured numbers, not a scripted success path.

#### Learn
- Demonstrating a system honestly, including its failure modes.
- Releasing with a stable, reproducible artifact.

#### Build in MendSpeech
- Extend only `app/audio_lab.py` with live or prerecorded audio, partial/final transcripts, confidence, triage actions, and the measured latency budget.
- Reproduce one frozen benchmark from a fresh environment and tag a stable release.

#### Experiment and Measure
- Verify every public chart regenerates from saved result files.
- Demonstrate at least one failure case, not only the success path.
- Confirm the demo's displayed numbers match the committed result files.

#### Required Output
['- `app/audio_lab.py`', '- `REPRODUCE.md`', '- `demos/final_demo.mp4`', '- `docs/architecture.md`']

#### Completion Check
> **Definition of Done for Day 56:**
> A new user can run, evaluate, and reproduce the system, and every number shown traces to a committed artifact.

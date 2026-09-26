# MendSpeech Complete 56-Day Plan

> **Generated reference.** Rebuilt from the day specs by `scripts/plan_docs.py`.
> Read the [execution plan](REVISED_EXECUTION_PLAN.md) for scope, phases, and gates.
> Status banners override bodies: merged and dropped days do not create sessions
> or artifact obligations. Day numbers are specification identifiers, not
> calendar deadlines. Archived PDFs are unchanged.
---
## Week 1: Audio DSP and the deterministic damage suite
[Week 1 guide](Week_1_MendSpeech_Daily_Plan.md)

### Day 01: Waveforms, sampling, and the MendSpeech baseline
[Full Day 01 spec](days/day_01.md)
**Compute:** Local CPU

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

#### Required Output Artifacts
- `notebooks/day01_waveform.ipynb`
- `data/clean_manifest.csv` (now includes a `transcript` column)
- `data/benchmark/` (reference-transcripted corpus slice)
- `docs/audio_baseline_notes.md`

#### Completion Check
> You can explain what is lost when sample rate is reduced and can reproduce the
> same preprocessing from code.

### Day 02: Fourier intuition and STFT
[Full Day 02 spec](days/day_02.md)
**Compute:** Local CPU

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

#### Required Output Artifacts
- `notebooks/day02_stft.ipynb`
- `results/day02_stft_parameter_grid.png`

#### Completion Check
> You can choose a reasonable frame and hop configuration and explain why.

### Day 03: Mel scale and log Mel features
[Full Day 03 spec](days/day_03.md)
**Compute:** Local CPU

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

#### Required Output Artifacts
- `src/audio/features.py`
- `notebooks/day03_logmel.ipynb`
- `tests/test_features.py`

#### Completion Check
> You can trace waveform to STFT to Mel filterbank to log Mel tensor.

### Day 04: Build SpeechDamageBench as a standalone package
[Full Day 04 spec](days/day_04.md)
**Compute:** Local CPU

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

#### Required Output Artifacts
- `speechdamagebench/pyproject.toml` (package name `speechdamagebench`; do not touch repo-root `pyproject.toml`)
- `speechdamagebench/speechdamagebench/audio_damage.py`
- `speechdamagebench/speechdamagebench/presets.py`
- `speechdamagebench/speechdamagebench/cli.py`
- `speechdamagebench/tests/test_determinism.py`
- `data/benchmark/` begun if still missing (manifest with relative paths + transcripts)

#### Completion Check
> Another project can install SpeechDamageBench and regenerate the same damaged
> clip from a manifest entry.

### Day 05: Objective audio measurements
[Full Day 05 spec](days/day_05.md)
**Compute:** Local CPU

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

#### Required Output Artifacts
- `src/metrics/audio_metrics.py`
- `results/week1_damage_metrics.csv`
- `docs/metric_limitations.md`

#### Completion Check
> You can explain what each metric says and what it fails to say.

### Day 06: Build the first MendSpeech audio console
[Full Day 06 spec](days/day_06.md)
**Compute:** Local CPU

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

#### Required Output Artifacts
- `app/audio_lab.py`
- `results/week1_audio_console.png`

#### Completion Check
> Another person can open the tool, damage an utterance, and understand the visual
> change without reading your code.

### Day 07: Review, explain, and freeze Week 1
[Full Day 07 spec](days/day_07.md)
**Compute:** Local CPU

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

#### Required Output Artifacts
- `reports/week1_audio_foundations.md`
- `data/benchmark_manifest.csv` (relative paths, `transcript`, `speaker_id`, and a `split` column: train/val/test)
- `speechdamagebench/README.md`
- `speechdamagebench/VERSION`

#### Completion Check
> You can teach the complete path from clean waveform to controlled corruption and
> feature tensor.

## Week 2: Data protocol, confidence, editor contract, and the early end-to-end baseline
[Week 2 guide](Week_2_MendSpeech_Daily_Plan.md)

### Day 08: Frame sequence to transcript
[Full Day 08 spec](days/day_08.md)
**Compute:** Modal L4 optional, CPU acceptable for
small runs

> **STATUS: CORE**

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

#### Required Output Artifacts
- `src/asr/baseline.py`
- `infra/modal_asr.py`
- `results/day08_baseline_transcripts.csv`
  setup/retry evidence, `feasible` or `deferred`; no overwritten historical results)

#### Completion Check
> You can draw the path from features to encoder states to token probabilities to text,
> and launch the same baseline locally or on Modal with a documented command.
> The bounded comparator check has an evidence-backed status; a documented deferral
> is sufficient for this external branch, but is not successful inpainting.

### Day 09: CTC from first principles
[Full Day 09 spec](days/day_09.md)
**Compute:** Local CPU

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

#### Required Output Artifacts
- `src/asr/ctc_decode.py`
- `tests/test_ctc_decode.py`
- `docs/ctc_explained.md`

#### Completion Check
> You can explain why a blank is needed and correctly decode repeated characters.

### Day 10: WER/CER, data roles, and evaluation protocol
[Full Day 10 spec](days/day_10.md)
**Compute:** Local CPU

> **STATUS: CORE**

#### Learn
- Substitution/deletion/insertion counts, normalization and empty-reference conventions.
- Group-level leakage, repeated corruptions versus independent samples.

#### Build in MendSpeech
- Implement/test scoring and named-entity/number/negation slices without adding a model-based judge.
- Audit frozen benchmark roles and source/speaker separation. Record immutable checksums and separate training, calibration, validation and final-test roles before downstream tuning.
- Declare diagnostic primary metrics and failure thresholds in experiments/protocol.md; new editor data is a separate manifest, not a changed core set.

#### Experiment and Measure
- Hand-check small examples and empty/identical inputs; run validation diagnostics by damage/severity.
- Report actual corpus counts/roles; insufficient training or calibration data blocks training, not permission to reuse test.

#### Required Output Artifacts
- `src/metrics/wer.py`
- `tests/test_wer.py`
- `results/day10_wer_by_damage.csv`
- `results/day10_error_types.csv`
- `reports/data_roles.md`
- `experiments/protocol.md`

#### Completion Check
> Scoring conventions and frozen data roles are explicit and tested; development uses validation, not repeated test selection.

### Day 11: Confidence definitions and failure cases
[Full Day 11 spec](days/day_11.md)
**Compute:** Local CPU

> **STATUS: CORE**

#### Learn
- Acoustic frame score, emitted-token confidence and word correctness are different quantities.
- Blank-heavy averages can hide errors; confidence is not calibrated merely because it lies in [0,1].

#### Build in MendSpeech
- Build token/word score records with model/head/tokenizer provenance; inspect the current baseline averaging semantics rather than trusting its docstring.
- Define label alignment and valid/missing states, with empty/blank/repeat tests.

#### Experiment and Measure
- On validation clips compare score versus correctness by corruption. Collect confident errors when present; report the observed count, never invent a required ten.
- Keep raw softmax and later calibrated probability distinct.

#### Required Output Artifacts
- `src/asr/confidence.py`
- `tests/test_confidence.py`
- `results/day11_confidence_by_damage.csv`
- `docs/day11_confident_wrong.md`

#### Completion Check
> Every score has a tested definition and correctness target; no raw confidence is called calibrated.

### Day 12: Transcript timestamps and correlated event tracing
[Full Day 12 spec](days/day_12.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Audio frame offsets versus wall-clock events; forced-alignment limitations.
- Monotonic clocks, request/stream IDs and event ordering.

#### Build in MendSpeech
- Implement transcript timestamp records plus a shared trace schema; separate word alignment from service-clock events.
- Test synthetic frame-index mapping, real short manually aligned speech, empty output and resampling; a click/tone is not a word-timestamp ground truth.

#### Experiment and Measure
- Report ms error on annotated validation speech and corruption cases with uncertainty; do not demand unchanged word timing when the recognized words differ.
- Round-trip trace IDs and start/end ordering for a replayed request.

#### Required Output Artifacts
- `src/asr/timestamps.py`
- `tests/test_timestamps.py`
- `src/bench/tracing.py`
- `tests/test_tracing.py`
- `results/day12_timestamp_error.csv`

#### Completion Check
> Word timing is validated against relevant annotations and trace clocks are explicit; the two are not conflated.

### Day 13: Conservative editor contract and evaluation data
[Full Day 13 spec](days/day_13.md)
**Compute:** Local CPU

> **STATUS: CORE**

#### Learn
- Formatting versus rewriting and protected content.
- Training/validation/test group splits and proposed versus delivered output.

#### Build in MendSpeech
- Implement/test identity and deterministic whitespace/casing/punctuation baseline under docs/EDITOR_AND_RL_CONTRACT.md. Define protected spans and fail-closed output guard.
- Create data/editor_manifest.jsonl with licensed/consented source groups, annotated allowed outputs and sealed held-out roles. Audit duplicates and instruction-like input.
- Declare pilot/full-set counts and human review protocol; exact wording/negation/numbers cannot be silently changed.

#### Experiment and Measure
- Score identity and deterministic formatting on validation, including already-correct and needs-edit slices.
- Exercise deleted negation, altered entities/numbers, blank/long output, prompt injection and punctuation-induced meaning changes; log manual reviewer limits.

#### Required Output Artifacts
- `src/llm/contracts.py`
- `src/llm/deterministic.py`
- `tests/test_editor_contract.py`
- `data/editor_manifest.jsonl`
- `reports/editor_data_audit.md`
- `results/day13_editor_baselines.csv`

#### Completion Check
> Editing rules and independent quality metrics exist before LLM training; the small pilot and full evaluation data are separated and incomplete annotation is reported.

### Day 14: Greedy, beam and one small LM comparison
[Full Day 14 spec](days/day_14.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- CTC collapse versus transducer search and external LM fusion.
- Decoder-only cached cost versus fresh end-to-end latency.

#### Build in MendSpeech
- Verify the current acoustic checkpoint/head and one supported decoder backend before comparing greedy, beam-only and beam+one n-gram LM. No second acoustic model or custom search engine.
- Record LM corpus license/hash/deduplication and exclude evaluation references; select a small predeclared beam/LM-weight list on validation only.
- Test token/blank/repeat mapping and alignment; initialize the shared repeatable benchmark harness with trace IDs, warm/cold and CPU worker metadata.

#### Experiment and Measure
- On identical validation cases report WER/CER, names/numbers, helpful and harmful changes and cached decoder versus fresh timing.
- Freeze selection for the later final test. Unsupported backend leaves decoding incomplete; independent baseline work may proceed with that blocker.

#### Required Output Artifacts
- `src/asr/decoding.py`
- `tests/test_asr_decoding.py`
- `configs/decoding.yaml`
- `data/lm_text_manifest.csv`
- `src/bench/benchmark_asr.py`
- `src/bench/environment.py`
- `tests/test_benchmark_asr.py`
- `results/day14_decoding_comparison.csv`
- `docs/day14_harmful_lm_changes.md`

#### Completion Check
> Three decoder conditions have controlled evidence and provenance; offline decoding is not claimed as live streaming and LM scores are not calibrated confidence.

## Week 3: Streaming capability, session loop, endpointing, and calibration
[Week 3 guide](Week_3_MendSpeech_Daily_Plan.md)

### Day 15: Early ASR-to-editor baseline and resource pilot
[Full Day 15 spec](days/day_15.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Causal-LM generation, prompt/output token limits and KV memory.
- Server/client TTFT versus completion and validated delivery.

#### Build in MendSpeech
- Select one small causal editor candidate via docs/EDITOR_AND_RL_CONTRACT.md, pin permitted weights/tokenizer/template and isolated optional environment; do not add unverified packages to core dependencies.
- Build src/llm/polish.py and an app/audio_lab.py file/replayed-audio path using the existing ASR, deterministic/identity/LLM choices, guard and bypass. Label it offline/replayed, not completed live streaming.
- Pilot one-L4 ASR+editor co-residency with fixed token caps, serialized work and combined peak memory. ASR text is data, not instructions; no tools or agents.

#### Experiment and Measure
- Measure validation quality of raw/deterministic/prompt-only outputs, fallback, names/numbers/negation and context-free correction risks.
- Capture correlated stage events, server TTFT/completion, startup and combined memory. Record CPU threads/topology; a failed co-residency test requests scope review, not a hidden second GPU.

#### Required Output Artifacts
- `src/llm/polish.py`
- `tests/test_llm_polish.py`
- `configs/llm.yaml`
- `infra/editor/requirements.txt`
- `docs/editor_model_card.md`
- `app/audio_lab.py`
- `results/day15_e2e_baseline.csv`
- `results/day15_resource_pilot.csv`

#### Completion Check
> One real ASR→guarded-editor baseline runs before optimization; quality, TTFT/completion, memory and topology evidence are recorded without a sub-500ms promise.

### Day 16: Early SFT and GRPO feasibility on the text editor
[Full Day 16 spec](days/day_16.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Completion-only SFT, LoRA gradients, group-relative advantages and reference KL.
- Why CTC acoustic outputs cannot be passed to a causal-LM GRPO trainer.

#### Build in MendSpeech
- Use the selected Day15 editor and a pinned compatible optional Transformers/PEFT/TRL environment. Implement training/editor_pilot.py with the exact bounded SFT/GRPO route in the editor contract, no scratch PPO or reward model.
- Implement provisional reward component tests; one pilot group uses fresh current-policy generations. Check names/counts of trainable adapters, completion masks/EOS and group/batch divisibility.
- Declare <=10 SFT and <=5 GRPO steps, group2, fixed token caps, wall-time/spend and nonfinite/OOM stop conditions. Unload ASR while training.

#### Experiment and Measure
- Show finite loss/logprobs/gradients and adapter weight changes, reload checkpoint and inspect completions. Measure policy/reference/optimizer/rollout memory, step time and cost.
- Log reward variance/zero-variance groups and KL; failed update or insufficient rollout diversity is blocked, not a null RL result. Forecast later training cost before approval.

#### Required Output Artifacts
- `training/editor_pilot.py`
- `configs/editor_pilot.yaml`
- `src/rl/reward.py`
- `tests/test_reward.py`
- `reports/day16_training_feasibility.md`
- `results/day16_training_pilot.csv`

#### Completion Check
> An actual tiny SFT and GRPO update on the causal text editor is verified, or the full training track remains explicitly blocked before further training expenditure.

### Day 17: Attention and Conformer architecture reading
[Full Day 17 spec](days/day_17.md)
**Compute:** Local CPU

> **STATUS: LEARN-ONLY**

#### Learn
- Attention tensor shapes, local convolution, residual/normalization placement and context limits.

#### Build in MendSpeech
- No standalone implementation. Use existing tested CTC/audio examples and the model documentation needed for Day18.

#### Experiment and Measure
- Explain state/context costs during Day18; optional scratch exercises do not gate release.

#### Required Output Artifacts
- None; learning is included in the absorbing Day18 estimate.

#### Completion Check
> Concepts support Day18 model selection; no extra artifact or build session is counted.

### Day 18: Streaming checkpoint and decoder capability gate
[Full Day 18 spec](days/day_18.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Cache-aware versus buffered inference, CTC/RNN-T head compatibility and chunk lookahead.

#### Build in MendSpeech
- Select one streaming-capable ASR checkpoint compatible with the planned runtime; record revision/head/tokenizer, cache signatures, chunk/right context and inference/export support in configs/model_baseline.yaml.
- Smoke one utterance offline and via the documented streaming interface; model/decoder changes invalidate old confidence thresholds. Verify the combined Day15 editor memory with the new checkpoint.

#### Experiment and Measure
- Record output/length/context evidence, offline/streaming and LM support separately. Do not rewrite a framework to force missing support.
- Compare supported baseline transcripts on validation and classify dependency blockers before writing optimizations.

#### Required Output Artifacts
- `src/asr/streaming_runner.py`
- `tests/test_streaming_runner.py`
- `configs/model_baseline.yaml`
- `results/day18_capability_check.csv`
- `docs/day18_capabilities.md`

#### Completion Check
> Pinned streaming/cache/head behavior and combined-resource feasibility are verified; unsupported live inference is a blocker, not a filename-based success.

### Day 19: Shape and mask checks inside the streaming capability gate
[Full Day 19 spec](days/day_19.md)
**Compute:** Within Day18

> **STATUS: MERGED**

#### Learn
- Feature length, cache shape, padding and valid output lengths.

#### Build in MendSpeech
- Checks are owned by Day18 tests/test_streaming_runner.py; no separate tiny encoder.

#### Experiment and Measure
- Use Day18 short/long/padded fixture results.

#### Required Output Artifacts
- None; evidence belongs to Day18.

#### Completion Check
> No standalone work; merge is explicit and does not duplicate artifacts.

### Day 20: Scratch production-encoder comparison outside release
[Full Day 20 spec](days/day_20.md)
**Compute:** None

> **STATUS: DROPPED**

#### Learn
- Optional architecture study only.

#### Build in MendSpeech
- No build.

#### Experiment and Measure
- No release experiment.

#### Required Output Artifacts
- None.

#### Completion Check
> Excluded from release scope; not a completion claim.

### Day 21: Correct chunk loop and per-stream state
[Full Day 21 spec](days/day_21.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Chunk clocks, partial/final decoding, cache ownership and last-chunk flush.

#### Build in MendSpeech
- Implement src/streaming/session.py with start/push/finish/reset, bounded state, variable final chunks and sequence validation.
- Add deterministic audio replay preserving input cadence; test isolated/interleaved streams and flush/reset exactly once.

#### Experiment and Measure
- Verify repeated single-stream replay agrees with isolated interleaved streams under identical model context. Offline full-context text need not match streaming text; compare the documented same-context reference.
- Log first partial, final timestamps and state sizes. Future-context dependence cannot be called causal.

#### Required Output Artifacts
- `src/streaming/session.py`
- `tests/test_streaming_session.py`
- `experiments/replay_audio.py`
- `results/day21_streaming_correctness.csv`

#### Completion Check
> An explicit tested chunk/session loop exists before profiling, with valid finals and no cross-stream state leakage.

## Week 4: Context trade-off, calibration, triage, and the end-to-end latency baseline
[Week 4 guide](Week_4_MendSpeech_Daily_Plan.md)

### Day 22: Temporal subsampling and context reasoning
[Full Day 22 spec](days/day_22.md)
**Compute:** Local CPU

> **STATUS: LEARN-ONLY**

#### Learn
- Subsampling changes frame counts; buffering/lookahead creates latency.

#### Build in MendSpeech
- No separate module; study the selected model metadata for Day24.

#### Experiment and Measure
- No standalone timing sweep.

#### Required Output Artifacts
- None.

#### Completion Check
> Theory informs context measurements without a second encoder project.

### Day 23: VAD, endpointing and finalization state machine
[Full Day 23 spec](days/day_23.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Frame energy/spectral VAD, hangover, pause/endpoint trade-offs and speech-end labels.

#### Build in MendSpeech
- Implement deterministic framing/VAD with one compatible local reference detector; tune thresholds only on validation.
- Integrate start/end/hangover state with streaming/session.py. Preserve too-quiet/short/silence failure cases; no diarization.

#### Experiment and Measure
- Measure precision/recall, onset/offset ms error and CPU RTF on a separate annotated validation slice; untouched core-test membership.
- Test mid-sentence pauses, trailing silence, short clips and delayed chunks; record endpoint-to-final and annotated-speech-end-to-final separately.

#### Required Output Artifacts
- `src/vad/baseline.py`
- `src/streaming/endpoint.py`
- `tests/test_vad.py`
- `tests/test_endpoint.py`
- `results/day23_endpointing.csv`

#### Completion Check
> Live finalization is tested and endpoint delay is measured, not postponed until after system profiling.

### Day 24: Fixed-context quality and latency frontier
[Full Day 24 spec](days/day_24.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Algorithmic lookahead, wall-clock finalization and WER versus responsiveness.

#### Build in MendSpeech
- Add only documented fixed context configurations and preserve model/head/cache compatibility.
- Use the shared replay/harness with trace IDs, warm/cold and settings recorded.

#### Experiment and Measure
- Compare supported context settings on validation at fixed decoder and batch; report time-to-first-partial and post-utterance finalization plus WER/CER.
- Fewer than two supported settings yields a single-setting baseline and explicit limit, not invented adaptive gains.

#### Required Output Artifacts
- `src/streaming/context.py`
- `tests/test_context.py`
- `results/day24_context_tradeoff.csv`

#### Completion Check
> Context costs are measured with a working chunk loop and endpoint detector; scope of supported settings is honest.

### Day 25: Fit and validate calibrated confidence and triage
[Full Day 25 spec](days/day_25.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Temperature scaling, word/utterance correctness labels, reliability, Brier/ECE, risk-coverage.

#### Build in MendSpeech
- Implement src/asr/calibration.py with an explicit score-to-correctness unit, calibration-only fitting and separate validation thresholds.
- Map scores to actual decoded tokens/words; if beam scores lack valid aligned features, retain verified greedy triage rather than calling fused scores probabilities.
- Bind calibration to checkpoint/head/tokenizer/decoder/precision and context mode; build accept/uncertain/reject policy that preserves raw text and bypasses editor on uncertainty.

#### Experiment and Measure
- Compare raw/calibrated reliability and risk-coverage on disjoint validation, with bins/counts and corruption slices.
- Test empty/blank/all-correct/all-wrong and serialization/refit; non-improvement is valid but unsupported calibrated claims are not.

#### Required Output Artifacts
- `src/asr/calibration.py`
- `src/controller/triage.py`
- `tests/test_calibration.py`
- `tests/test_triage.py`
- `configs/calibration.yaml`
- `configs/triage_thresholds.yaml`
- `results/day25_reliability.csv`
- `results/day25_reliability.png`
- `results/day25_risk_coverage.csv`

#### Completion Check
> Calibration is actually fitted and independently evaluated, distinct from threshold selection, for the exact shipping-candidate configuration.

### Day 26: Early streaming-to-editor integration and budget
[Full Day 26 spec](days/day_26.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- End-of-speech finalization, editor trigger semantics and guard-before-delivery.

#### Build in MendSpeech
- Extend one app/audio_lab.py with replay/microphone streaming, final-ASR→editor request, raw bypass and visibly provisional output.
- Instrument all clocks in the latency contract; execute one-L4 joint-memory/interference test, not isolated-model timing only.

#### Experiment and Measure
- Collect raw/deterministic/LLM quality and correlated request-level stage intervals on validation.
- Record first partial, post-utterance ASR finalization, server/client TTFT and final usable text. State microphone versus paced replay and model residence/cold state.

#### Required Output Artifacts
- `app/audio_lab.py`
- `tests/test_pipeline_contract.py`
- `results/day26_e2e_baseline.csv`
- `docs/day26_latency_baseline.md`

#### Completion Check
> The full baseline, including streaming, endpointing, guarded editing and combined resource use, exists before optimization.

### Day 27: Profile the current end-to-end path
[Full Day 27 spec](days/day_27.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Kernel versus wall time, CPU launch overhead, queue/prefill/decode/generation and cold-versus-warm.

#### Build in MendSpeech
- Add per-stage instrumentation to src/bench/profile_ops.py on the Day26 pipeline; no optimization technique is applied yet.
- Separate start/compile/load from steady state; report per-request critical paths, not just aggregated percentiles.

#### Experiment and Measure
- Rank measured time contributors and note overlapping/serialized stages. Identify where p99 requests diverge from median, using the same request IDs.
- State expected ceiling for each candidate; unsupported profiling granularity is recorded, not guessed.

#### Required Output Artifacts
- `src/bench/profile_ops.py`
- `results/day27_profile.csv`
- `docs/day27_optimization_targets.md`

#### Completion Check
> The top measured contributors and the p99 divergence are named with request-level evidence, not assumption.

### Day 28: torch.compile and graph capture experiment
[Full Day 28 spec](days/day_28.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Graph capture, fusion, recompilation on new shapes/frames and static versus dynamic cost.

#### Build in MendSpeech
- Apply torch.compile to a pinned steady-state configuration and experiment with CUDA graphs in src/asr/optimized_runner.py. Pin static shapes; treat chunked streaming as dynamic.
- Test parity on same input/precision; separate warmup/compile time from p50/p95/p99.

#### Experiment and Measure
- Measure WER/latency/RTF/memory against Day27 including joint editor; count recompiles and reject measures that slow p99 or break parity.
- Negative/zero gains are documented with numbers; do not chase a checklist.

#### Required Output Artifacts
- `src/asr/optimized_runner.py`
- `tests/test_optimized_parity.py`
- `results/day28_compile.csv`

#### Completion Check
> A measured, parity-checked before/after for compilation/graph capture, or an honest zero-gain result.

## Week 5: Profiling, compile/graph capture, batching, precision, and the scorecard
[Week 5 guide](Week_5_MendSpeech_Daily_Plan.md)

### Day 29: Batching, concurrency and queueing
[Full Day 29 spec](days/day_29.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Static versus dynamic batching, queue wait versus service time, batch-size latency/throughput knee.

#### Build in MendSpeech
- Add bounded batching and per-stream queues in src/serve/batching.py; keep the co-residency pilot constraint in force.

#### Experiment and Measure
- Sweep batch size/concurrency within budget; report throughput, per-stream p50/p95/p99, queue wait, achieved concurrency and memory.
- Verify no cross-stream state contamination and report starvation/fairness; batching is not required to be selected.

#### Required Output Artifacts
- `src/serve/batching.py`
- `tests/test_batching.py`
- `results/day29_batch_sweep.csv`
- `docs/day29_queueing.md`

#### Completion Check
> Throughput and single-stream latency are reported separately with the knee and fairness behavior.

### Day 30: Precision and export parity
[Full Day 30 spec](days/day_30.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- FP16/INT8 (dynamic and static), calibration sets, exported-versus-original parity.

#### Build in MendSpeech
- Smoke-check export/precision support for the selected backend in an isolated pinned environment; document in docs/day30_quant_notes.md.
- Verify original-versus-exported parity first; then apply supported FP16/INT8 with a calibration slice from training/calibration only, never the frozen test.

#### Experiment and Measure
- Measure WER/CER, latency percentiles, RTF, memory and confidence/logit shifts per precision; recheck Day25 calibration binding.
- State explicitly if a precision is slower or degrades accuracy; blocked precision leaves the required comparison incomplete with recorded blocker.

#### Required Output Artifacts
- `src/asr/quantized_runner.py`
- `docs/day30_quant_notes.md`
- `results/day30_quantization_tradeoffs.csv`
- `app/audio_lab.py`

#### Completion Check
> Parity is verified before any precision claim, and every supported precision has measured accuracy/latency/memory or a documented blocker.

### Day 31: Streaming fast path and cache-failure evidence
[Full Day 31 spec](days/day_31.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- State reuse versus recomputation across chunks and reset/truncation failure modes.

#### Build in MendSpeech
- Add a streaming fast path in src/streaming/fast_path.py reusing the best supported variant, with state equivalence tests.
- Break the cache deliberately at chosen boundaries via src/streaming/cache_stress.py to characterize failure.

#### Experiment and Measure
- Report steady-state per-chunk latency separate from first chunk; verify cached vs uncached transcripts.
- Record WER changes and whether errors cluster or propagate at reset points; this failure evidence is required release material.

#### Required Output Artifacts
- `src/streaming/fast_path.py`
- `tests/test_fast_path_parity.py`
- `src/streaming/cache_stress.py`
- `results/day31_cache_failures.md`
- `results/day31_fast_path.csv`

#### Completion Check
> A measured streaming fast path with parity and a concrete, reproducible cache-state failure.

### Day 32: Optimization scorecard and selection
[Full Day 32 spec](days/day_32.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Multi-objective selection, Pareto frontiers and honest negative reporting.

#### Build in MendSpeech
- Build the scorecard generator in src/bench/scorecard.py over all measured variants.
- Record a what_did_not_help section; select one shipping candidate and mark the selection provisional pending final-stack revalidation.

#### Experiment and Measure
- Tabulate WER, latency percentiles, RTF, memory and calibration status per variant.
- Select on measured grounds and identify the largest remaining bottleneck for the end-to-end path.

#### Required Output Artifacts
- `src/bench/scorecard.py`
- `results/day32_optimization_scorecard.csv`
- `docs/day32_optimization_report.md`

#### Completion Check
> A defensible provisional shipping configuration with Pareto evidence including the techniques that failed.

### Day 33: Rebuild and revalidate the end-to-end pipeline
[Full Day 33 spec](days/day_33.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Provisional optimization results must be re-checked after pipeline integration.

#### Build in MendSpeech
- Rebuild src/streaming, src/serve and app/audio_lab.py around the selected candidate with the Day25 calibration and guard contract intact.
- Re-verify streaming parity, confidence binding and the joint memory/interference pilot on the selected stack.

#### Experiment and Measure
- Re-measure Day26 baseline quality/latency/memory end-to-end; deltas vs provisional are explained.
- If the candidate is infeasible jointly, record the blocker and scope-review options instead of forcing the stack.

#### Required Output Artifacts
- `results/day33_rebuild_revalidation.csv`
- `docs/day33_rebuild_notes.md`

#### Completion Check
> The optimized end-to-end pipeline is revalidated, and any change in behavior versus the provisional baseline is explained.

### Day 34: RL reward definition and falsifiability
[Full Day 34 spec](days/day_34.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Reward hacking, faithful rewards, and a bounded group-relative RL objective on the text editor.

#### Build in MendSpeech
- Design the conservative reward on the Day13 contract: edit/format fidelity, protected-span safety, length and fluency penalties, with a per-example safety floor.
- Show a short-falsifiable prediction before training: which validation failures the reward should reduce and which must not increase.
- Use the pilot path from Day16; do not train on the CTC acoustic model and do not handcraft a per-example rewrite.

#### Experiment and Measure
- Construct adversarial cases (empty/truncated output, negation removal, entity edits, prompt-echo, runaway length) and verify each is penalized.
- Confirm the reward is computable offline on validation; log reward variance and any zero-variance groups. A reward that cannot be gamed by refusal earns nothing on needs-edit cases.

#### Required Output Artifacts
- `src/rl/reward.py`
- `configs/editor_reward.yaml`
- `tests/test_reward.py`
- `docs/day34_reward_design.md`

#### Completion Check
> A written falsifiable prediction plus reward tests that demonstrate the failure modes the reward is designed to penalize.

### Day 35: Editor SFT and continued-SFT control
[Full Day 35 spec](days/day_35.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Supervised fine-tuning for constrained text editing and the compute-matched continued-SFT control.

#### Build in MendSpeech
- Run editor SFT via training/editor_sft.py on the Day13 train split; freeze the SFT checkpoint as the RL reference.
- Run a continued-SFT control matched to the future RL wall-time/token budget, so extra compute is not mistaken for the RL algorithm.

#### Experiment and Measure
- Evaluate SFT and continued-SFT on validation (protected-content violations, formatting accuracy, identity vs needs-edit, risk-coverage).
- Log trainable-parameter names/counts, memory, step time and cost; artifacts for both arms.

#### Required Output Artifacts
- `training/editor_sft.py`
- `configs/editor_sft.yaml`
- `results/day35_sft_vs_continued.csv`
- `docs/day35_sft_notes.md`

#### Completion Check
> An SFT editor and a compute-matched continued-SFT control exist so any later RL gain cannot be explained by extra training alone.

## Week 6: Editor reward, SFT and compute-matched control, GRPO, and ASR robustness adaptation
[Week 6 guide](Week_6_MendSpeech_Daily_Plan.md)

### Day 36: Training pipeline anatomy for the editor
[Full Day 36 spec](days/day_36.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Loss curves, overfitting detection, tokenizer/label masking and checkpoint reproducibility.

#### Build in MendSpeech
- Diagnose the Day35 SFT loss/eval curves and add reproducibility checks in training/editor_sft.py.
- Document data mix, label masking and checkpoint reload determinism.

#### Experiment and Measure
- Verify checkpoint reload reproduces validation numbers.
- Relate loss/overfit behaviour to editor data size so RL budgets are set on evidence.

#### Required Output Artifacts
- `docs/day36_editor_training_diagnosis.md`
- `results/day36_editor_loss_curves.csv`

#### Completion Check
> Editor training behaviour is diagnosable and reproducible, giving evidence-based budgets for the RL run.

### Day 37: ASR robustness adaptation dataset and leakage audit
[Full Day 37 spec](days/day_37.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Acoustic corruption manifests and the frozen benchmark invariant.

#### Build in MendSpeech
- Build data/train_manifest.jsonl and val_manifest.jsonl from separate non-frozen source audio; keep data/test_manifest.jsonl byte-identical to the frozen set.
- Audit source/speaker leakage and corruption provenance; document in reports/data_audit.md.

#### Experiment and Measure
- Prove no source/speaker crosses splits; report severity distribution.
- The frozen test set is not used for training, tuning or checkpoint selection in this phase.

#### Required Output Artifacts
- `data/train_manifest.jsonl`
- `data/val_manifest.jsonl`
- `data/test_manifest.jsonl`
- `reports/data_audit.md`

#### Completion Check
> The adaptation dataset is leakage-free and the frozen evaluation set is provably untouched.

### Day 38: ASR robustness fine-tuning (not personalization)
[Full Day 38 spec](days/day_38.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Transfer learning, frozen versus trainable layers, mixed precision. This is acoustic robustness, not user personalization.

#### Build in MendSpeech
- Fine-tune the pinned streaming checkpoint with training/asr_finetune.py under configs/asr_finetune.yaml (steps, LR, seed, sampling frozen).
- Bind and re-validate Day25 calibration for the adapted checkpoint before it informs triage.

#### Experiment and Measure
- Compare base vs adapted on the frozen test set per corruption/severity with clean-speech regression.
- An adaptation that helps damaged speech but harms clean speech is a documented trade-off; no personalization claim is made.

#### Required Output Artifacts
- `training/asr_finetune.py`
- `configs/asr_finetune.yaml`
- `results/day38_base_vs_adapted.csv`
- `reports/day38_robustness_adaptation.md`

#### Completion Check
> Acoustic robustness is measured with clean-speech regression on the frozen set, and is reported as adaptation rather than personalization.

### Day 39: Augmentation ablation on corrupted audio
[Full Day 39 spec](days/day_39.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- SpecAugment, room impulse-response augmentation and training-time confounds.

#### Build in MendSpeech
- Run one controlled augmentation arm with identical steps/seed via experiments/augmentation_ablation.py.
- Test augmentation strength and label-preserving transforms; never alter the frozen set.

#### Experiment and Measure
- Compare no-augmentation vs augmentation at equal budget, then give the extra steps to the unaugmented baseline.
- Record the gain (or its absence) and per-corruption effect.

#### Required Output Artifacts
- `experiments/augmentation_ablation.py`
- `results/day39_augmentation.csv`

#### Completion Check
> The effect of augmentation is separated from the effect of extra training time.

### Day 40: Bounded GRPO post-training on the editor
[Full Day 40 spec](days/day_40.md)
**Compute:** Modal L4; spend/spend-capped

> **STATUS: CORE**

#### Learn
- Group-relative policy optimization, KL to the SFT reference, reward variance, rollout cost.

#### Build in MendSpeech
- Run the bounded GRPO run using training/editor_rl.py with the Day34 reward and Day35 SFT reference; default to LoRA-scale updates, modest steps, and a declared stop/spend budget.
- Log reward curves, KL to reference, group reward variance, completion length, adapter grad norms and refusals.

#### Experiment and Measure
- Evaluate against the SFT and continued-SFT controls on validation: protected-content violations, formatting accuracy, identity vs needs-edit, risk-coverage.
- Stop on nonfinite loss, repeated OOM, or safety violations rising >2pp above the SFT baseline at two consecutive evals; a valid null is kept, a failed run is blocked not disguised.

#### Required Output Artifacts
- `training/editor_rl.py`
- `configs/editor_rl.yaml`
- `results/day40_rl_vs_sft.csv`
- `docs/day40_rl_findings.md`
- `results/day40_reward_curve.csv`

#### Completion Check
> A bounded, controlled GRPO run on the text editor with SFT/continued-SFT comparison and explicit stop criteria, or a documented blocked/null outcome.

### Day 41: Personalization-adjacent robustness and error analysis
[Full Day 41 spec](days/day_41.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Acoustic robustness adaptation versus user personalization; error taxonomy for the shipped path.

#### Build in MendSpeech
- Build the failure casebook in reports/casebook.md for the robustness-adapted checkpoint plus the RL editor across corruption/severity.
- Explicitly label Day38 as robustness adaptation; define the personalization evaluation this release does NOT claim.

#### Experiment and Measure
- Rank failure modes by frequency and severity across the frozen matrix.
- Verify each failure mode has an owner stage (ASR, triage, editor, guard, or endpoint) so the report can attribute causality.

#### Required Output Artifacts
- `reports/casebook.md`
- `results/day41_failure_frequency.csv`

#### Completion Check
> A ranked, stage-attributed failure casebook, and an honest statement that this release measures acoustic robustness rather than user personalization.

### Day 42: Editor personalization feasibility check (scope, not claim)
[Full Day 42 spec](days/day_42.md)
**Compute:** Local CPU

> **STATUS: CORE**

#### Learn
- User-specific vocabulary/corrections would be personalization; this session sizes it without claiming it.

#### Build in MendSpeech
- Write a feasibility memo on what per-user enrollment, correction history and a personal lexicon would require in data, time and budget.
- Compare to the released scope; recommend keep, defer or drop with reasons.

#### Experiment and Measure
- No model training. Report effort estimates and dependency blockers.
- The memo must state that a personalization claim is not made unless this work is separately approved and executed.

#### Required Output Artifacts
- `docs/day42_personalization_feasibility.md`

#### Completion Check
> A written feasibility memo decides the scope of personalization without pretending it was achieved.

## Week 7: Serving, load, editor selection, and the correlated latency budget
[Week 7 guide](Week_7_MendSpeech_Daily_Plan.md)

### Day 43: Serving contract and WebSocket message schema
[Full Day 43 spec](days/day_43.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- WebSocket message schemas, per-stream isolation, cancellation and backpressure semantics.

#### Build in MendSpeech
- Define schema in src/serve/schema.py: audio chunks in; partial/final transcripts, confidence, stage events and latency fields out.
- Specify timeout, disconnect, cancellation and bounded-queue semantics; write contract tests in tests/test_serve_schema.py.

#### Experiment and Measure
- Verify the schema round-trips a recorded session.
- Ensure latency fields match the contract definitions and never expose unmeasured values.

#### Required Output Artifacts
- `src/serve/schema.py`
- `tests/test_serve_schema.py`
- `docs/day43_serving_contract.md`

#### Completion Check
> A testable WebSocket contract with explicit latency, confidence, failure and backpressure semantics.

### Day 44: Async streaming service
[Full Day 44 spec](days/day_44.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- FastAPI/async WebSocket handling, per-stream state isolation and clean cancellation.

#### Build in MendSpeech
- Implement the service in src/serve/app.py around the Day33 candidate, pinned one L4, serialized queue and bounded state.
- Containerize reproducibly under infra/serve/.

#### Experiment and Measure
- Verify concurrent streams do not share or corrupt cache/session state.
- Confirm a mid-utterance disconnect leaves no orphaned GPU work; report cold start separately from warm latency.

#### Required Output Artifacts
- `src/serve/app.py`
- `tests/test_serve_isolation.py`
- `infra/serve/Dockerfile`
- `infra/serve/README.md`

#### Completion Check
> Concurrent streams are isolated, disconnects are clean, and cold start is measured separately from warm latency.

### Day 45: Load test to saturation and failure recovery
[Full Day 45 spec](days/day_45.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Load methodology, saturation/queue growth, and throughput-at-saturation versus user experience.

#### Build in MendSpeech
- Build a load harness in src/serve/loadtest.py (configurable concurrency, fixed input, bounded budgets).
- Reproduce overload, disconnect and recovery scenarios.

#### Experiment and Measure
- Sweep concurrency within the authorized limit; report the knee, offered vs achieved rates, per-stream p50/p95/p99, queue wait and rejected/failed work.
- Show that rejection does not masquerade as capacity; document recovery in docs/day45_failure_recovery.md.

#### Required Output Artifacts
- `src/serve/loadtest.py`
- `results/day45_load_curve.csv`
- `docs/day45_failure_recovery.md`
- `reports/day45_serving.md`

#### Completion Check
> A load curve to saturation, a named concurrency knee, and a reproduced failure-and-recovery case within budget.

### Day 46: LLM stage measurement and consolidation
[Full Day 46 spec](days/day_46.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Server TTFT vs completion vs client-observed latency; prefix-cache support detection.

#### Build in MendSpeech
- Measure the Day15 editor (and Day40 RL editor if available) for TTFT/completion, quality, and prefix-cache hit/miss if the runtime exposes it.
- Consolidate prompt-only/SFT/RL editor results on the same frozen editor-test set with the Day13 quality metrics.

#### Experiment and Measure
- Compare editor variants at fixed workload; report misses, fallback rate, and quality.
- If co-residency/throughput prevents joint measurement, record the blocker; do not present isolated numbers as end-to-end.

#### Required Output Artifacts
- `configs/llm.yaml`
- `src/llm/polish.py`
- `results/day46_editor_variants.csv`
- `docs/day46_editor_selection.md`

#### Completion Check
> The editor variant used in serving is selected on measured quality/latency, and the choice is re-checked under load.

### Day 47: Per-stage correlated latency budget
[Full Day 47 spec](days/day_47.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Critical-path latency attribution; tail ownership by request ID, not percentile sum.

#### Build in MendSpeech
- Instrument the full path in src/bench/budget.py with trace IDs, clock sync notes, stage start/end, queue/prefill/decode/generation/guard/network.
- Produce correlated per-request critical paths for the shipped configuration.

#### Experiment and Measure
- Report per-stage p50/p95/p99, counts, cold/warm and length slices.
- Identify the p99 owner by inspecting the same slow requests; do not sum percentiles or claim a sub-500ms end-to-end guarantee.

#### Required Output Artifacts
- `src/bench/budget.py`
- `results/day47_latency_budget.csv`
- `results/day47_latency_budget.png`
- `docs/day47_latency_budget.md`

#### Completion Check
> A reproducible, request-level decomposition that names the tail owner and the largest optimization target, with no unsupported end-to-end claim.

### Day 48: End-to-end optimization round informed by the budget
[Full Day 48 spec](days/day_48.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Choosing one change from measured evidence and separating it from drift.

#### Build in MendSpeech
- Apply the change the Day47 budget identifies as the largest target in src/.
- Re-run the full Day47 decomposition and the Day32 scorecard after the change.

#### Experiment and Measure
- Report before/after p50/p95/p99 and quality with enough repetitions to separate a real gain from noise.
- A change that does not help, or that breaks a guard, is recorded as a negative result.

#### Required Output Artifacts
- `results/day48_e2e_optimization.csv`
- `docs/day48_optimization_outcome.md`
- `app/audio_lab.py`

#### Completion Check
> A measured end-to-end before/after tied to the budget, or a documented negative outcome with evidence.

### Day 49: Progress checkpoint against the plan (revised)
[Full Day 49 spec](days/day_49.md)
**Compute:** Local CPU

> **STATUS: CORE**

#### Learn
- Comparing measured results to plan claims; re-planning from actual throughput.

#### Build in MendSpeech
- No build. Re-read the plan gate table and update the day-10-onward effort forecast from observed sessions.

#### Experiment and Measure
- Record which gates have evidence and which remain open.
- Adjust remaining estimates; do not compress evidence to preserve a date.

#### Required Output Artifacts
- `docs/day49_progress_review.md`

#### Completion Check
> A short, honest checkpoint that updates the forecast from observed work without claiming completion.

## Week 8: Frozen evaluation, report, and release
[Week 8 guide](Week_8_MendSpeech_Daily_Plan.md)

### Day 50: Freeze the evaluation protocol and claims
[Full Day 50 spec](days/day_50.md)
**Compute:** Local CPU

> **STATUS: CORE**

#### Learn
- Pre-registration, null outcomes and the claims this release will not make.

#### Build in MendSpeech
- Freeze code/model/data revisions, hardware, corruption configs, editor prompt/reward, and metrics in configs/frozen.yaml.
- Write experiments/protocol.md: baselines, statistical caveat, null outcomes, failure criteria and excluded claims.

#### Experiment and Measure
- Run a dry run confirming every required field has a measurement or explicit status.
- No test-set inspection after this point; new experiments get new configs, never a new test set.

#### Required Output Artifacts
- `configs/frozen.yaml`
- `experiments/protocol.md`
- `docs/day50_protocol.md`

#### Completion Check
> The evaluation protocol, baselines and claim limits are frozen before the final measurement phase.

### Day 51: Release SpeechDamageBench v1 and freeze the evaluation set
[Full Day 51 spec](days/day_51.md)
**Compute:** Local CPU

> **STATUS: CORE**

#### Learn
- Severity grids, speaker-separated evaluation, seed control and checksum verification.

#### Build in MendSpeech
- Finalize the standalone package and lock manifest checksums under benchmarks/.
- Document a one-command reproduction example in speechdamagebench/README.md.

#### Experiment and Measure
- Reinstall in a clean environment; regenerate a sample and verify its checksum.
- Verify clean references are byte-identical after regeneration.

#### Required Output Artifacts
- `speechdamagebench/CHANGELOG.md`
- `benchmarks/manifest.csv`
- `benchmarks/README.md`

#### Completion Check
> A clean environment reproduces a benchmark item from the manifest and clean references are provably unchanged.

### Day 52: Robustness matrix on the frozen set
[Full Day 52 spec](days/day_52.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Sliced evaluation, multiple comparisons and variance.

#### Build in MendSpeech
- Run the full corruption x severity x decoder grid in src/bench/run_matrix.py on the frozen set.
- Report the robustness-adapted checkpoint and the base checkpoint in the same matrix.

#### Experiment and Measure
- Report WER/CER, names/numbers and confidence behaviour per cell.
- Estimate variance on a representative subset; identify the worst cell.

#### Required Output Artifacts
- `src/bench/run_matrix.py`
- `results/day52_robustness_matrix.csv`
- `results/day52_robustness_matrix.png`

#### Completion Check
> The complete robustness matrix is measured on the frozen set, with the worst cell and variance identified.

### Day 53: Optimization, serving and editor ablations on the frozen harness
[Full Day 53 spec](days/day_53.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Pareto frontiers; holding inputs fixed so comparisons mean something.

#### Build in MendSpeech
- Run every optimization variant, serving configuration and editor variant on the identical frozen subset in src/bench/run_ablations.py.
- Keep live measurements separate from simulated estimates.

#### Experiment and Measure
- Plot WER vs p99 latency and mark Pareto-efficient points.
- Report editor variants by quality/latency and state the shipped configuration.

#### Required Output Artifacts
- `src/bench/run_ablations.py`
- `results/day53_ablations.csv`
- `results/day53_pareto.png`

#### Completion Check
> A measured ablation set with Pareto frontiers and a stated shipping configuration.

### Day 54: Adaptation and RL final comparison
[Full Day 54 spec](days/day_54.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Separating acoustic robustness adaptation from text-editor post-training.

#### Build in MendSpeech
- Run base, robustness-adapted, SFT-editor and RL-editor conditions through the frozen harness in src/bench/run_personalization.py.
- Report each condition on its own axis: ASR WER/robustness for the checkpoint; editor quality for the editor.

#### Experiment and Measure
- Report the frozen-test numbers for all conditions, including clean-speech regression.
- State whether adaptation and RL each earned their place; a null result is reported, not hidden.

#### Required Output Artifacts
- `src/bench/run_personalization.py`
- `results/day54_conditions_final.csv`
- `reports/day54_conditions_final.md`

#### Completion Check
> The final conditions are compared on frozen evidence, with adaptation and editor quality kept distinct and nulls reported.

### Day 55: Technical report and reproduction guide
[Full Day 55 spec](days/day_55.md)
**Compute:** Local CPU

> **STATUS: CORE**

#### Learn
- Observation versus causal claim; honest negative results; reproducibility.

#### Build in MendSpeech
- Write REPORT.md with reproduction commands and environment capture; write REPRODUCE.md.
- Include the latency budget, robustness matrix, adaptation/RL results and editor quality as dedicated sections.

#### Experiment and Measure
- Audit every major claim against a table, figure or experiment.
- List every blocked, deferred or partial capability in docs/limitations_and_claims.md and soften unsupported conclusions.

#### Required Output Artifacts
- `REPORT.md`
- `REPRODUCE.md`
- `results/final_figures/`
- `docs/limitations_and_claims.md`

#### Completion Check
> A technical reader understands the contribution, trade-offs and limitations without opening the source.

### Day 56: Final demo, clean reproduction and release
[Full Day 56 spec](days/day_56.md)
**Compute:** Modal L4 for measured GPU work; local CPU for checks

> **STATUS: CORE**

#### Learn
- Honest demonstration including failure modes; reproducible release.

#### Build in MendSpeech
- Extend only app/audio_lab.py: live/replayed audio, partial/final transcripts, confidence, triage, guarded edit, latency budget, one failure case.
- Reproduce one frozen benchmark from a fresh environment and tag a release.

#### Experiment and Measure
- Verify every public chart regenerates from saved result files.
- Confirm displayed numbers match committed artifacts; show a failure case, not only the success path.

#### Required Output Artifacts
- `app/audio_lab.py`
- `REPRODUCE.md`
- `demos/final_demo.mp4`
- `docs/architecture.md`

#### Completion Check
> A new user can run, evaluate and reproduce the system, and every number shown traces to a committed artifact.

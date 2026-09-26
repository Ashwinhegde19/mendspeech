# Revised Execution Plan (v2 — Focused Restoration Scope)

> **Scope revision: September 26, 2026.** This document governs pacing,
> session compression, add-ons, and release gates. The individual day files
> specify implementation and artifact contracts. This revision changes the
> plan, not the completion status of any experiment.
>
> Earlier October calendar targets are superseded. Gates advance on measured
> evidence, not elapsed dates. Do not infer that a gate passed from its number.

## 1. Release Boundary

MendSpeech tests whether uncertainty-guided selective reconstruction improves
damaged speech while retaining reliable original audio under latency constraints.
SpeechDamageBench remains an independently installable, deterministic package.

The release contains one measured ASR/streaming pipeline, one repair policy,
one selected TTS stack, one evolving application, one serving endpoint, and
one reproducible evaluation suite. The scratch Conformer block is a tested
learning component, not a second production recognizer.

### Approved scope decisions

| Area | v2 decision | Evidence retained |
| :--- | :--- | :--- |
| Encoder | Day 19 merges into Day 18; no separate tiny encoder or depth sweep | Real log-Mel projection, masks, shape and gradient checks |
| Architecture review | Keep Day 21; remove the separate inspector UI | Static shape trace and architecture report |
| Application | Extend `app/audio_lab.py`; reusable UI components are allowed | Milestone screenshots, reports, and results, not parallel app implementations |
| ASR decoding | Compare greedy, beam-only, and beam plus one small n-gram LM on the same acoustic model | Held-out accuracy, names/numbers, helpful/harmful changes, and separate decoder/end-to-end timings |
| TTS | Use one compatible stack, including its vocoder; target two verified languages including one Indian language | Language-specific quality, bounded adaptation, repair-focused prosody, and seam diagnostics |
| TTS latency | Measure short-span synthesis and check the selected backend's streaming capability | First playable audio, completion, RTF, and explicit native/chunked/full-waveform labels |
| VAD | Keep baseline, reference comparison, and endpointing | No timed rebuild or stopwatch completion requirement |
| Systems drills | Optional learning reference | No quota or release dependency |
| Voice-agent loop | Add-on D is removed from the release scope | Streaming and synthesis demonstrated in MendSpeech itself |
| External restoration | One candidate and a bounded feasibility check | Correct capability labels; blocked comparisons explicitly deferred |
| Adaptive context | Fixed streaming first; bounded supported-context experiment | Separate live, simulated, and unavailable outcomes |

Do not add multi-cloud deployment, mobile/edge ports, a diarization subsystem,
custom CUDA kernels, scratch vocoder training, a separate emotion-generation
subsystem, or an architecture survey to this release. Do not select a second
TTS stack merely to obtain streaming. Such extensions require a measured need
and a separate scope decision.

## 2. Session Accounting and Protocol

- Day numbers are stable specification identifiers, not consecutive calendar days.
- Weeks 3–4 contain **9 build sessions**: Days 15, 16, 18, 21, 23, 24, 25, 26,
  and 28. Day 17 remains learn-only with its build absorbed by Day 18; Day 19
  merges into Day 18; Day 20 remains dropped; Day 22 merges into Day 23;
  Day 27 merges into Day 28.
- Week 8 retains **5 build sessions**: Days 50, 51, 52+53, 54, and 55+56.
- Starting at Day 10, the nominal remaining core is **40 build sessions**:
  5 recognition + 9 encoder + 7 streaming + 7 robustness + 7 TTS + 5 capstone.
  Add-ons A, B, and C budget approximately 2 sessions each: **46 base slots**
  before debugging, training/data preparation, or feasibility-driven extensions.
- The approved decoding and TTS amendments expand Days 24/26/28/41 and 43–49
  without renumbering or creating another add-on. The 46-slot base is **not an
  updated delivery estimate**: LM text preparation, decoder integration,
  two-language review and listening, and latency checks require extra work.
  Re-estimate these after Day 24 and Day 43 compatibility checks; apply the
  scope-review rule below rather than silently fitting them into two-hour slots.
- Six sessions per week remains a planning cadence, not a completion promise.
  Sunday stays recovery-only. Record revised estimates from observed work;
  do not compress evidence to protect a date.
- Retain Learn, Build, Measure, and explanation checks. Optional drills do not
  replace the day's learning. A task may use at most two extra sessions before
  explicitly reporting its remaining scope and obtaining a revised decision.
  Required failed checks remain incomplete; only explicitly conditional branches
  below may be deferred without inventing results.

## 3. Data and Measurement Contracts

- The Week 1 benchmark is frozen: at least 30 transcripted utterances, at least
  5 speakers, speaker-separated splits. No new speakers or relabeling in place.
- Training, validation, calibration, and test roles must be explicit. Corrupted
  copies retain source IDs so a source cannot leak across splits. Choose model
  checkpoints and policy thresholds without consulting the test outcomes.
- Add-on C uses a separate manifest; never replace or enlarge the frozen core
  benchmark to improve a result. Record consent/license, transcript verification,
  language, speaker, source, and normalization policy.
- Day 26 records LM text provenance in `data/lm_text_manifest.csv`; exclude
  evaluation references and duplicate text from LM fitting. Decoder settings
  are selected on validation only. Day 43 creates `data/tts_eval_manifest.csv`
  for separate held-out two-language synthesis evaluation. Its sentences and
  reference recordings cannot become training/tuning data. Track consent,
  language, normalization, split roles, and known pretraining-overlap limits.
- Every generated sample records corruption, severity, seed, source, parameters,
  and package version. Preserve previous result files; new experiments get new
  configurations and result artifacts with entries in `results/README.md`.
- Normal repair consumes predicted text. Reference-transcript reconstruction is
  a separately labeled **oracle** experiment, never an end-to-end system result.
- Record hardware, model/software revisions, batch size, and timing boundaries.
  All comparable GPU latency, RTF, and memory measurements use Modal L4.
  Keep CPU VAD/decoder timing distinct and record host/worker configuration;
  do not silently mix hardware tiers. Cached-logit decoding time is not fresh
  audio-to-transcript latency. Fused LM search scores are not calibrated confidence.
- The existing approximate $15–30 compute envelope is a constraint, not an
  estimate for newly scoped training. Check remaining budget and expected L4
  cost before a TTS adaptation run; stop rather than silently exceed it.

## 4. Milestone Gates

| Gate | Sessions | Exit evidence |
| :--- | :--- | :--- |
| **Gate 1** | 02–07 | Tested audio lab, standalone SpeechDamageBench v0, frozen labeled benchmark |
| **Gate 2** | 08–14 | ASR, WER/CER, confidence/timing, safe policy, reproducible Modal metadata; external comparator feasibility status recorded; then Add-on A |
| **Gate 3** | 15–28 compressed | One tested Conformer block; measured pretrained baseline, capability record, efficiency harness, greedy/beam/LM decoding evidence, and failure casebook |
| **Gate 4** | 29–35 | Correct cache-aware streaming, VAD/endpointing, fixed-context measurements; adaptive experiment labeled live/simulated/deferred; then Add-on B |
| **Gate 5** | 36–42 | One ASR adaptation experiment with clean regression; supported export/precision comparisons or explicit blocked optimization status; model/decoder/precision-specific calibration and robustness report |
| **Gate 6** | 43–49 | One TTS stack, two-language evaluation including an Indian language, adaptation result or explicit feasibility deferral, repair-focused prosody and short-span latency, verified streaming status, selective repair, and tested abstention |
| **Gate 7** | 50–56 compressed | Frozen controlled comparisons, clean regression, bounded external comparator outcome, technical report, demo, and clean reproduction; finish Add-on C measurements |

An explicit deferral is not successful implementation of that capability.
The release report must list it, explain the blocker, and narrow its claims.
Core safe repair, streaming correctness, calibration, and reproducibility cannot
be replaced by a feasibility note.
Required three-way decoding and two-language evaluation remain incomplete if
blocked; continued independent work does not close those requirements. Narrowing
them requires an explicit scope review. Native streaming TTS remains conditional:
an unsupported backend still requires a measured full-waveform latency baseline,
not a streaming claim. TTS adaptation retains its separate feasibility gate.

## 5. Retained Add-Ons

### Add-on A — VAD and Endpointing Baseline (after Gate 2, approximately 2 sessions)

- Implement a small deterministic frame-level energy or spectral VAD, with
  framing/timestamp/silence tests, on a fixed labeled 30–50-file subset.
- Compare it with one local reference VAD. Measure precision/recall/F1, false
  alarms, missed speech, onset/offset error in ms, and CPU RTF on clean/damaged
  cases. Carry the measured choice into Day 35 and Add-on B.
- Artifacts: `src/vad/baseline.py`, `tests/test_vad.py`,
  `results/addon_a_vad_benchmark.csv`, and `docs/addon_a_notes.md`.
- Completion: explain the detector's observed failure modes and reproduce the
  measurements. No timed rebuild; no separate diarization or denoising branch.

### Add-on B — Serving Deployment (after Gate 4, approximately 2 sessions)

- Wrap the streaming system in one async FastAPI/WebSocket service on Modal.
  Record the reproducible container configuration; use the same VAD/endpointing
  behavior as the core pipeline. Bound queues and make timeout, disconnect,
  retry, and fallback behavior explicit.
- Measure 1, 4, and 8 streams on fixed hardware: time to first partial,
  finalization delay, end-to-end p50/p95/p99, RTF, cold start, utilization,
  peak GPU memory, queue depth, and dropped/delayed chunks.
- Artifacts: `infra/serve/`, `results/addon_b_serving.csv`, and
  `reports/addon_b_serving.md`.
- Completion: identify the bottleneck and reproduce one controlled failure and
  recovery. One provider and one endpoint; no second deployment exercise.

### Add-on C — Indic and Code-Mixed Evaluation (approximately 2 sessions, split)

- After Gate 2, establish a separate verified manifest for one Indian language,
  Indian English, and a code-mixed slice. Include multiple speakers, names,
  numbers, transliteration, and clean/noisy/dropout conditions. Verify checkpoint
  and tokenizer support before claiming evaluation in that language.
- Record offline WER/CER and entity errors when feasible. Complete endpointing,
  first-partial, and latency measurements after streaming exists, and finish the
  report with Gate 7. Do not fill future metrics with invented values.
- Where compatible data and model support exist, reuse this scope in the single
  Days 37–38 ASR adaptation experiment with disjoint train/validation/test data.
  Otherwise keep it evaluation-only; do not start another model-training track.
- Prefer the same verified Indian language in the separate Day 43 TTS set when
  supported. ASR language coverage does not establish TTS coverage, and two
  monolingual synthesis slices do not establish code-mixed synthesis. Keep
  manifests and task results distinct; do not add a third required TTS slice.
- Artifacts: `data/indic_codemix_manifest.csv`,
  `results/addon_c_indic_codemix.csv`, and `reports/addon_c_speech_readiness.md`.
- Completion: explain at least three observed language/code-mixing failure
  modes, or report why coverage is insufficient. A missing compatible model
  leaves that slice incomplete, not a claim of multilingual performance.

## 6. Bounded Experiments

**ASR decoding (Days 24, 26, 28, 41):** Day 24 verifies the selected acoustic
checkpoint's decoder/head, tokenizer, and optional backend requirements. Day 26
extends the existing harness with offline greedy, beam without an external LM,
and beam plus one small n-gram LM. Reuse identical acoustic outputs only where
the head/backend permits; a CTC recipe is not automatically an RNN-T recipe.
Keep lexicon, insertion settings, and normalization controlled; use a small
predeclared validation search, not an open-ended sweep. Report WER/CER,
names/numbers, help/hurt cases, cache status, decoder-only and fresh end-to-end
timing. Tests cover token mapping, empty input and repeated/blank CTC behavior
where applicable. Artifacts and required checks live in [Day 26](days/day_26.md).
Unsupported dependencies remain a blocker, not permission for another acoustic
model or a scratch decoder. Day 28 retains a verified streaming-compatible
decoder; offline LM decoding cannot silently replace it. Day 41 validates
confidence for the chosen model/decoder/precision before repair decisions.

**Two-language synthesis (Days 43–46):** Target two languages supported by the
same stack, including at least one Indian language; neither English nor
code-mixed support is assumed. Freeze at least ten held-out sentences per
language as a small diagnostic set with names/numbers and competent language
review, not a population benchmark. Document same-speaker versus unseen-speaker
conditions and separate training languages from evaluated languages. A native
API smoke check must establish each claimed capability; unsupported language,
prosody, or streaming features are distinct from an adaptation blocker.

**TTS adaptation:** Day 43 records one stack's permitted data/checkpoint use,
trainable parameters, held-out sentences, speaker protocol, memory and cost.
Day 46 runs a bounded base-versus-adapted comparison only after this check
passes. Keep configurations and provenance, compare intelligibility, naturalness,
duration and latency, and report non-improvement honestly. If blocked, keep the
pretrained repair path and explicitly defer adaptation; no second TTS installation.
Day 46 compares both language slices, including regression in a language not
used for adaptation. Use only language-supported intelligibility proxies and
record evaluator limitations alongside listening evidence.

**Repair prosody and synthesis latency (Days 44–45, 49):** Day 44 compares the
base condition with supported native controls; bounded DSP remains separately
labeled, never learned emotion control. Measure duration error, voiced pitch,
energy continuity, intelligibility, and blinded seam judgments. Day 45 records
`results/day45_tts_latency.csv` on fixed L4, including sample count, cold/warm
conditions, p50/p95, RTF, memory, and first playable audio versus completion.
Define the timing start and playable buffer before measuring. Label actual
behavior `native_streaming`, `phrase_chunked`, or `full_waveform_delivery`;
network chunks of a completed waveform do not prove incremental synthesis.
Run native streaming checks only if the selected pinned backend supports them,
including ordering, finalization, sample coverage, and chunk-boundary quality.
State whether full text is required up front. Day 49 separately measures the
complete repair path, including context buffering and stitching; synthesis-only
timing cannot establish end-to-end repair latency.

**Adaptive context:** Day 24 verifies supported context/cache behavior. Day 32
establishes fixed settings; Day 34 tests supported switching or clearly labeled
simulation. A simulation cannot establish live tail-latency or compute savings.
Do not implement a new inference framework to rescue this secondary experiment.

**External restoration:** Week 2 records one candidate's real interface, pinned
revision, license, input/output format, mask support, and smoke-test outcome.
Use at most one setup session plus one focused compatibility retry. If blocked,
stop and record the deferral. Day 54 compares the verified system with the core
baselines or documents the unsupported external question. General restoration
is not automatically masked inpainting. No scratch inpainting fallback or
open-ended model search; see [baseline notes](baseline_install_notes.md).

## 7. Release Evidence and Optional Learning

Keep raw damaged audio, full resynthesis, naive selective repair, boundary-matched
repair, calibrated versus raw confidence, fixed-context streaming, and clean
regression controls. Record false repairs, abstentions, retained audio, seams,
intelligibility, and latency. Negative results remain valid measurements.

Keep the report and clean reproduction. Each claim must point to a result,
each conditional omission to a limitation. The [systems drills](SPEECH_ML_SYSTEMS_DRILLS.md)
are optional study material, not release gates. Archived PDFs remain unchanged.
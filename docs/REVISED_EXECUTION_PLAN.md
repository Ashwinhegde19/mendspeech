# Revised Execution Plan (v3 — Real-Time Voice Interface)

> **Scope revision, September 26, 2026.** This document governs pacing,
> session compression, and release gates. Day files specify implementation and
> artifact contracts. This revision changes the plan, not the completion status
> of any experiment.
>
> **Gates advance on measured evidence, not elapsed dates.** Do not infer that a
> gate passed from its number. A blocked required target is incomplete, not done.

## 1. Project Thesis

MendSpeech is a **real-time voice interface**: damage-robust streaming ASR, an
LLM post-processing stage, calibrated confidence, and bounded RL
personalization — measured as a single latency budget from waveform to polished
text.

`SpeechDamageBench` remains an independently installable, deterministic
package and becomes the project's **robustness evaluation suite**.

### Research questions, in priority order

1. **Where does the end-to-end latency budget actually live?** Decompose
   audio-to-transcript, LLM time-to-first-token, and full response into
   `p50/p95/p99`, find the tail owner, and improve it with measured evidence.
2. **How far can a speech model be pushed for a specific acoustic condition?**
   Measure controlled adaptation and RL post-training, including where they
   fail.
3. **What does confidence mean, and when is it safe to act on it?** Calibrate
   against correctness and identify confident-but-wrong cases.

### Non-goals for this release

Speech restoration and TTS synthesis are **out of scope**. So are diarization,
mobile/edge ports, multi-cloud deployment, custom CUDA kernels, and
architecture surveys. Additions require a measured need and an explicit scope
decision.

---

## 2. Requirement Mapping

| Requirement | Phase | Required evidence |
| :--- | :--- | :--- |
| Optimize ML inference / research systems | **P4** | INT8/FP16, `torch.compile`, CUDA graphs, batching, throughput and cost curves |
| Sub-500ms end-to-end response | **P3, P6** | Per-stage `p50/p95/p99`, tail owner identified and improved |
| Personalize speech models: fine-tuning | **P5** | Leakage-audited adaptation run, clean regression, validation-only selection |
| Personalize with reinforcement learning | **P5** | RL run against a falsifiable reward, base vs. adapted comparison |
| Build with LLMs | **P6** | Pinned small model, measured TTFT, streaming/prefix-cache/batching effects |
| Engineering systems for research | **P7** | One harness, one command, regression suite, benchmark registry, reproduction |

---

## 3. Approved Scope Decisions

| Area | v3 decision | Evidence retained |
| :--- | :--- | :--- |
| Encoder | Day 19 merges into Day 18; no separate tiny encoder or depth sweep | Real log-Mel projection, masks, shape and gradient checks |
| Application | Extend `app/audio_lab.py`; reusable UI components allowed | Screenshots, reports, results — not parallel apps |
| TTS / synthesis | **Removed.** No TTS stack, adaptation, prosody, or latency work | — |
| Restoration / repair policy | **Removed.** No stitching, seam metrics, or comparator | — |
| External restoration baseline | **Removed** | — |
| Indic / code-mixed add-on | **Removed** | — |
| Decoding | Greedy vs beam vs beam + one small n-gram LM on one acoustic model | Held-out accuracy, names/numbers, helpful/harmful changes, separate decoder and end-to-end timing |
| Inference optimization | **Expanded** to the project centerpiece | Measured accuracy/latency/memory tradeoffs per technique |

---

## 4. Session Accounting and Protocol

- Day numbers are stable specification identifiers, not consecutive days.
- Starting at Day 10, the nominal remaining scope is **39 build sessions**:
  5 recognition + 6 streaming + 7 optimization + 6 personalization/RL +
  6 serving/LLM + 5 evaluation + 5 report. Known merges reduce this further.
- Days 17 (learn-only) and 19 (merged into 18) remain as in v2; Day 20 is
  dropped; Day 22 merges into Day 23; Day 27 merges into Day 28.
- A planning cadence of 5–6 sessions per week is a target, not a promise.
  Estimate from observed throughput and re-plan when it misses. Never
  compress evidence to protect a date.
- Retain Learn, Build, Measure, and explanation checks. A task may use at most
  two extra sessions before an explicit scope review.
- **Required failed checks remain incomplete.** Only explicitly conditional
  branches may be deferred, with a recorded blocker and narrowed claims.

### Data and measurement contracts

- The Week 1 benchmark is frozen: at least 30 transcripted utterances, at least
  5 speakers, speaker-separated splits. No relabeling in place.
- Training, validation, calibration, and test roles must be explicit.
  Corrupted copies retain source IDs so a source cannot leak across splits.
  Thresholds, decoding parameters, and reward tuning use validation only.
- Every generated sample records corruption, severity, seed, source,
  parameters, and package version. Preserve previous results; new experiments
  get new artifacts with rows in `results/README.md`.
- Record hardware, model and software revisions, batch size, warm-up, and
  timing boundaries. **All comparable GPU latency, RTF, and memory measurements
  use Modal L4.** CPU VAD and CPU decoder timings stay separate and labelled.
- Cached-logit decoding time is not fresh audio-to-transcript latency. Fused LM
  search scores are not calibrated confidence. Latency percentiles are not
  means. LLM time-to-first-token is not full response time.
- The approximate **$15–30** compute envelope is a constraint, not an estimate
  for newly scoped training or LLM serving. Check remaining budget and expected
  cost before each L4 run; stop rather than silently exceed it.

---

## 5. Phases and Gates

| Phase | Days | Sessions | Exit evidence |
| :--- | :--- | ---: | :--- |
| **P1 Foundation** ✅ | 01–09 | 9 (done) | Audio lab, standalone SpeechDamageBench, frozen labeled benchmark, Wav2Vec2 baseline, 30-run corruption benchmark |
| **P2 Recognition quality** | 10–14 | 5 | WER/CER and error taxonomy, token confidence, time alignment, validation-based calibration, three-way decoding comparison |
| **P3 Streaming and endpointing** | 15–26 | 6 | Correct cache-aware streaming, VAD/endpointing, fixed-lookahead measurements, streaming latency frontier |
| **P4 Inference optimization** ⭐ | 27–33 | 7 | INT8/FP16, `torch.compile`, CUDA graphs, batching, streaming fast path; measured accuracy/latency/memory tradeoffs |
| **P5 Personalization and RL** ⭐ | 34–39 | 6 | Leakage-audited fine-tune with clean regression, augmentation ablation, bounded RL run against a falsifiable reward |

---

## 6. Bounded Experiments

**Decoding (P2).** One acoustic checkpoint. Greedy, beam without an external
LM, and beam plus one small n-gram LM. Reuse identical acoustic outputs only
where the head permits; a CTC recipe is not automatically an RNN-T recipe.
Control lexicon, insertion settings, and normalization. Use a small
predeclared validation search, not a sweep. Record LM text provenance in
`data/lm_text_manifest.csv`, excluding evaluation references and duplicates.
Report WER/CER, names/numbers, **and both helpful and harmful transcript
changes** — a lower WER does not prove repair-safety. Measure decoder-only
cached time separately from fresh audio-to-transcript latency.

**Calibration (P2).** Confidence is defined for the actual model, decoder, and
precision in use. A fused beam score is not a probability. Greedy thresholds
cannot silently transfer to LM-altered hypotheses. Validate token and timestamp
alignment, or retain the verified greedy path where alignment fails.

**Streaming (P3).** Fixed lookahead settings with a measured WER-versus-latency
frontier. Adaptive context is a bounded experiment using only supported
settings, labelled `live`, `simulated`, or `unavailable`. A simulation cannot
establish live latency savings, and no new inference framework is written to
rescue it.

**Optimization (P4).** One model, one L4 tier, one batch-size discipline.
Verify export parity before trusting any optimized variant. Report actual WER,
latency, RTF, and memory per technique. **INT8 or FP16 may be slower or less
accurate and need not be selected.** Report per-variant `measured` or `blocked`
status with reasons; never claim a speedup without a measurement.

**Personalization (P5).** One adaptation recipe, reused for the augmentation
control. Audit source and speaker leakage before training. Compare base,
fine-tuned, and RL variants on held-out data with a clean-speech regression
check. RL uses a **falsifiable reward** — penalize plausible but acoustically
unsupported output — and is compared against the fine-tuned baseline. If RL
does not help, that is a valid result. A blocked RL run leaves fine-tuning
intact and the RL target incomplete.

**LLM stage (P6).** One small pinned model, post-processing only, behind an
adapter so the core ASR result is reproducible without it. Measure TTFT and
full-response latency, streaming versus batched generation, prefix-cache hit
rate, and effect under concurrency. Report quality impact on the polished-text
output. Do not add an agent loop, tool use, or a second model.

**Serving (P6).** One provider, one endpoint, one async WebSocket service.
Load-test to saturation and report the concurrency knee, queue depth, and
per-stream latency distribution. Reproduce one controlled failure and recovery.
Modal deployment is the measured environment; it is not evidence of named
cloud-provider experience, and the report must not imply it.

---

## 7. Release Evidence

Required before the release is claimed complete:

1. Frozen benchmark results with speaker-separated splits and the statistical
   caveat stated.
2. Decoding comparison including harmful LM-induced changes.
3. Streaming correctness, cache-failure evidence, and a measured latency
   frontier.
4. Optimization table with per-technique accuracy/latency/memory and explicit
   blocked rows.
5. Fine-tuning and RL results with clean regression, or a recorded RL blocker.
6. Serving load curve, backpressure behavior, and one reproduced failure.
7. LLM stage TTFT and full-response latency, or a recorded blocker.
8. One-command reproduction of at least one benchmark from a clean environment.
9. A technical report where every claim points to a table, figure, or
   experiment, and every omitted capability appears as a limitation.

Negative results are valid measurements and belong in the report. The
[systems drills](SPEECH_ML_SYSTEMS_DRILLS.md) are optional study material, not
release gates. Archived PDFs remain unchanged.

| **P6 Serving and LLM stage** ⭐ | 40–45 | 6 | FastAPI/WebSocket streaming service, load-to-saturation, backpressure and recovery, pinned small LLM, per-stage latency decomposition |
| **P7 Evaluation infrastructure** | 46–48, 50–51 | 5 | One harness and one command, regression suite, benchmark registry, frozen evaluation, robustness matrix |
| **P8 Report and release** | 52–56 | 5 | Latency budget report, technical report, demo, clean reproduction, tagged release |

An explicit deferral is not a completed capability. The release report lists
it, explains the blocker, and narrows its claims. Core recognition correctness,
streaming correctness, calibration, and reproducibility cannot be replaced by a
feasibility note.

| LLM stage | One small pinned model, post-processing only | TTFT, full-response latency, cache and batching effects, quality |
| RL personalization | Bounded post-training against a falsifiable reward | Base vs. adapted, and adapted vs. RL |
| VAD | Keep baseline, reference comparison, endpointing | No stopwatch or timed-rebuild requirement |
| Systems drills | Optional learning reference | No quota or release dependency |

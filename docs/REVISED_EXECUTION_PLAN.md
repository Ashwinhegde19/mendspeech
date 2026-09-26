# Revised Execution Plan (v4 — Meaning-Preserving Dictation)

> **Scope revision, September 26, 2026.** This document governs scope, phases,
> and release gates. The individual day specs define implementation and artifact
> contracts; [`plan_manifest.json`](plan_manifest.json) is the machine-readable
> source of truth for statuses, prerequisites, and effort.
>
> **Gates advance on measured evidence, not elapsed dates.** A blocked required
> target is incomplete. It requires an explicit scope decision, never a relabelled
> result. Regenerate the derived documents with `python scripts/plan_docs.py --write`
> and verify them with `--check`.

---

## 1. Project Thesis

MendSpeech is a **meaning-preserving dictation system**: damage-robust streaming
ASR, confidence calibrated against correctness, a conservative transcript
editor, acoustic robustness adaptation, and bounded post-training of the editor —
all measured through a single correlated latency budget.

`SpeechDamageBench` remains an independently installable, deterministic package
and serves as the robustness evaluation suite.

The central question is narrow and testable:

> **Can a system measurably improve the readability of a transcript without
> changing what the speaker said, and what does that cost in latency?**

Readability gains are only real if protected content survives. Formatting,
casing, and punctuation may be improved; names, numbers, negation, and units may
not be altered. A fluent rewrite of an ASR error is a failure, not a success.

Supporting questions, in priority order:

1. **Where does the latency budget actually live?** Decompose waveform to final
   text per stage, attribute the p99 by request, and improve the largest owner.
2. **When is confidence safe to act on?** Fit calibration on held-out data and
   characterize confident-but-wrong transcripts.
3. **Does bounded post-training help the editor, and against which control?**
   Supervised editing versus group-relative RL, matched for compute.

### Explicitly out of scope for this release

Speech synthesis, speech restoration, and diarization. So are mobile/edge ports,
multi-cloud deployment, custom CUDA kernels, scratch RL engines, and
user-specific personalization. Additions require a measured need and an
explicit scope decision recorded in the manifest.

---

## 2. Controlling documents

| Document | Purpose |
| :--- | :--- |
| [`MendSpeech_Project_Blueprint.md`](MendSpeech_Project_Blueprint.md) | Architecture, metrics, and definition of done |
| [`LATENCY_AND_QUALITY_CONTRACT.md`](LATENCY_AND_QUALITY_CONTRACT.md) | Timing boundaries, workload, sampling, and fair-comparison rules |
| [`EDITOR_AND_RL_CONTRACT.md`](EDITOR_AND_RL_CONTRACT.md) | Editing limits, data roles, model route, reward design, and anti-shortcut tests |
| [`plan_manifest.json`](plan_manifest.json) | Statuses, prerequisites, phases, and effort ranges |
| `docs/days/day_NN.md` | Per-session specifications and artifact contracts |

These documents take precedence over this file when any two disagree, and this
file takes precedence over generated week and compiled views.

---

## 3. Phases and session accounting

Session status is authoritative only in the manifest. Effort ranges are focused
hours including learning and tests; they are estimates, not deadlines.

| Phase | Days | Core sessions | Effort (hours) | Exit evidence |
| :--- | :--- | ---: | ---: | :--- |
| **P1** Foundation ✅ | 01–09 | 9 | complete | Audio lab, standalone `SpeechDamageBench`, frozen labeled benchmark, CTC from first principles, Wav2Vec2 baseline, 30-run corruption benchmark |
| **P2** Protocol and first end-to-end path | 10–16 | 7 | 19–31 | Scoring and frozen data roles, confidence definitions, correlated tracing, conservative editor contract and data, three-way decoder comparison, **ASR→editor baseline with a resource pilot**, and an **SFT/GRPO feasibility test** |
| **P3** Streaming correctness and calibration | 18–26 | 6 | 15–27 | Streaming capability gate, tested chunk/session loop, endpointing state machine, fixed-context frontier, **fitted calibration and triage**, and an end-to-end pipeline with a latency baseline |
| **P4** Measured inference optimization | 27–33 | 7 | 15–27 | Profile first, then compile/graph capture, batching, precision parity, cache-failure evidence, scorecard, and pipeline revalidation |
| **P5** Editor post-training and acoustic robustness | 34–42 | 9 | 23–39 | Falsifiable reward, editor SFT with a compute-matched control, bounded GRPO run, failure casebook, and a personalization scope decision |
| **P6** Serving and the latency budget | 43–49 | 7 | 14–26 | Serving contract, async endpoint, load to saturation, editor selection under load, **correlated per-stage latency budget**, one evidence-driven optimization, and a progress review |
| **P7** Frozen evaluation and report | 50–55 | 6 | 10–17 | Frozen protocol and checksums, robustness matrix, Pareto ablations, final condition comparison, technical report, and reproduction guide |
| **P8** Release | 56 | 1 | 2–3 | Final demo including a failure case, clean reproduction, tagged release |

- **Core sessions: 52.** Days 01–09 are complete; **43 remain** from Day 10.
- **Total effort across the full plan: 98–170 focused hours**, including the
  completed foundation sessions. Estimate the remaining days from observed
  throughput rather than from this range.
- Days 17, 19, and 20 carry no build session: 17 and 22 are learn-only, 19 is
  merged into 18, and 20 is dropped.
- Day 16 is a feasibility experiment, not a claim of completed RL. It gates
  whether the post-training track proceeds.
- Day 42 is a scope decision about personalization, not an implementation of it.

---

## 4. Ordering rules that the manifest enforces

- Every prerequisite refers to an earlier specification, and the dependency
  tests fail the build if one does not.
- **P2 precedes P4.** A full ASR→editor pipeline with a measured resource
  pilot exists before any optimization technique is applied.
- **P2 precedes P5.** The editor's reward, SFT baseline, and compute-matched
  control are defined before any post-training run.
- **P3 precedes P4.** Streaming, endpointing, and calibration exist before
  per-operator profiling, because the profile must cover the real path.
- **P3 precedes P6.** The service wraps a revalidated pipeline.
- **P7 is last.** The protocol is frozen before the final measurement phase.
- The pipeline is revalidated after optimization and after training, because
  optimization and adaptation both invalidate earlier numbers.

---

## 5. Data, measurement, and honesty rules

### Frozen and separate data

- The Week 1 speech benchmark is frozen: at least 30 transcripted utterances,
  at least 5 speakers, speaker-separated splits, locked checksums. Its test
  membership is never altered; Day 51 verifies that invariant.
- Training, validation, calibration, and final-test roles are declared in
  `experiments/protocol.md` and audited before downstream tuning.
- Corrupted copies retain source IDs, so a source cannot leak across splits.
  Audits run before training, not after results look wrong.
- Editor data lives in `data/editor_manifest.jsonl`, separate from the speech
  benchmark, with group-aware splitting, provenance, and sealed held-out roles.
- ASR adaptation data is separate and non-frozen; the frozen test set is used
  for evaluation only.

### Measurement discipline

- Identical inputs, fixed hardware (Modal L4 for comparable GPU work), fixed
  batch discipline, recorded warm-up, and recorded timing boundaries for every
  comparison.
- Correlation samples are grouped by source utterance; repeated corruptions of
  one utterance are not independent observations.
- Report counts, seeds, variance, and exclusions alongside every percentage.
  Small diagnostic sets support bounded statements, not population claims.
- Cached-logit decoder time is not fresh audio-to-transcript latency. Fused
  decoder scores are not calibrated probabilities. Time-to-first-token is not
  completion, and neither is end-to-end latency.
- Percentiles are not additive. The p99 owner is identified by inspecting the
  same slow request identifiers and their critical paths.

### Blocking and deferral

- Optional runtime features are selected by measurement, not by checklist.
  Unsupported prefix caching, quantization, or graph capture is recorded as
  unsupported; it is never assumed.
- A speedup requires a measurement. A slower, more accurate, or unmeasured
  variant is a valid result and is reported as such.
- A blocked required target leaves that target incomplete and requires a scope
  decision. Partial delivery is reported as partial.
- A failed trainer, an OOM, or a nonfinite update is a blocked experiment, not
  a negative research result.

---

## 6. Release evidence

The release is complete only when each item below exists as a committed,
reproducible artifact:

1. Frozen benchmark results with speaker-separated splits and the statistical
   caveat stated.
2. A data-role and leakage audit that proves the frozen test set is untouched.
3. Three-way decoder comparison including **harmful** language-model changes.
4. Calibrated confidence with reliability and risk-coverage evidence, bound to
   the shipped configuration.
5. Tested streaming correctness, cache-failure evidence, and endpointing error
   in milliseconds.
6. An end-to-end baseline preceding optimization, plus a measured before/after
   for any change that shipped.
7. Editor quality evidence: protected-content violations, formatting accuracy,
   identity versus needs-edit behavior, fallback rate, and independent review.
8. A post-training comparison against supervised and compute-matched controls,
   or a documented blocked or null outcome.
9. A load curve to saturation with a named knee and a reproduced failure and
   recovery.
10. A correlated per-stage latency budget naming the p99 owner.
11. A technical report in which every claim points to a table, figure, or
    experiment, and every blocked or partial capability appears as a limitation.
12. One-command reproduction of at least one benchmark from a clean environment.

Negative results are measurements and belong in the report. The
[optional systems drills](SPEECH_ML_SYSTEMS_DRILLS.md) are study material, not
release gates. Archived PDFs are historical and remain unchanged.

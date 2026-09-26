# External Restoration Comparator: Feasibility and Capability Record

> **v2 correction, September 26, 2026:** This replaces the original Day 08
> candidate notes, not a measured experiment. No successful local installation,
> smoke test, or masked-inpainting capability is established by this document.

## 1. Candidate and Verified Documentation

Retain **VoiceFixer** as the single candidate for a general speech-restoration
comparison. Its official README describes restoration of noise, reverberation,
bandwidth degradation, and clipping. The documented `restore` call takes input
and output paths, a CUDA option, and a mode; it does not expose an explicit
damaged-span mask. Mode 0 is the original model, mode 1 adds preprocessing,
and mode 2 is described as train mode. Mode 2 is not evidence of an inpainting
interface. The supplied vocoder uses 44.1 kHz, so output format must be verified
and converted explicitly for the project's 16 kHz evaluation.

Primary documentation checked September 26, 2026:

```text
https://github.com/haoheliu/voicefixer
```

**Current status:** candidate only; repository-level feasibility is unverified.
Do not report packet-loss recovery, masked reconstruction, exact waveform
preservation, determinism, runtime, or language coverage without measurements.

## 2. Bounded Feasibility Check

Use one setup session plus at most one focused compatibility retry in Week 2.
Do not install another model or implement an interpolation/phase-reconstruction
fallback to avoid reporting a blocker.

Before the comparator enters a benchmark, record:

- Pinned code, package and weight revisions, permitted use, and environment.
- Exact invocation, native sample rate, channel count, sample count, and any
  resampling/alignment applied by the adapter.
- Whether a mask is accepted; whether the output changes samples outside a
  target interval; whether processing is full-utterance or local.
- One clean and one damaged smoke-test case, source IDs, parameters and seed,
  repeatability observations, and expected L4 memory/cost.
- Final `status`: `feasible` only after a verified smoke test, otherwise `deferred`
  when the bounded attempt stops. Record `attempt_outcome=blocked` and the error
  or stop reason where applicable. The current unverified candidate is not yet
  either a successful check or an executed experiment.

The planned adapter is `src/baselines/direct_audio_restore.py`. It records
capabilities rather than implying that every backend performs masked inpainting.
Do not create it until the candidate passes the feasibility check.

## 3. Comparison Contract

Day 54 evaluates the verified comparator on identical damaged cases alongside
raw damaged audio, full resynthesis, naive selective repair, and boundary-matched
selective repair. Record output format, locality, reference access and hardware.
Only compare L4 timing/memory under controlled conditions. Mark unsupported
metrics or failed cases explicitly instead of inventing values or hiding failures.

Normal MendSpeech repair uses predicted text. A supplied correct transcript or
ground-truth mask is privileged information and must be disclosed in a separately
labeled oracle comparison. Full-utterance restoration is not proof of local
masked reconstruction or preservation of untouched speech.

If the candidate remains blocked, retain the core controls and publish the
external-comparator limitation. The masked-inpainting question remains deferred;
the report must not claim that this question was evaluated or resolved.

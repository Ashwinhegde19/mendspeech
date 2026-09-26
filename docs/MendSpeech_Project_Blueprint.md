# MendSpeech Project Blueprint

> **A Real-Time Voice Interface: Streaming ASR, Latency Budget, and Personalization.**
>
> **v3 scope:** One streaming recognition pipeline, one LLM post-processing
> stage, one serving endpoint, and one evaluation harness. The
> [execution plan](REVISED_EXECUTION_PLAN.md) governs optimization,
> personalization, RL, and the latency budget. These are target behaviors, not
> claims that the current implementation is complete.

---

### Core Principle
> *Measure the whole pipeline, find what actually limits it, improve that, and never hide an architectural limitation.*

---

## 1. Product Behavior
- Accept live microphone audio, a stream of audio chunks, or an uploaded file.
- Optionally generate controlled acoustic conditions through `SpeechDamageBench`.
- Transcribe incrementally with cache-aware streaming ASR.
- Emit partial and final transcripts with word-level timestamps and confidence.
- Triage each utterance: **accept**, **low confidence**, or **reject**.
- Post-process the accepted transcript into polished text with one small LLM.
- Report, as a single decomposed number, where the end-to-end latency lives.

---

## 2. Pipeline Stages

| Stage | Component | What Is Measured |
| :--- | :--- | :--- |
| **Capture and VAD** | Frame-level energy/spectral VAD | Precision/recall/F1, onset/offset error in ms, CPU RTF |
| **ASR** | Cache-aware streaming Conformer | WER/CER, per-corruption accuracy, clean regression |
| **Decode** | Greedy / beam / beam + n-gram LM | WER/CER, names/numbers, helpful *and* harmful changes, decoder-only vs fresh latency |
| **Confidence** | Calibrated score for the exact model+decoder+precision | ECE/Brier, reliability diagram, confident-but-wrong cases |
| **Personalization** | Fine-tuning, then RL post-training | Held-out WER, clean regression, adaptation vs RL |
| **LLM** | One small pinned post-processing model | Time-to-first-token, full response, prefix-cache hit rate, quality |
| **Serving** | One async WebSocket endpoint | Concurrency knee, queue depth, p50/p95/p99, backpressure, failure/recovery |

Latency is reported **per stage**, never as a single blended average. Cached
decoder time is not fresh audio-to-transcript time. LLM time-to-first-token is
not full response time. A tail is identified by a percentile, not a mean.

---

## 3. Robustness Evaluation Suite

`SpeechDamageBench` is a standalone, versioned package, not a private utility.

| Damage Family | Controlled Variables | Purpose |
| :--- | :--- | :--- |
| **Additive Noise** | SNR, noise type, random seed | Test masking robustness. |
| **Clipping** | Threshold, severity | Test lost peaks and saturation. |
| **Bandwidth Limits** | Sample rate, filter settings | Simulate narrow channels (telephony, codecs). |
| **Dropouts** | Span length, frequency, random seed | Simulate missing speech and packet loss. |
| **Reverberation** | Impulse response / room severity | Test temporal smearing. |

> [!NOTE]
> Every generated sample records corruption name, severity, random seed, clean source ID, parameter values, and package version.

---

## 4. Metrics

| Metric | Why It Matters |
| :--- | :--- |
| **WER & CER** | Recognition correctness, per corruption and severity. |
| **Names and numbers error rate** | Entity errors that a blended WER can hide. |
| **Latency percentiles (p50/p95/p99)** | Responsiveness and tail behavior. A mean hides the tail. |
| **Time to first partial transcript** | What the user actually perceives first. |
| **LLM time-to-first-token** | Separates first-token responsiveness from full response. |
| **Real-Time Factor (RTF)** | Whether processing keeps up with live speech. |
| **Peak GPU memory** | Deployment cost and memory pressure. |
| **Throughput / concurrency knee** | Where added load stops being free. |
| **Prefix-cache hit rate** | Whether repeated system context is being reused. |
| **Calibration (ECE / Brier)** | Whether confidence supports triage decisions. |
| **Confident-but-wrong rate** | The failure that breaks a confidence-gated system. |
| **Helpful vs harmful LM changes** | Lower WER does not imply safer output. |
| **Cost per 1000 hours of audio** | The number an infra team budgets with. |

---
## 5. Required Ablations
- **Decoding:** Greedy vs. beam-only vs. beam plus one LM; validation-only tuning, unchanged acoustic model, helpful and harmful text changes.
- **Context policy:** Supported fixed low/high lookahead, plus a bounded adaptive experiment labelled live, simulated, or unavailable.
- **Confidence:** Raw vs. calibrated confidence, per model/decoder/precision.
- **Triage thresholds:** Accept / low-confidence / reject policies, selected on validation.
- **Optimization:** Each technique (quantization, `torch.compile`, CUDA graphs, batching) measured independently against the same baseline, on fixed L4 and batch discipline.
- **Streaming fast path:** Cached versus uncached state, measured separately.
- **Personalization:** Base vs. fine-tuned vs. RL, on held-out data with a clean-speech regression check.
- **LLM stage:** Enabled vs. disabled, batched vs. streamed generation, cold vs. warm prefix cache.
- **Serving:** One provider and one endpoint; concurrency sweep to saturation with one reproduced failure and recovery.
- **Clean-speech regression:** Already-clean speech must not be degraded by the pipeline.

---
## 6. Repository Target Structure

```text
mendspeech/
├── src/
│   ├── audio/          # Waveform loaders, STFT, log-Mel, normalization
│   ├── asr/            # Streaming ASR, CTC decode, confidence, calibration
│   ├── streaming/      # Cache-aware runners, lookahead, endpointing
│   ├── vad/            # Frame-level VAD baseline and comparison
│   ├── rl/             # Reward definition and policy-gradient update
│   ├── llm/            # LLM post-processing adapter
│   ├── serve/          # Async WebSocket service and load harness
│   ├── metrics/        # WER, CER, RTF, latency percentiles, calibration
│   └── bench/          # One benchmark harness used by every experiment
├── speechdamagebench/  # Standalone versioned robustness suite
├── infra/              # Modal execution scripts and container definitions
├── app/                # audio_lab.py is the evolving demo; shared UI components
├── training/           # Fine-tuning and RL entry points
├── configs/            # Frozen experiment configurations
├── experiments/        # Frozen experiment configs
├── results/            # Measured tables and figures; audio/checkpoints/logs ignored
└── reports/            # Latency budget, technical report, casebooks
```

---
## 7. Definition of Done
1. A new user can reproduce benchmark results with documented single-command sequences.
2. The interactive demo streams live or prerecorded audio and shows partial/final transcripts with confidence and latency.
3. The latency budget report decomposes every stage with p50/p95/p99 and names the tail owner.
4. At least one optimization is applied end-to-end with a measured before/after, or the negative result is documented with evidence.
5. Fine-tuning and RL are compared against baseline on held-out data with a clean-speech regression check, or a blocker is recorded.
6. The serving endpoint is load-tested to saturation with one reproduced failure and recovery.
7. The LLM stage is measured for TTFT and full response, or its blocker is recorded.
8. You can explain every major component from first principles without relying on library names.
9. Benchmark results are reported at a fixed documented scale (>=30 utterances, >=5 speakers, speaker-separated splits) with the statistical caveat stated.
10. The report contains at least one surprising result and one limitation that materially constrains its claims.
11. Every deferred or blocked capability appears explicitly in the limitations section.
12. Release evidence is independent of optional learning drills and extra UI pages.

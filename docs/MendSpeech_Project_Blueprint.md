# MendSpeech Project Blueprint

> **Meaning-preserving dictation: streaming ASR, conservative transcript
> editing, and a measured latency budget.**
>
> **v4 scope.** One streaming recognition pipeline, one conservative editor, one
> serving endpoint, and one evaluation harness. The
> [execution plan](REVISED_EXECUTION_PLAN.md) governs phases and gates; the
> [latency and quality contract](LATENCY_AND_QUALITY_CONTRACT.md) and the
> [editor and RL contract](EDITOR_AND_RL_CONTRACT.md) govern measurement and
> post-training. Everything below is a target behavior, not a claim that the
> current implementation is complete.

---

### Core principle

> *Never present text the speaker did not say. Improve readability only where a
> guard can prove the meaning is unchanged, and measure the whole pipeline
> honestly.*

---

## 1. Product behavior

- Accept live microphone audio, a stream of chunks, or an uploaded recording.
- Optionally generate controlled acoustic conditions via `SpeechDamageBench`.
- Transcribe incrementally with cache-aware streaming ASR.
- Emit partial and final transcripts with word timestamps and raw confidence.
- Fit calibration and triage each utterance: **accept**, **low confidence**, or
  **reject**.
- Offer the accepted transcript to a conservative editor that may format but not
  rewrite.
- Validate every proposed edit before delivery; otherwise return the raw
  transcript with a reason.
- Report where the latency lives, per stage, per request, with tail attribution.

Three invariants:

1. The raw ASR transcript is always available.
2. A rejected, malformed, timed-out, or low-confidence edit returns raw text
   with a reason code — never an invented sentence.
3. Names, numbers, negation, and units are never altered by the editor.

---

## 2. Pipeline stages

| Stage | Component | Measured |
| :--- | :--- | :--- |
| **Endpointing** | Frame-level VAD plus an endpoint state machine | Precision/recall, onset/offset error in ms, endpoint-to-final delay |
| **ASR** | Cache-aware streaming Conformer | WER/CER per corruption, names/numbers/negation, clean regression |
| **Decode** | Greedy / beam / beam + one n-gram LM | Accuracy, helpful *and* harmful changes, cached decoder vs fresh latency |
| **Confidence** | Score calibrated for the exact model, head, decoder, and precision | Reliability, ECE/Brier, risk-coverage, confident-but-wrong cases |
| **Triage** | Accept / uncertain / reject, fitted on validation | Coverage versus error rate, bypass rate |
| **Editor** | One pinned small causal LM, formatting only | Protected-content violations, formatting accuracy, identity vs needs-edit, fallback rate, independent review |
| **Guard** | Deterministic pass/fail before delivery | Rejection rate, rejected-edit quality |
| **Post-training** | Editor SFT, compute-matched control, bounded GRPO | Improvement against controls, violation rate, reward-validity checks |
| **Robustness** | ASR fine-tuning for acoustic conditions | Held-out WER, clean-speech regression, cost |
| **Serving** | One async WebSocket endpoint | Concurrency knee, queue depth, percentiles, backpressure, failure and recovery |

Latency is reported **per stage and per request**. Cached decoder time is not
fresh audio-to-transcript latency. Time-to-first-token is not completion.
Percentiles are not additive, and a tail is attributed by inspecting the same
slow requests rather than by comparing independent distributions.

---

## 3. Robustness evaluation suite

`SpeechDamageBench` is a standalone, versioned package.

| Damage family | Controlled variables | Purpose |
| :--- | :--- | :--- |
| **Additive noise** | SNR, noise type, random seed | Masking robustness |
| **Clipping** | Threshold, severity | Lost peaks and saturation |
| **Bandwidth limits** | Sample rate, filter settings | Narrow channels (telephony, codecs) |
| **Dropouts** | Span length, frequency, random seed | Missing speech and packet loss |
| **Reverberation** | Impulse response, room severity | Temporal smearing |

> Every generated sample records corruption, severity, seed, source ID,
> parameters, and package version. The frozen evaluation set is never enlarged
> to improve a result; new experiments use new configurations.

---

## 4. Metrics

| Metric | Why it matters |
| :--- | :--- |
| **WER / CER** | Recognition correctness, sliced by corruption and severity |
| **Names, numbers, negation error** | Entity errors a blended WER hides |
| **Confidence reliability (ECE, Brier)** | Whether confidence supports triage |
| **Confident-but-wrong rate** | The failure that breaks a confidence-gated system |
| **Risk-coverage** | Error rate traded against coverage kept |
| **Endpoint error (ms)** | When a final answer is sent |
| **Per-stage p50/p95/p99** | Where responsiveness is actually lost |
| **Time to first partial** | First user-perceived feedback |
| **TTFT and completion** | Separating first-token responsiveness from full output |
| **RTF, peak memory, combined footprint** | Whether the pipeline keeps up and fits |
| **Throughput and concurrency knee** | Where added load stops being free |
| **Protected-content violation rate** | Whether editing changed meaning |
| **Fallback and edit coverage** | How often the editor abstains or acts |
| **Cost per 1000 audio hours** | Extrapolated from measured cost, with assumptions stated |

---

## 5. Required ablations

- **Decoding:** greedy vs beam-only vs beam + one LM; validation-only tuning,
  with helpful and harmful changes reported.
- **Context:** supported fixed lookahead settings, with algorithmic latency
  measured rather than asserted.
- **Confidence and triage:** raw vs calibrated confidence; threshold policies
  fitted on validation.
- **Editor:** raw vs deterministic formatting vs prompt-only vs SFT vs RL, on
  the same held-out cases, evaluating both proposed and delivered output.
- **Post-training:** RL vs supervised editing vs a compute-matched control, so
  extra training cannot masquerade as the algorithm.
- **Robustness:** base vs robustness-adapted ASR, with clean-speech regression.
- **Optimization:** each technique measured independently against one baseline,
  on fixed hardware and batch discipline, with parity checks.
- **Serving:** one provider, one endpoint, concurrency swept to saturation, with
  one reproduced failure and recovery.

---

## 6. Repository target structure

```text
mendspeech/
├── src/
│   ├── audio/          # Waveform loaders, STFT, log-Mel
│   ├── asr/            # Streaming ASR, decoding, confidence, calibration
│   ├── streaming/      # Session loop, cache, context, endpointing
│   ├── vad/            # Deterministic VAD baseline
│   ├── controller/     # Triage policy and bounded adaptive context
│   ├── llm/            # Editing contract, deterministic baseline, editor adapter, guard
│   ├── rl/             # Reward definition and group-relative training entry points
│   ├── serve/          # Async WebSocket service, batching, load harness
│   ├── metrics/        # WER, entity error, risk-coverage, timing
│   └── bench/          # One benchmark harness, tracing, profiling, budget
├── training/           # ASR fine-tuning and editor post-training entry points
├── configs/            # Frozen experiment configurations
├── experiments/        # Ablations and the frozen protocol
├── speechdamagebench/  # Standalone versioned robustness suite
├── infra/              # Modal and container definitions
├── app/                # audio_lab.py, the single evolving demo
├── results/            # Measured tables and figures
└── reports/            # Latency budget, casebook, technical report
```

---

## 7. Definition of done

1. A new user reproduces benchmark results with documented single commands.
2. The demo streams audio and shows partial and final transcripts, confidence,
   triage actions, the guarded edit, and the measured latency budget.
3. The latency budget report decomposes every stage with percentiles and names
   the p99 owner by request identifier.
4. At least one optimization ships with a measured before/after, or the
   negative result is documented with evidence.
5. Confidence is fitted and independently evaluated for the shipping
   configuration, distinct from threshold selection.
6. Editor quality is measured on held-out cases, including protected-content
   violations and an independent review, with raw text always recoverable.
7. Post-training is compared against supervised and compute-matched controls,
   or the blocker is recorded.
8. The serving endpoint is load-tested to saturation with a reproduced failure
   and recovery.
9. Every major component can be explained from first principles without relying
   on library names.
10. Results are reported at a fixed documented scale, with the statistical
    caveat stated.
11. Every blocked, deferred, or partial capability appears explicitly in the
    limitations section.

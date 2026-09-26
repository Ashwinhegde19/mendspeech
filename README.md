# MendSpeech

**Meaning-preserving dictation: streaming ASR, conservative transcript editing, and a measured latency budget.**

Live dictation is not clean audio. It arrives with noise, clipping, missing
packets, background speech, and words the model has never heard. MendSpeech is
a real-time voice interface built to survive that: cache-aware streaming ASR,
calibrated confidence, one conservative editor, and bounded post-training of
that editor.

Everything is measured: word error, entity error, latency percentiles,
real-time factor, calibration, and peak memory.

The project answers three questions, in order:

1. **Where does the end-to-end latency budget actually live?** Decompose
   waveform to polished text stage by stage, find what owns the tail, improve
   it, and show the before/after.
2. **How should a bounded post-training budget be split between supervised
   editing and RL, and what does each buy?** A null result is still a result.
3. **When is confidence safe to act on?** Calibrate it against correctness and
   find the cases where a high score is still wrong.

---

## What ships

| Product | Role |
| :--- | :--- |
| **MendSpeech** | Streaming recognition, calibrated confidence, a triage policy, a conservative editor with a guard, bounded editor post-training, acoustic robustness adaptation, and a measured latency budget (`src/`). |
| **SpeechDamageBench** | A standalone, versioned robustness suite (noise, clipping, bandwidth limits, dropouts, reverberation). Every sample records corruption, severity, seed, and source. Usable without MendSpeech. |

The release is one streaming pipeline, one serving endpoint, one LLM stage,
and one evaluation harness. Architecture learning exercises support this system;
they do not create parallel products.

Working format: 16 kHz mono float32. Speech defaults for analysis windows
are 25 ms FFT / 10 ms hop.

---

## Status

Early build. Days 01–09 are complete: the audio lab, `SpeechDamageBench`, the
frozen labeled benchmark, CTC from first principles, and a Wav2Vec2 baseline. Milestones live in the
[execution plan](docs/REVISED_EXECUTION_PLAN.md). Architecture and the
definition of done live in the
[blueprint](docs/MendSpeech_Project_Blueprint.md).

| Subsystem | Purpose | Status |
| :--- | :--- | :--- |
| `src/audio` | Waveform I/O, resampling, STFT, log-Mel features | **exists** (tested) |
| SpeechDamageBench | Deterministic damage generation + frozen evaluation sets | in progress |
| `src/asr` | Decoding, token confidence, calibration | planned |
| `src/streaming`, `src/vad` | Session loop, cache, lookahead, endpointing | planned |
| `src/controller` | Triage policy, bounded adaptive context | planned |
| `src/llm` | Editing contract, deterministic baseline, editor adapter, guard | planned |
| `src/rl` | Reward definition and group-relative post-training | planned |
| `src/serve`, `src/bench` | Async endpoint, load harness, tracing, latency budget | planned |

Audio files and checkpoints are gitignored. Manifests and measured
results are tracked. The [results index](results/README.md) ties every
committed artifact to the finding that produced it.

---

## Quickstart

```bash
git clone <this-repo> && cd mendspeech
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest
```

Python ≥ 3.10. Core dependencies: `torch`, `torchaudio`, `soundfile`,
`librosa`, `scipy`, `matplotlib`, `pandas`, `numpy`. No GPU is required
for the audio stack. Cloud measurements use Modal L4 so latency and RTF
numbers stay on one hardware tier.

```python
from src.audio.loader import load_audio
from src.audio.stft import spectrogram_db

wave, sr = load_audio("clip.wav", target_sr=16000)
spec = spectrogram_db(wave, n_fft=400, hop_length=160)
```

---

## How the system is built

Work proceeds in measured slices: implement the next subsystem, run a
controlled experiment, record the number, keep going. The
[blueprint](docs/MendSpeech_Project_Blueprint.md) is the architecture
contract. The [execution plan](docs/REVISED_EXECUTION_PLAN.md) is the
pacing and gate contract.

| Phase | What gets built |
| :--- | :--- |
| Audio lab | Loaders, STFT, log-Mel, SpeechDamageBench v0, frozen labeled eval set |
| Recognition | WER/CER, confidence, timestamps, triage policy, decoding comparison |
| Streaming | Cache-aware inference, VAD/endpointing, lookahead cost, cache-failure evidence |
| Optimization | Profiling, `torch.compile`, CUDA graphs, batching, quantization, scorecard |
| Post-training | Editor reward, SFT with a compute-matched control, bounded GRPO, and ASR robustness adaptation |
| Serving | Async WebSocket endpoint, load to saturation, editor selection, **correlated latency budget** |
| Release | Frozen evaluation, ablations, technical report, reproduction, demo |

Session notes and experiment specs live under [`docs/`](docs/INDEX.md).
Contributor rules — code style, determinism, git, compute — are in
[`AGENTS.md`](AGENTS.md).

---

## Documentation

- [Project blueprint](docs/MendSpeech_Project_Blueprint.md) — architecture, metrics, definition of done
- [Execution plan](docs/REVISED_EXECUTION_PLAN.md) — gates, scope, compute
- [Roadmap](docs/MendSpeech_8_Week_Master_Roadmap.md) — build order and depth
- [Docs index](docs/INDEX.md) • [Results index](results/README.md)

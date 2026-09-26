# MendSpeech

**Selective semantic speech restoration under real-time constraints.**

Speech recordings get damaged — packet loss, clipping, noise, dropouts.
MendSpeech does not resynthesize everything. It detects *which spans* are
untrustworthy using calibrated ASR uncertainty, decides whether to
**preserve, inspect, repair, or abstain**, and reconstructs only what the
evidence justifies.

Everything is measured: word error, latency percentiles, real-time factor,
calibration error, how much original audio was kept, and how clean the
seams are.

The project also answers a concrete architectural question on the same
damaged spans:

**selective ASR → TTS repair vs. full resynthesis and a verified audio-restoration
comparator** — where does each path win, and what does each throw away?
Masked-inpainting claims require an actually verified mask-aware baseline;
blocked external comparisons are reported as limitations, not results.

---

## What ships

| Product | Role |
| :--- | :--- |
| **MendSpeech** | Streaming recognition, calibrated uncertainty, a preserve / inspect / repair / abstain policy, and selective reconstruction with boundary-matched stitching (`src/`). |
| **SpeechDamageBench** | A standalone, versioned damage generator (noise, clipping, bandwidth limits, dropouts, reverberation). Every sample records corruption, severity, seed, and source. Usable without MendSpeech. |

The focused release uses one ASR pipeline, one selected TTS stack, one evolving
application, and one shared evaluation suite. Architecture learning exercises
support this system; they do not create parallel products.

Working format: 16 kHz mono float32. Speech defaults for analysis windows
are 25 ms FFT / 10 ms hop.

---

## Status

Early build. Audio foundations are in place; recognition, streaming, and
repair come next. Milestones live in the
[execution plan](docs/REVISED_EXECUTION_PLAN.md). Architecture and the
definition of done live in the
[blueprint](docs/MendSpeech_Project_Blueprint.md).

| Subsystem | Purpose | Status |
| :--- | :--- | :--- |
| `src/audio` | Waveform I/O, resampling, STFT, log-Mel features | **exists** (tested) |
| SpeechDamageBench | Deterministic damage generation + frozen evaluation sets | in progress |
| `src/asr` | FastConformer, CTC, token confidence, calibration | planned |
| `src/streaming` | Cache-aware real-time inference, lookahead control | planned |
| `src/controller` | Preserve / inspect / repair / abstain policy | planned |
| `src/tts`, `src/repair` | Speaker-conditioned synthesis, boundary matching, seam diagnostics | planned |
| `src/baselines`, `src/metrics` | Verified restoration comparison; WER, RTF, ECE, seam scores | planned |

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
| Recognition | Pretrained ASR, CTC, confidence, time-aligned uncertain spans |
| Encoders | One tested Conformer block from first principles; measured FastConformer baseline |
| Streaming | Cache-aware inference, VAD/endpointing, adaptive lookahead |
| Robustness | Fine-tuning on damage, quantization, calibration |
| Repair | One speaker-conditioned TTS stack, bounded adaptation, boundary matching, seam diagnostics |
| Comparison | Controlled repair baselines and one feasibility-checked restoration comparator |

Session notes and experiment specs live under [`docs/`](docs/INDEX.md).
Contributor rules — code style, determinism, git, compute — are in
[`AGENTS.md`](AGENTS.md).

---

## Documentation

- [Project blueprint](docs/MendSpeech_Project_Blueprint.md) — architecture, metrics, definition of done
- [Execution plan](docs/REVISED_EXECUTION_PLAN.md) — gates, scope, compute
- [Roadmap](docs/MendSpeech_8_Week_Master_Roadmap.md) — build order and depth
- [Docs index](docs/INDEX.md) • [Results index](results/README.md)

# MendSpeech

**Meaning-preserving dictation: streaming ASR, confidence calibration, and
conservative transcript editing.**

MendSpeech is an experimental voice-AI platform. The current repository
contains reusable audio, ASR, metrics, benchmark, and evaluation components.
The long-term product direction is to identify speech-recognition failures,
select valuable training examples, correct transcripts, fine-tune existing ASR
models, and evaluate improvements on unseen speech, including Hindi-English
(Hinglish) recognition.

## Current status

The codebase is an early build. Implemented and tested in this repository:

- Waveform ingestion, resampling, STFT, and log-Mel feature extraction
- SpeechDamageBench, a standalone deterministic damage-generation package
- Frozen LibriSpeech benchmark manifests with speaker-separated data roles
- Word and character error-rate scoring, with Levenshtein substitution,
  deletion, and insertion counts
- Text normalization, protected-content slicing, and leakage checks
- Wav2Vec2 CTC ASR baseline decoding and token confidence definitions
- Interactive Gradio audio comparison console
- Modal L4 ASR entry point and reproducible experiment scripts

Planned, but not yet implemented:

- Hindi-English failure analysis and data selection
- Human transcript correction workflows
- Targeted fine-tuning and evaluation harness for Hinglish ASR
- Streaming ASR, triage policy, conservative editor, serving, and latency work
  described in the blueprint and contracts

## Quickstart

```bash
git clone <this-repo> && cd mendspeech
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest
```

Python >= 3.10 is required. Core dependencies include `torch`,
`torchaudio`, `soundfile`, `librosa`, `scipy`, `matplotlib`, `pandas`, and
`numpy`.

Example:

```python
from src.audio.loader import load_audio
from src.audio.stft import spectrogram_db

wave, sr = load_audio("clip.wav", target_sr=16000)
spec = spectrogram_db(wave, n_fft=400, hop_length=160)
```

## Repository structure

| Path | Purpose |
| :--- | :--- |
| `src/audio/` | Waveform I/O, resampling, STFT, log-Mel features |
| `src/asr/` | Wav2Vec2 CTC baseline and confidence definitions |
| `src/metrics/` | WER/CER, normalization, data roles, protected-content slicing |
| `speechdamagebench/` | Standalone deterministic damage-generation package |
| `data/` | Benchmark manifests and metadata only |
| `tests/` | pytest suite for reusable modules and metrics |
| `docs/` | Architecture, contracts, evaluation, and technical references |
| `infra/` | Modal L4 ASR entry point |
| `app/` | Gradio clean-versus-damaged comparison console |
| `scripts/` | Reproducible experiment entry points |

## Documentation

- [Project blueprint](docs/MendSpeech_Project_Blueprint.md) — architecture, metrics, and definition of done
- [Execution plan](docs/REVISED_EXECUTION_PLAN.md) — scope, phases, and release gates
- [Latency and quality contract](docs/LATENCY_AND_QUALITY_CONTRACT.md) — workload and comparison rules
- [Editor and RL contract](docs/EDITOR_AND_RL_CONTRACT.md) — conservative editing and evaluation rules
- [Data roles](reports/data_roles.md) — frozen benchmark role declaration and leakage checks
- [Results index](results/README.md) — committed measured artifacts
- [SpeechDamageBench](speechdamagebench/README.md) — standalone damage-generation package

## Learning branch

Session-level learning material, day specifications, notebooks, and day-specific
measurements are preserved on `research/learning`.
```
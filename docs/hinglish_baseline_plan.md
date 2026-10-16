# Hinglish ASR Baseline Plan

This document describes the initial Hinglish baseline work on
`feat/hinglish-asr-baseline`. It does not claim measured ASR improvements.

## Baseline dataset

- Dataset: `dianavdavidson/MUCS-Hinglish`
- Original source: https://openslr.org/104/
- License: CC BY 4.0
- Scale: 52,825 training utterances across 520 speakers, 3,136 test
  utterances across 30 speakers
- Sample rate: 16 kHz
- Columns include segment IDs, speaker IDs, transcripts, language ratios,
  English word counts, and Hindi word counts

## Baseline model

- Primary: `openai/whisper-small` with multilingual transcription
- Additional candidate comparisons may use approved Whisper variants after
  the first reliable baseline
- Comparison against existing Wav2Vec2 English-only baseline is diagnostic, not
  a complete Hinglish ASR strategy

## Official splits

- Training set: model and data experiments
- Validation set: speaker-separated carve-out from training speakers
- Test set: untouched until final baseline reporting

## Current foundation

- `src/hinglish/`: normalization, MUCS metadata, code-switch metrics, and
  offline prediction scoring
- `scripts/prepare_mucs_hinglish_manifest.py`: streams metadata without
  downloading audio
- `scripts/evaluate_hinglish_predictions.py`: scores prediction CSVs
- `infra/modal_hinglish_asr.py`: explicit T4 or L4 inference with a safe
  dry-run mode
- Unit tests for normalization, data, metrics, evaluation, and Modal dry-run

## GPU use

GPU inference must use Modal. The first GPU step must be a dry run, followed by
a separately approved 5 to 10 utterance smoke test. Full evaluation is not
automatic.
# Results Index

Measured artifacts produced by the daily sessions. Day numbers refer to the
plans in [`docs/days/`](../docs/days/) — the experiment registry for this
project. Result files follow the `dayNN_<what>.<ext>` naming convention.

| Day | Artifact | Finding |
| :--- | :--- | :--- |
| 01 | `day01_resampling_comparison_table.csv` | Downsampling to 8/16/24/48 kHz preserves RMS level (−18.5 dB across all rates) but halves bandwidth each step — everything above the Nyquist frequency is gone for good. |
| 01 | `day01_sampling_rate_spectrum_comparison.png` | Visual proof of the table: spectra at each sample rate truncate at their Nyquist frequency; speech formant energy survives at 16 kHz, so 16 kHz is the project's working rate. |
| 02 | `day02_stft_parameter_grid.png` | STFT tradeoff measured on one utterance: 16 ms windows resolve plosive timing but blur harmonics; 128 ms windows resolve harmonics but smear onsets. Hop changes sampling density only, not resolution. Speech standard: 25 ms window / 10 ms hop at 16 kHz (n_fft=400, hop=160). |
| 03 | `day03_mel_bins_comparison.png` | Mel bin count sweep on one 6 s utterance: 40/80/128 bands → 93.9/187.8/300.5 KB per feature matrix, visual detail grows but 128 bands at n_fft=400 leave top filters with zero linear-bin coverage (torchaudio warning) — 80 bands is the supported ASR standard. Log-Mel shape contract: n_mels fixed, frames scale with duration (601 frames @ 6.00 s vs 651 @ 6.50 s). |
| 04 | `day04_determinism.csv` | Same seed (7) reproduces an identical medium-noise waveform; seed 8 changes the realization while SNR stays 10 dB. Clipping and bandwidth ignore the seed (they are parameter-only) and still record it. |
| 04 | `day04_severity_grid.png` | One labeled LibriSpeech sentence, seed 7: mild / medium / severe for noise, clipping, bandwidth, dropout, and reverberation. Length and 16 kHz rate stay fixed. |
| 05 | `week1_damage_metrics.csv` | 150 rows: 10 clips × 5 corruptions × 3 severities. Additive noise SNR matches presets exactly (20/10/0 dB). Reverberation produces negative SNR (-8 to -3 dB) despite sounding natural at mild severity — SNR is misleading for correlated distortions. Clipping mild can be a no-op when peak < threshold. Bandwidth loss yields low SNR even at mild severity while speech remains intelligible. See docs/metric_limitations.md. |
| 06 | `week1_audio_console.png` | Gradio console for side-by-side clean/damaged playback with waveform, spectrogram, and seed-aware measurements. Desktop comparison columns remain paired, while the layout collapses to one column on narrow screens. |

| 07 | `day07_recreation.png` | Blank-notebook recreation: log-Mel (80 bins, 400/160) scales with duration (1496 frames @ 14.95 s) and additive_noise medium seed 7 reproduces SNR 10.00 dB deterministically — validates clean→corruption→feature path is teachable. |
| 08 | `day08_baseline_transcripts.csv` | 30 runs: 5 clean clips × (1 clean + 5 medium corruptions). Clean ASR confidence averages 0.969; additive noise drives confidence down to 0.886 and produces phonetic substitutions/omissions; clipping preserves intelligibility at 0.970; reverberation and bandwidth loss cause perceptual misrecognitions while confidence stays relatively high, confirming raw softmax confidence alone requires calibration. |

| 10 | `day10_wer_by_damage.csv` | 192 runs: 12 utterances (calibration + validation roles) × (1 clean + 5 corruptions × 3 severities), scored against real LibriSpeech ground truth. Clean WER 1.22% (4 edits / 374 words) is the genuine model floor, not a self-comparison. Damage ordering is additive_noise (86.74% WER at severe) > reverberation (47.54%) > bandwidth (14.58%) > dropout (13.01%) > clipping (1.48%); clipping is nearly lossless at every severity. WER exceeds CER everywhere, ~2.5× at severe noise. |
| 10 | `day10_error_types.csv` | Substitution/deletion/insertion totals per condition. Substitutions dominate until severe noise, where deletions overtake them (197 S vs 148 D) — severe noise destroys words rather than corrupting them. Insertions stay near zero (max 4), so the baseline rarely hallucinates words on this corpus. |
| 10 | `day10_slice_rates.csv` | Protected-content slice rates. Severe additive noise drives negation to 80% and numbers to 100% error, far above the 37.5% overall WER — damage concentrates on meaning-bearing content. **Name slice is `null`, not 0.0**: all 11 name words in the corpus fall in the train and final-test roles, so the scored subset contains none. Reported as not-measured per the protocol. |
| 10 | `day10_raw_runs.csv` | One row per scored run (192), carrying clip ID, speaker, role, seed, severity, confidence, WER/CER, S/D/I counts and per-slice counts. Supports re-aggregation without re-running ASR. |
| 11 | `day11_confidence_by_damage.csv` | 256 confidence-bin rows from 96 validation runs (6 utterances × 16 conditions). Words binned by confidence show 99.46% accuracy at p ≥ 0.95 but only 38.5% correct below 0.90 — high confidence predicts correctness, low confidence does not predict error. The Day 10 aggregate correlation (r = −0.99) does not describe per-word behavior. |
| 11 | `day11_token_scores.csv` | 2,432 per-word rows with score, alignment state and correctness. Contains exactly 10 confident errors (p ≥ 0.95, wrong) and 137 low-confidence-correct words. Deleted words carry no probability by design. |

## Naming rules

- Result and notebook files carry the day prefix (`dayNN_<what>.<ext>`).
- Source and test modules use functional names (`src/audio/stft.py`) — no day counts in permanent code.
- Audio, checkpoints, and run logs are gitignored; only manifests, tables, figures, and notes are tracked.

# Week 1

> **Days 01–07**
> **Navigation:** [← Index](INDEX.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Executive Plan](REVISED_EXECUTION_PLAN.md)

---

> [!IMPORTANT]
> **Week theme:** Audio DSP and the deterministic damage suite
> Audio foundations and the standalone damage suite are complete; treat this week as frozen evidence.

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 01** | Waveforms, sampling, and the MendSpeech baseline | `Local CPU` | CORE | [Open Day 01](days/day_01.md) |
| **Day 02** | Fourier intuition and STFT | `Local CPU` | CORE | [Open Day 02](days/day_02.md) |
| **Day 03** | Mel scale and log Mel features | `Local CPU` | CORE | [Open Day 03](days/day_03.md) |
| **Day 04** | Build SpeechDamageBench as a standalone package | `Local CPU` | CORE | [Open Day 04](days/day_04.md) |
| **Day 05** | Objective audio measurements | `Local CPU` | CORE | [Open Day 05](days/day_05.md) |
| **Day 06** | Build the first MendSpeech audio console | `Local CPU` | CORE | [Open Day 06](days/day_06.md) |
| **Day 07** | Review, explain, and freeze Week 1 | `Local CPU` | CORE | [Open Day 07](days/day_07.md) |

---

## Daily Detailed Operating Plans

### DAY 01: Waveforms, sampling, and the MendSpeech baseline
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_01.md`](days/day_01.md)


#### Learn
- Waveform amplitude and time axes.
- Sampling rate, Nyquist intuition, bit depth, mono versus stereo.
- Why speech systems often standardize to 16 kHz.
- Duration, peak amplitude, RMS energy, and clipping.

#### Build in MendSpeech
- Create the repository and a minimal audio loader.
- Record or collect five clean speech clips with consent.
- Acquire a reference-transcripted subset (e.g., a LibriSpeech dev-clean slice) into `data/benchmark/`; the frozen benchmark will run on labeled clips, not only waveforms.
- Normalize all clips to a consistent sample rate and mono format.

#### Experiment and Measure
- Compare 8 kHz, 16 kHz, 24 kHz, and 48 kHz versions by listening and plotting.
- Measure duration, RMS energy, and file size for each version.

#### Required Output Artifacts
- `notebooks/day01_waveform.ipynb`
- `data/clean_manifest.csv` (now includes a `transcript` column)
- `data/benchmark/` (reference-transcripted corpus slice)
- `docs/audio_baseline_notes.md`

#### Completion Check
> You can explain what is lost when sample rate is reduced and can reproduce the
> same preprocessing from code.

---

### DAY 02: Fourier intuition and STFT
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_02.md`](days/day_02.md)


#### Learn
- Frequency, phase, harmonics, and spectral energy.
- Fourier transform intuition without memorizing derivations.
- STFT frames, window length, hop length, overlap.
- Tradeoff between time resolution and frequency resolution.

#### Build in MendSpeech
- Implement STFT visualization with PyTorch or TorchAudio.
- Plot the same utterance with several window and hop settings.

#### Experiment and Measure
- Hold audio constant and change one STFT setting at a time.
- Write what phonetic or transient detail becomes easier or harder to see.

#### Required Output Artifacts
- `notebooks/day02_stft.ipynb`
- `results/day02_stft_parameter_grid.png`

#### Completion Check
> You can choose a reasonable frame and hop configuration and explain why.

---

### DAY 03: Mel scale and log Mel features
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_03.md`](days/day_03.md)


#### Learn
- Human frequency perception and the Mel scale.
- Mel filterbanks and log compression.
- Number of Mel bins and dynamic range.
- Normalization of acoustic features.

#### Build in MendSpeech
- Implement or inspect a log Mel feature pipeline.
- Build a function that returns features plus metadata needed for reproducibility.

#### Experiment and Measure
- Change Mel bin count and compare visual structure and compute size.
- Verify consistent feature shapes for different utterance lengths.

#### Required Output Artifacts
- `src/audio/features.py`
- `notebooks/day03_logmel.ipynb`
- `tests/test_features.py`

#### Completion Check
> You can trace waveform to STFT to Mel filterbank to log Mel tensor.

---

### DAY 04: Build SpeechDamageBench as a standalone package
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_04.md`](days/day_04.md)


#### Learn
- Deterministic corruption design and seed control.
- Additive noise, clipping, bandwidth limitation, dropouts, and reverberation.
- Why a benchmark should be reusable outside the main application.
- Versioned severity presets and manifest metadata.

#### Build in MendSpeech
- Create SpeechDamageBench as a **nested standalone package**. Do **not**
  overwrite the repo-root `pyproject.toml` (that file belongs to `mendspeech`).
- Implement noise, clipping, bandwidth reduction, dropout, and simple reverberation modules.
- Add a seed controlled configuration object and a small command line entry point.
- Record corruption name, severity, seed, parameters, and clean source id for every output.
- Presets live in `speechdamagebench/speechdamagebench/presets.py`. If you
  also want YAML, keep it inside the package (`presets/damage_levels.yaml`)
  so there is one source of truth.

#### Experiment and Measure
- Generate mild, medium, and severe examples from the same clean sentence.
- Reproduce the exact same damaged waveform from the same seed.
- Change only the seed and verify that the corruption changes while all configured parameters remain fixed.

#### Required Output Artifacts
- `speechdamagebench/pyproject.toml` (package name `speechdamagebench`; do not touch repo-root `pyproject.toml`)
- `speechdamagebench/speechdamagebench/audio_damage.py`
- `speechdamagebench/speechdamagebench/presets.py`
- `speechdamagebench/speechdamagebench/cli.py`
- `speechdamagebench/tests/test_determinism.py`
- `data/benchmark/` begun if still missing (manifest with relative paths + transcripts)

#### Completion Check
> Another project can install SpeechDamageBench and regenerate the same damaged
> clip from a manifest entry.

---

### DAY 05: Objective audio measurements
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_05.md`](days/day_05.md)


#### Learn
- RMS and peak level.
- Simple SNR calculation when the clean reference is known.
- Spectral distance intuition.
- Why perceptual speech quality is not fully captured by one scalar metric.

#### Build in MendSpeech
- Add baseline metrics for clean versus corrupted pairs.
- Store results in a tidy CSV schema with clip id, corruption, severity, seed, and measurements.

#### Experiment and Measure
- Run all corruption levels on at least ten **labeled** clips from
  `data/benchmark/` (or `data/benchmark/` plus `data/clean_manifest.csv` if
  those rows have transcripts). If fewer than ten labeled clips exist, finish
  the corpus first — do not invent metrics on five unlabeled files.
- Look for cases where a metric disagrees with your listening judgment.

#### Required Output Artifacts
- `src/metrics/audio_metrics.py`
- `results/week1_damage_metrics.csv`
- `docs/metric_limitations.md`

#### Completion Check
> You can explain what each metric says and what it fails to say.

---

### DAY 06: Build the first MendSpeech audio console
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_06.md`](days/day_06.md)


#### Learn
- Audio playback in a lightweight interface.
- Waveform and spectrogram synchronization.
- Before and after comparison design.

#### Build in MendSpeech
- Build a local page or notebook dashboard with clean and damaged playback.
- Add corruption controls and immediately regenerate the damaged clip.
- Display waveform, spectrogram, and basic measurements.

#### Experiment and Measure
- Test with three speakers and several corruption types.
- Write down usability problems that would block later live debugging.

#### Required Output Artifacts
- `app/audio_lab.py`
- `results/week1_audio_console.png`

#### Completion Check
> Another person can open the tool, damage an utterance, and understand the visual
> change without reading your code.

---

### DAY 07: Review, explain, and freeze Week 1
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_07.md`](days/day_07.md)


#### Learn
- Review waveform, STFT, Mel features, SNR, clipping, dropouts, and reverberation.

#### Build in MendSpeech
- Clean repository structure.
- **Freeze only.** Do not start collecting a new corpus on this day. The
  labeled set must already exist from Days 04–05. Freeze SpeechDamageBench
  v0.1 severity presets and a benchmark set of **≥30 utterances across ≥5
  speakers with reference transcripts** (lab scale is typically ~5 speakers;
  ≥5 is the floor, not a race to dozens), written as speaker-separated
  train/val/test splits in one manifest.
- Tag the benchmark package schema and add a minimal usage example independent of MendSpeech.

#### Experiment and Measure
- From a blank notebook, recreate one corruption and one log Mel plot without copying previous cells.

#### Required Output Artifacts
- `reports/week1_audio_foundations.md`
- `data/benchmark_manifest.csv` (relative paths, `transcript`, `speaker_id`, and a `split` column: train/val/test)
- `speechdamagebench/README.md`
- `speechdamagebench/VERSION`

#### Completion Check
> You can teach the complete path from clean waveform to controlled corruption and
> feature tensor.

---

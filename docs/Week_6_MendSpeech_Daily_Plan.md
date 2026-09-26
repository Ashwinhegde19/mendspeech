# Week 6: Robustness, Fine Tuning, RNNT, and Calibration

> **Days 36 to 42**  
> **Navigation:** [← Week 5](Week_5_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Week 7 →](Week_7_MendSpeech_Daily_Plan.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Adapt the recognizer to damaged speech while learning training and calibration discipline.
>
> **v2 evidence gate:** Days 36–42 remain CORE. Gate 5 requires one ASR
> adaptation experiment with clean regression, validation-based calibration and
> the robustness report in the one app. Day 40 retains the quantization lab with
> compatibility/parity checks: measured supported precision or explicit blocked
> optimization status, never an invented speedup. No calendar deadline applies.

---

## Week Map

| Day | Focus | Minimum Evidence / Artifact | Compute | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 36** | Training pipeline anatomy | You can diagnose whether a run is learning, diverging, or overfitting from basic
evidence. | `Modal L4` | [Open Day 36](days/day_36.md) |
| **Day 37** | Build a robust fine tuning dataset | Source/speaker leakage audit, provenance and immutable core evaluation; optional separate verified extension. | `Local CPU` | [Open Day 37](days/day_37.md) |
| **Day 38** | Fine tune for damaged speech robustness | One reproducible base/adapted experiment with clean regression and validation-only selection. | `Modal L4` | [Open Day 38](days/day_38.md) |
| **Day 39** | SpecAugment and augmentation ablation | You can separate the effect of augmentation from the effect of extra training time. | `Modal L4` | [Open Day 39](days/day_39.md) |
| **Day 40** | RNN-T concepts and quantization lab | Export parity and supported same-L4 precision measurements, or explicit blockers; calibration never uses test. | `Modal L4` | [Open Day 40](days/day_40.md) |
| **Day 41** | Confidence calibration for repair decisions | Repair thresholds are now justified from held out evidence rather than guessed. | `Modal L4 for logits, local CPU for
analysis` | [Open Day 41](days/day_41.md) |
| **Day 42** | Week 6 robustness milestone | Single-app measured adaptation/calibration, clean controls and honest precision status. | `Modal L4` | [Open Day 42](days/day_42.md) |

**Compression map:** all seven sessions remain CORE. One adaptation recipe is
reused for the Day 39 augmentation control; optional compatible Indic data does
not create another training track. Add-on C uses its separate manifest prepared
after Gate 2 and completes later metrics/report at Gate 7. Blocked optimization
is not successful quantization, and no session is complete without its evidence.

---

## Reference Spine
- NVIDIA NeMo ASR training documentation\nRNNT primary references\nCalibration and reliability diagram references

---

## Daily Detailed Operating Plans

### DAY 36: Training pipeline anatomy
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_36.md`](days/day_36.md)

#### Learn
- Manifest format.
- Batching variable duration audio.
- Loss curves.
- Learning rate.
- Validation split.
- Checkpointing.

#### Build in MendSpeech
- Create a tiny reproducible training configuration.
- Run a short smoke training job and verify loss decreases.

#### Experiment and Measure
- Deliberately use a bad learning rate and record the failure signature.

#### Required Output
- `configs/train_smoke.yaml`
- `results/day36_training_smoke.csv`
- `docs/training_debug_notes.md`

#### Completion Check
> You can diagnose whether a run is learning, diverging, or overfitting from basic
evidence.

---

### DAY 37: Build a robust fine tuning dataset
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_37.md`](days/day_37.md)

> **v2 STATUS: CORE.** Prepare one ASR adaptation dataset. The frozen core
> evaluation set is immutable; any Indic extension stays separate.

#### Learn
- Train, validation, test separation.
- Speaker leakage.
- Synthetic corruption sampling.
- Balanced severity distribution.

#### Build in MendSpeech
- Prepare train/validation manifests pairing verified transcripts with clean
  and corrupted audio for one adaptation experiment. Record source/speaker IDs,
  language, license/consent, transcript verification and normalization, corruption,
  severity, seed, parameters and package version; retain clean examples.
- Preserve the already frozen test membership and speaker-separated splits.
  All clean/corrupted copies of a source stay in one split. A test manifest is
  a reference to the frozen evaluation set, never a new sample or rewritten set.
- Optionally reuse the separate `data/indic_codemix_manifest.csv` prepared after
  Gate 2 for the same adaptation experiment only if the checkpoint/tokenizer
  supports the language and a competent transcript verifier is available.
  Keep extension train/validation/test speakers disjoint from each other and
  the core evaluation speakers. Otherwise retain it as evaluation-only and
  record the limitation; do not add a second training track.

#### Experiment and Measure
- Audit source-level duplicates, speaker leakage and corruption provenance.
  Report split counts, clean/severity balance and immutable core test membership.
- Record extension language support/verification evidence and inclusion or
  evaluation-only status. Add-on C streaming metrics remain for later evaluation,
  not invented data-preparation results.

#### Required Output
- `data/train_manifest.jsonl`
- `data/val_manifest.jsonl`
- `data/test_manifest.jsonl` (frozen-set reference; preserve existing contents)
- `reports/data_audit.md`

#### Completion Check
> The audit demonstrates no source or speaker leakage into training/validation,
preserves the frozen core evaluation set and documents provenance. Any optional
extension is separately identified and does not create another training track.

---

### DAY 38: Fine tune for damaged speech robustness
- **Compute:** `Modal L4; use a manageable training configuration within budget`
- **Dedicated Daily File:** [`docs/days/day_38.md`](days/day_38.md)

> **v2 STATUS: CORE.** One bounded ASR adaptation experiment with a clean-speech
> regression control; negative results count as evidence, failed runs do not.

#### Learn
- Transfer learning.
- Frozen versus trainable layers.
- Mixed precision.
- Gradient accumulation.

#### Build in MendSpeech
- Adapt the existing compatible ASR checkpoint once using Day 37's audited data.
  Fix trainable layers, learning rate, steps, seed, clean/damaged sampling,
  precision, gradient accumulation and budget before running; choose the best
  checkpoint using validation only, never frozen test outcomes.
- If the optional Indic data passes Day 37's support and verification checks,
  include it in this same experiment and report its slice separately. Otherwise
  retain evaluation-only coverage; do not install/train a second recognizer.
- Save the reproducible configuration and provenance: base code/weight revisions,
  manifest hashes, source permissions, software/hardware and selection rule.
  Keep checkpoints and raw training logs local/gitignored. Day 39's augmentation
  ablation reuses this recipe; it is not a separate adaptation track.

#### Experiment and Measure
- Compare base and adapted checkpoints on identical frozen clean/damaged cases
  with the same decoder and normalization. Report WER/CER by condition, clean
  regression, confidence shifts and failures, not only aggregate improvement.
- Report any extension results separately from the frozen core benchmark. All
  comparable latency/RTF/peak-memory measurements use the same L4 configuration.
  Record actual cost and a failed run honestly instead of claiming adaptation.

#### Required Output
- `training/finetune.py`
- `configs/finetune.yaml`
- `checkpoints/week6_best/` (local/gitignored weights and raw logs)
- `results/day38_base_vs_adapted.csv`
- `reports/day38_adaptation.md` (provenance, selection rule, cost and limitations)

#### Completion Check
> The single base-versus-adapted experiment is reproducible and states what
improved, what did not and whether clean speech regressed. Test data never
selects the checkpoint, and optional language coverage is labeled separately.

---

### DAY 39: SpecAugment and augmentation ablation
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_39.md`](days/day_39.md)

#### Learn
- Time masking.
- Frequency masking.
- Data augmentation as invariance training.

#### Build in MendSpeech
- Add one augmentation intervention to a controlled short run.

#### Experiment and Measure
- Compare no augmentation versus selected augmentation with the same seed and training budget.

#### Required Output
- `experiments/specaugment_ablation.py`
- `results/day39_augmentation.csv`

#### Completion Check
> You can separate the effect of augmentation from the effect of extra training time.

---

### DAY 40: RNN-T concepts and quantization lab
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_40.md`](days/day_40.md)

> **v2 STATUS: CORE.** Retain the quantization lab behind a compatibility check.
> Unsupported precision is a documented blocker, not a completed optimization.

#### Learn
- Encoder.
- Prediction network.
- Joint network.
- Blank handling.
- Streaming emission behavior.
- Difference from CTC independence.
- Post-training quantization: dynamic vs static INT8, and why static needs a calibration set.
- What quantization can and cannot preserve in an ASR model (logit sharpness, confidence behavior).

#### Build in MendSpeech
- Smoke-check the selected ASR model's export and precision support on one
  compatible L4 backend. Pin model/backend revisions and verify supported
  operators, dynamic lengths, decoding and actual device/kernel placement.
  Do not add a provider sweep or silently compare CPU INT8 with GPU inference.
- Export the Day 38 checkpoint only through the supported path. Check the
  original model versus exported model at the same precision on validation
  clips before quantization: output/logit tolerances where exposed, decoded
  text, lengths and decoding settings. Record and resolve parity failures first.
- Compare FP16 and INT8 only where the same backend supports their intended
  execution on L4. For static INT8, select and record a representative calibration
  slice from validation, disjoint by source/speaker from test; never calibrate
  using the frozen test benchmark. State if the backend does not need calibration.
- If export, parity or precision support blocks measurement, retain working
  inference and record the exact failed check, error and unsupported precision.
  Do not change models/backends repeatedly to manufacture an INT8 result.

#### Experiment and Measure
- After parity passes, measure original/exported baseline and supported FP16/INT8
  variants on identical frozen cases, decoder, batch size, timing boundaries,
  warm-up and L4 hardware. Report actual WER/CER, latency/RTF and peak memory;
  INT8 may be slower or less accurate and need not be selected for deployment.
- Record logit/confidence shifts for Day 41 validation-based calibration. Report
  precision coverage and CPU fallbacks explicitly; a mixed-device run is not
  a controlled GPU speed comparison.
- Give each variant `measured` or `blocked` status with reasons and blank
  unavailable metrics. Gate 5 can carry an explicit blocked optimization status,
  but cannot claim successful quantization or a speedup without measurements.

#### Required Output
- `docs/rnnt_walkthrough.md` (theory summary from the Learn block)
- `docs/day40_quantization_notes.md` (compatibility, parity, calibration source,
  backend/precision settings and blockers)
- `results/day40_quantization_tradeoffs.csv` (actual metrics or blocked rows)

#### Completion Check
> You can explain RNN-T streaming behavior and demonstrate original/export
parity plus measured supported-precision tradeoffs on L4, or identify the exact
compatibility/parity blocker. A blocked branch stays explicitly unimplemented;
no unsupported INT8, speedup or calibration claim is presented as complete.

---

### DAY 41: Confidence calibration for repair decisions
- **Compute:** `Modal L4 for logits, local CPU for
analysis`
- **Dedicated Daily File:** [`docs/days/day_41.md`](days/day_41.md)

#### Learn
- Reliability diagrams.
- Expected calibration error intuition.
- Threshold selection from validation data.

#### Build in MendSpeech
- Build a simple calibration analysis for confidence versus correctness.
- Choose policy thresholds on validation, not test.

#### Experiment and Measure
- Compare raw and calibrated confidence if a simple method is feasible.

#### Required Output
- `src/asr/calibration.py`
- `results/day41_reliability.png`
- `configs/repair_modes_calibrated.yaml`

#### Completion Check
> Repair thresholds are now justified from held out evidence rather than guessed.

---

### DAY 42: Week 6 robustness milestone
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_42.md`](days/day_42.md)

> **v2 STATUS: CORE — Gate 5 is evidence-based.** Extend the one app; preserve
> explicit blocked precision status rather than claiming unmeasured optimization.

#### Learn
- Review fine tuning, augmentation, RNNT, and calibration.

#### Build in MendSpeech
- Extend `app/audio_lab.py` to switch between base and adapted recognizer;
  retain the existing ASR/streaming/policy controls rather than create another app.
- Show clean WER, damaged WER, confidence calibration, and repair percentage.
- Show model/policy versions, raw versus calibrated confidence, validation-selected
  thresholds, safe action/reason codes and Day 40 measured/blocked precision
  status. Do not offer an unavailable export as if it were implemented.

#### Experiment and Measure
- Run one fixed benchmark suite and freeze results for Week 8 comparisons.
- Keep matched clean/raw-damaged controls and separate optional Indic extension
  results from the immutable core test set. Report regression, negative outcomes,
  inspect/abstain behavior, adaptation provenance and actual L4 configurations.

#### Required Output
- `app/audio_lab.py`
- `results/week6_frozen_baseline.csv`
- `reports/week6_training.md`

#### Completion Check
> The one app and report demonstrate measured adaptation and calibration results,
including clean regression or a negative result. Precision tradeoffs are supported
by controlled L4 measurements or explicitly blocked, never falsely completed.

---

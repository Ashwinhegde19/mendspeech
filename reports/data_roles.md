# Data Roles — Frozen Speech Benchmark

> Day 10, concepts 5 and 6. Declared before any downstream tuning, per
> [`docs/days/day_10.md`](../docs/days/day_10.md). Generated facts are checked
> by `src/metrics/roles.py`; run `pytest tests/test_roles.py` to verify.

## Frozen artifact

| Field | Value |
| :--- | :--- |
| Canonical manifest | `data/benchmark_manifest.csv` |
| SHA-256 | `f61c1ad157c619df24bed39aedbff4e2cf545a30aabe3acb8fc9f68d76d77782` |
| Legacy duplicate | `data/benchmark/manifest.csv` — byte-identical, asserted by test |
| Source | `openslr/librispeech dev-clean`, extracted 2026-08-31, 16 kHz mono |
| Utterances | 30 |
| Speakers | 5 |
| Reference words | 772 |

The freeze is verifiable, not merely asserted: `file_checksum()` recomputes
the digest and a test fails if the legacy copy drifts from the canonical file.

## Declared roles

The manifest ships three splits. The spec requires **four** roles — training,
calibration, validation and final test — so calibration is carved out of
training by speaker. No role shares a voice with any other.

| Role | Manifest split | Speakers | Utterances | Source |
| :--- | :--- | :--- | :--- | :--- |
| train | `train` | `spk_1272`, `spk_1462` | 12 | train minus `spk_1673` |
| calibration | `train` | `spk_1673` | 6 | train, `spk_1673` only |
| validation | `val` | `spk_1919` | 6 | unchanged |
| final_test | `test` | `spk_1988` | 6 | sealed |

Totals reconcile: 12 + 6 + 6 + 6 = 30, and the five speakers are each in
exactly one role.

### Why calibration is carved from training

The manifest had **no calibration role at all**, and the quality contract
requires confidence to be "fitted on calibration data, checked with
reliability/ECE/Brier" before it is ever called calibrated. The two options
were:

- **Carve from training (chosen).** Keeps all four roles speaker-disjoint.
  Costs 6 of 18 training utterances — a third of the training data.
- **Let calibration ride on validation.** Preserves training data, but every
  threshold chosen on validation is then optimistic on validation, and
  validation is the split Day 11 uses to select models.

Carving is chosen because the spec requires the roles be *separate*, and
because re-splitting after Day 11 has begun would retroactively invalidate
whatever was tuned. The cost is declared rather than hidden.

**Consequence, stated plainly:** 12 training utterances across 2 speakers is
very little. If acoustic fine-tuning is attempted, this count is a likely
blocker, and per the spec an insufficient training or calibration set **blocks
training — it is not permission to reuse test data.**

## Leakage and grouping audit

`check_leakage()` reports, on the frozen manifest:

| Check | Result |
| :--- | :--- |
| Speaker shared between any two roles | **none** |
| Duplicate reference transcripts | **none** |
| Transcripts shared across roles | **none** |
| Overall verdict | `checks_passed = True` |

Tests inject a deliberate speaker overlap and a deliberate duplicate, and
assert the audit catches both, so a passing audit means something.

### Repeated corruption is not an independent sample

This is the grouping trap the spec names, and it is easy to get wrong:

- **30** independent utterances
- **5** independent speakers
- **180** scored rows for a full sweep (30 utterances × (1 clean + 5 corruptions))

The row count overstates the sample by **6×**. Corruption conditions applied
to one utterance are repeated measurements of the same utterance, not new
observations. Any significance claim must count **utterances and speakers**,
never rows. Per-utterance aggregation is mandatory, and per-speaker variance
matters more than per-row variance because speaker identity is the grouping
factor that actually generalizes.

## Rules for downstream sessions

1. Score against the manifest `transcript` column. Never use a model's own
   clean output as its reference — that measures drift, not accuracy.
2. Tune on **validation**. Report **final_test** once, at the end.
3. Fit confidence on **calibration** only. Validation numbers that were used
   to select anything are not calibration evidence.
4. New editor data goes in `data/editor_manifest.jsonl`, a separate manifest.
   The frozen benchmark is not modified or extended.
5. Report the independent sample size alongside any rate.

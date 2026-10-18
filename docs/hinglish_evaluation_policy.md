# Hinglish Evaluation Policy

Evaluation policy for the MUCS Hinglish (`dianavdavidson/MUCS-Hinglish`)
workstream. It was declared before the adaptation and translation-layer
experiments that will use the final held-out reserve, and it records the
speaker roles, the pre-registered reserve-selection rule, the immutability
rules, the sealed official test partition, and the known limitations. The
frozen development-cohort numbers live in
[results/hinglish/validation_baseline_metrics.json](../results/hinglish/validation_baseline_metrics.json).

## Speaker role inventory

The MUCS-Hinglish training split exposes 52,825 utterances across 520
speakers. Roles are assigned per speaker; no speaker appears in two roles.
The official test partition of 30 speakers and 3,136 utterances is sealed (see
below) and is not part of this inventory.

| Role | Speakers | State | Permitted use |
| :--- | ---: | :--- | :--- |
| Training pool | 443 | includes the diagnostic speakers | adaptation, diagnostics, and training |
| Validation pool | 77 | speaker-disjoint from training | model and configuration selection |
| — Development cohort | 30 | evaluated; results inspected | guides product choices, not a final unbiased estimate |
| — Final held-out reserve | 47 | untouched | final evaluation only, once, after all choices are frozen |

443 + 77 = 520 speakers, and the 30-speaker development cohort plus the
47-speaker reserve are exactly the 77-speaker validation pool. Five previously
analyzed diagnostic speakers are reserved inside the training pool and can
never enter validation. Each role was audited for speaker overlap; the audit
result was empty, and the reserve was additionally checked to be disjoint from
the development cohort and the training pool.

## Development cohort (already evaluated)

- Selection version: `mendspeech.mucs_validation.v1`, seed 42.
- 30 speakers: 28 Hindi-dominant and 2 mixed-code by speaker-average English
  ratio.
- One utterance per speaker: the eligible record with the smallest `clip_id`,
  where eligibility means valid language metadata under the rule below.
- Frozen manifest SHA-256:
  `2669cb56ddf2a9c12602b9fcbe406a38724a8e3d1d65be0eeda228ad78955d67`.
- Because these results have been inspected, they are development data: they
  may guide thresholds and product choices, but they cannot serve as an
  unbiased final measurement.

## Final held-out reserve (pre-registered rule)

Pool version `mendspeech.mucs_holdout_pool.v0` was frozen from the
metadata-only inventory before any adaptation experiment. The pre-registered
rule is:

1. **Candidate set.** The 77 validation-pool speakers minus the 30 development
   speakers, yielding 47 speakers: 44 Hindi-dominant and 3 mixed-code by
   speaker-average English ratio (v1 bucket boundaries: `>= 0.6` English is
   English-dominant, `<= 0.4` is Hindi-dominant, otherwise mixed-code).
2. **Metadata only.** Construction reads `segment_id`, `speaker_id`, language
   ratios, word counts, `segment_part`, and `shard`. Transcripts are never
   read, printed, or stored by the reserve tooling.
3. **Eligibility.** A record is eligible when its metadata is valid under the
   frozen v1 rule: ratios within `[0, 1]`, ratios summing to 1, and the
   declared English ratio consistent with the word counts within 0.05. This is
   the same policy class as the development cohort.
4. **One clip per speaker.** The eligible record with the lexicographically
   smallest `clip_id` is selected. Selection is deterministic and involves no
   randomness or seed.
5. **No eligible record.** If a candidate speaker had no eligible record, the
   speaker would be dropped and logged, because the v1 transcript-based
   fallback would require reading transcripts and is forbidden for the
   reserve. This tightening does not trigger for any of the 47 speakers; the
   smallest eligible-clip count is 35.
6. **Stratified reporting.** Metrics are reported for the reserve overall and
   per language bucket, with buckets computed per speaker (not per clip) from
   the average English ratio over eligible records. Per-clip buckets are
   recorded in the pool artifact for transparency but are not the reporting
   strata.
7. **Secondary robustness check (reserved, not primary).** A future check may
   use all eligible clips per speaker from the same frozen inventory,
   aggregated per speaker (speaker-macro average) under the same
   corruption/evaluation configurations. It must never replace the primary
   one-clip-per-speaker metric; the primary cohort stays the 47 selected
   clips.

The frozen artifact is a JSON file listing the 47 speaker IDs, the selected
clip ID per speaker, eligibility metadata, and the policy text, plus a sibling
SHA-256 checksum file. Both files are stored outside the repository in the
private artifact store. The pool SHA-256 is:

```text
5dc051dab2e55da6586614676d0b6bfbcb3e7a809b8f28d737e78c399d8b4ff5
```

`scripts/reserve_hinglish_holdout_pool.py` builds the pool. `--dry-run`
prints the composition summary without writing, and `--freeze` writes the pool
and checksum outside the repository, refusing to overwrite an existing pool
file. Re-freezing the same inventory and split is byte-identical; this was
verified during the freeze.

## Immutability rules

1. The reserve was frozen before adaptation experiments began and must not be
   modified afterwards. Its checksum is the reference for every comparison
   that uses it.
2. New experiments change the corruption or evaluation configuration, never
   the evaluation set. A new evaluation set is a new experiment, not a repair
   of this one.
3. Any post-freeze exclusion or correction (for example, an audio extraction
   failure) is reported explicitly as a documented deviation with its counts.
   If it would change the speaker set, a new pool version is created and both
   versions are reported; the frozen version stays authoritative for results
   already published against it.
4. The reserve is evaluated after all model, decoding, and threshold choices
   are frozen. Its numbers are reported once, as final measurements, and are
   added to the [results index](../results/README.md).

## Sealed official test partition

The official MUCS-Hinglish test partition (30 speakers, 3,136 utterances) is
sealed. It is never downloaded, streamed, queried, or used for diagnostics,
threshold selection, or any intermediate measurement. It is reserved for final
reporting exactly as described in the
[Hinglish baseline plan](hinglish_baseline_plan.md).

## Documented limitations

**(a) Source-recording overlap cannot be verified.** The dataset metadata
exposes `segment_part` and `speaker_id` but no source-recording identifier.
Speaker-level separation was audited and holds, but two clips attributed to
different speakers could still originate from the same source recording, and
that overlap cannot be detected or excluded from the available fields. Both
the development cohort and the reserve inherit this limitation.

**(b) Possible Whisper pretraining exposure cannot be ruled out.** The
baseline uses `openai/whisper-small`, whose pretraining corpora are
incompletely documented. Exposure to read-speech corpora with read-speech
acoustic characteristics, possibly including data that overlaps the MUCS
domain, cannot be excluded, which would make clean-audio accuracy optimistic.
This exposure remains incompletely verified; reported numbers are
comparative measurements under this caveat, not unbiased estimates of
generalization to unseen speakers and recordings.

## Reproducibility anchors

| Artifact | Identifier |
| :--- | :--- |
| Training metadata inventory | SHA-256 `75aa1d591318182e8be6c22c8ecaa0ca270a7a6b1ec3975413f96c11b0223e04` |
| Speaker split | seed 42, SHA-256 `d7aff52df53bb7fbe6633dcefd00e0a89c4268905893bf22a8649e41088c891f` |
| Development cohort manifest | SHA-256 `2669cb56ddf2a9c12602b9fcbe406a38724a8e3d1d65be0eeda228ad78955d67` |
| Final held-out reserve | `mendspeech.mucs_holdout_pool.v0`, SHA-256 `5dc051dab2e55da6586614676d0b6bfbcb3e7a809b8f28d737e78c399d8b4ff5` |

# Evaluation Protocol

> Day 10, concept 7. Declared **before** downstream tuning, per
> [`docs/days/day_10.md`](../docs/days/day_10.md). Data roles are in
> [`reports/data_roles.md`](../reports/data_roles.md). Changing anything here
> after a model has been selected invalidates the comparison and requires a
> scope review.

## 1. Primary diagnostic metrics

Reported per condition, always with the independent sample size alongside.

| Metric | Unit | Why it is primary |
| :--- | :--- | :--- |
| **WER** | word | The dictation unit. A wrong word is a wrong word. |
| **CER** | character | Localizes damage inside a word; lower than WER for word swaps. |
| **S / D / I counts** | word | Separates a missed word from a wrong one. |
| **Slice rates** | word, per slice | Names, numbers, negation are protected content. |

Normalization is fixed at `mendspeech.v1`
(`src/metrics/normalization.py`) and is recorded in every result file. Scoring
code never changes after a number is published.

### Metrics that are *not* primary, and why

- **Raw softmax confidence** is never called calibrated. A score in [0,1] is
  not a probability until it has been fitted and checked.
- **Mean confidence over a whole utterance** hides errors, because
  blank-heavy frames dominate the average.
- **Any aggregate that hides a slice rate.** An overall WER of 3% with a 40%
  negation error rate is a failure, not a good score.

## 2. What each number must state

Every reported figure carries:

- corruption name, severity, and **seed**
- source clip ID and speaker ID
- the role it was scored under (train / calibration / validation / final test)
- **independent sample size** — utterances and speakers, not rows
- normalization version
- model, decoder, and precision used to produce the hypothesis

A number missing any of these is not a result and must not be committed as
one.

## 3. Data usage rules

| Rule | Statement |
| :--- | :--- |
| Tuning | Validation only. |
| Confidence fitting | Calibration only. |
| Final test | Reported once, in the frozen release phase, then sealed. |
| Test reuse | An insufficient train or calibration set **blocks training**. It is never permission to reuse test. |
| Editor data | Separate manifest (`data/editor_manifest.jsonl`). The frozen benchmark is not extended. |
| Grouping | Count utterances and speakers. Corruption rows are repeated measures, not new samples. |

## 4. Failure thresholds

Stated now so a later session cannot quietly relax them.

| Condition | Threshold | Action |
| :--- | :--- | :--- |
| Protected-content (name/number/negation) slice rate | > 2 percentage points above the reference baseline | Stop, diagnose. Direct from the editor contract. |
| Slices with fewer than 20 reference words | any rate reported | Report the count; treat the rate as diagnostic only. Not a stop condition. |
| Overall WER regression vs clean | > 20 percentage points on a single corruption | Investigate before reporting a conclusion. |
| Empty slice (`covered = False`) | rate is `null` | Report as *not measured*. **Never** substitute 0.0. |
| Confidence claimed calibrated | no calibration fit performed | Not permitted. Selecting a threshold is not calibration. |
| Any result with 0 independent speakers | any | Not a measurement. |

## 5. Known limitations of this protocol

Declared up front rather than discovered later:

1. **The corpus is small.** 30 utterances, 5 speakers, 772 reference words.
   Slice denominators are smaller still: 9 number, 12 negation, 11 name
   reference words. A single utterance moves a slice rate by 8–11 percentage
   points. **No slice rate on this corpus is a stable statistic**, and none
   should be presented as one.
2. **One speaker per non-training role.** Validation and final test each have
   exactly one speaker, so between-speaker variance cannot be estimated
   within either split.
3. **British read speech only.** `dev-clean` is a single narrow domain. The
   results say nothing about noisy or far-field audio.
4. **The S/D/I split is alignment-dependent.** When several alignments tie
   for the optimal distance, how the distance divides across operations
   depends on backtrace order. The **total** is reliable; the **distribution
   is one valid reading**, not a unique fact. In particular, do not treat the
   insertion count as a standalone hallucination signal.
5. **Normalization is a stated convention, not a truth.** It deletes
   apostrophes rather than splitting them, and does not expand contractions
   or rewrite numbers. Results are only comparable under the same version.
6. **No significance testing is claimed.** With 5 speakers, any p-value would
   be a decoration. Report counts and rates.

## 6. Reproducibility

Every result file is regenerable from a committed script with a fixed seed.
The benchmark manifest checksum is recorded in
[`reports/data_roles.md`](../reports/data_roles.md); if it changes, every
number produced against it is void.

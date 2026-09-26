# Week 2: ASR, CTC, Confidence, and Repair Localization

> **Days 08 to 14**  
> **Navigation:** [← Week 1](Week_1_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Week 3 →](Week_3_MendSpeech_Daily_Plan.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Build the recognition and uncertainty layer, plus a reusable Modal execution path.
>
> **v2 evidence gate:** Days 08–14 remain CORE. Gate 2 requires ASR, WER/CER,
> confidence/timing, safe policy decisions and reproducible model/run metadata.
> Record the bounded external-comparator status; a documented deferral is not
> inpainting success. Then complete required Add-on A baseline/reference VAD
> measurements. No calendar deadline or timed rebuild determines completion.

---

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 08** | Frame sequence to transcript | `Modal L4 optional, CPU acceptable for
small runs` | DONE | [Open Day 08](days/day_08.md) |
| **Day 09** | CTC from first principles | `Local CPU` | DONE | [Open Day 09](days/day_09.md) |
| **Day 10** | WER, CER, and error taxonomy | `Local CPU` | CORE | [Open Day 10](days/day_10.md) |
| **Day 11** | Token confidence and where it fails | `Local CPU` | CORE | [Open Day 11](days/day_11.md) |
| **Day 12** | Time alignment and word timestamps | `Modal L4` | CORE | [Open Day 12](days/day_12.md) |
| **Day 13** | Confidence thresholds and error triage policy | `Local CPU` | CORE | [Open Day 13](days/day_13.md) |
| **Day 14** | Decoding comparison: greedy, beam, and beam plus LM | `Modal L4` | CORE | [Open Day 14](days/day_14.md) |

---

## Phase Focus

Recognition quality, confidence, calibration, and decoding

---

## Daily Detailed Operating Plans
### DAY 08: Frame sequence to transcript
- **Compute:** `Modal L4 optional, CPU acceptable for
small runs`
- **Dedicated Daily File:** [`docs/days/day_08.md`](days/day_08.md)
#### Learn
- Why acoustic frames outnumber output tokens.
- Encoder outputs, vocabulary logits, and decoding.
- CTC versus transducer versus attention decoder at a high level.
#### Build in MendSpeech
- Run a pretrained ASR model on clean and damaged SpeechDamageBench clips.
- Store transcript, token outputs if available, and timing metadata.
- Add a reusable Modal entry point so the same command can run ASR experiments on an L4 without editing deployment code each day.
  VoiceFixer's documented interface is not evidence of mask-aware inpainting;
  label only verified capabilities. Use one setup session plus at most one
  focused compatibility retry, then stop. No model search or scratch fallback.
- Record license/permitted use, code/package/checkpoint revisions, invocation,
  supported conditions, native input/output format, mask support, and whether
  processing changes audio outside a target interval. Verify sample-rate and
  length conversion explicitly; unknown behavior remains unverified.
- Attempt one clean and one damaged smoke case. Record final feasibility
  `feasible` or `deferred`, attempt outcomes and blockers in the notes. If
  feasible, the only planned adapter is `src/baselines/direct_audio_restore.py`;
  do not create a second mask-specific adapter. Day 50/54 consume this record.
#### Experiment and Measure
- Compare clean and corrupted transcripts on the exact same utterances.
- Keep comparator smoke evidence separate from ASR results. Record source IDs,
  corruption parameters/seed, repeatability and any unsupported condition; leave
  unavailable metrics blank with a reason. Comparable GPU timing/memory uses L4.
#### Required Output
- `src/asr/baseline.py`
- `infra/modal_asr.py`
- `results/day08_baseline_transcripts.csv`
  setup/retry evidence, `feasible` or `deferred`; no overwritten historical results)
#### Completion Check
> You can draw the path from features to encoder states to token probabilities to text,
and launch the same baseline locally or on Modal with a documented command.
The bounded comparator check has an evidence-backed status; a documented deferral
is sufficient for this external branch, but is not successful inpainting.

---

### DAY 09: CTC from first principles
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_09.md`](days/day_09.md)
#### Learn
- CTC blank symbol.
- Repeated labels and collapse operation.
- Why many frame paths map to one transcript.
- Conditional independence assumption and its consequence.
#### Build in MendSpeech
- Implement CTC collapse yourself without a library decoder.
- Create hand written alignment examples and unit tests.
#### Experiment and Measure
- Enumerate several legal paths for a tiny target word.
- Break your decoder deliberately with repeated letters and fix it.
#### Required Output
- `src/asr/ctc_decode.py`
- `tests/test_ctc_decode.py`
- `docs/ctc_explained.md`
#### Completion Check
> You can explain why a blank is needed and correctly decode repeated characters.

---

### DAY 10: WER, CER, and error taxonomy
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_10.md`](days/day_10.md)

> **v3 STATUS: CORE — recognition quality.** This is the accuracy axis every later trade-off is measured against.
#### Learn
- Word error rate: substitutions, deletions, insertions.
- Character error rate and when it helps.
- Why WER alone hides error severity.
- Names and numbers as a separate error class.
#### Build in MendSpeech
- Implement or verify WER and CER calculations in `src/metrics/wer.py`.
- Add an error analyzer labelling substitution, deletion, and insertion spans in `src/metrics/wer.py`.
- Add a names-and-numbers extractor so entity errors are counted separately in `src/metrics/wer.py`.
#### Experiment and Measure
- Score clean audio versus every SpeechDamageBench severity.
- Find which corruption type drives deletion errors fastest.
- Write tests for empty references, identical strings, and empty hypotheses in `tests/test_wer.py`.
#### Required Output
['- `src/metrics/wer.py`', '- `tests/test_wer.py`', '- `results/day10_wer_by_damage.csv`', '- `results/day10_error_types.csv`']
#### Completion Check
> You can compute WER by hand for a short example and explain each error class, and entity errors are reported separately from the blended rate.

---

### DAY 11: Token confidence and where it fails
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_11.md`](days/day_11.md)

> **v3 STATUS: CORE — confidence is a signal, not a truth.** This session exists to find the cases where confidence is confidently wrong.
#### Learn
- Frame softmax probability versus token confidence.
- Why mean confidence hides per-token failures.
- Confident-but-wrong: the failure mode that breaks a confidence-gated system.
#### Build in MendSpeech
- Extract per-token confidence and align it to emitted tokens in `src/asr/confidence.py`.
- Build a word-level confidence timeline aligned to the transcript in `src/asr/confidence.py`.
#### Experiment and Measure
- Compare confidence across clean, noisy, clipped, and dropout audio.
- Collect at least ten confident-but-wrong examples and write them up in `docs/day11_confident_wrong.md`.
- Record the rate at which a fixed confidence threshold would have accepted a wrong token.
#### Required Output
['- `src/asr/confidence.py`', '- `tests/test_confidence.py`', '- `results/day11_confidence_by_damage.csv`', '- `docs/day11_confident_wrong.md`']
#### Completion Check
> You can state when low confidence is informative, and you have documented concrete cases where high confidence was wrong.

---

### DAY 12: Time alignment and word timestamps
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_12.md`](days/day_12.md)

> **v3 STATUS: CORE — timestamps for latency attribution and interface feedback.** Word timing is what lets a streaming UI and a latency report say where time went.
#### Learn
- Frame index to wall-clock mapping.
- Token timestamps versus forced alignment.
- Why timestamps must be validated before they are trusted downstream.
#### Build in MendSpeech
- Map emitted tokens to audio time spans in `src/asr/timestamps.py`.
- Validate timestamps against a synthetic event at a known offset.
#### Experiment and Measure
- Inject dropouts at known offsets and check that surrounding token boundaries stay stable.
- Report timestamp error in milliseconds per corruption type in `results/day12_timestamp_error.csv`.
- Confirm that a decoding change does not silently shift timestamps.
#### Required Output
['- `src/asr/timestamps.py`', '- `tests/test_timestamps.py`', '- `results/day12_timestamp_error.csv`']
#### Completion Check
> Token timestamps are accurate to a stated millisecond tolerance and survive a decoding change, or the failure is documented.

---

### DAY 13: Confidence thresholds and error triage policy
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_13.md`](days/day_13.md)

> **v3 STATUS: CORE — triage, not repair.** Replaces the old repair-policy session. A downstream consumer needs to know accept, low-confidence, or reject; it does not need a synthesis policy.
#### Learn
- Choosing a threshold from validation data rather than test data.
- Risk-coverage: what fraction of traffic a threshold accepts and at what error rate.
- Why a single threshold is a policy decision, not a modelling result.
#### Build in MendSpeech
- Implement three triage policies — accept, low-confidence, reject — in `src/controller/triage.py`.
- Each policy maps confidence to an action with a reason code in `src/controller/triage.py`.
#### Experiment and Measure
- Sweep thresholds on the validation split and plot risk against coverage.
- Report the accepted fraction and the error rate inside the accepted set per corruption type.
- Freeze the chosen thresholds in `configs/triage_thresholds.yaml` using validation only.
#### Required Output
['- `src/controller/triage.py`', '- `tests/test_triage.py`', '- `results/day13_risk_coverage.csv`', '- `configs/triage_thresholds.yaml`']
#### Completion Check
> Thresholds are chosen from held-out validation evidence and you can state the error rate you accept in exchange for the coverage you keep.

---

### DAY 14: Decoding comparison: greedy, beam, and beam plus LM
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_14.md`](days/day_14.md)

> **v3 STATUS: CORE** Week 2 integration and the accuracy-versus-latency trade-off. A lower WER is not automatically better: a fluent but acoustically wrong transcript is the failure this project must catch.
#### Learn
- Greedy versus beam search: accuracy gained against search cost.
- External n-gram language models: why a plausible transcript can be acoustically wrong.
- Cache reuse for isolating decoder cost from acoustic cost.
#### Build in MendSpeech
- Extend the baseline runner to support greedy, beam, and beam plus one small n-gram LM in `src/asr/decoding.py`.
- Record LM text provenance, normalization, and split roles in `data/lm_text_manifest.csv`; exclude evaluation references and duplicates.
- Freeze one LM order and a small validation-only beam/LM-weight candidate list in `configs/decoding.yaml`.
#### Experiment and Measure
- Report WER/CER and names-and-numbers error for all three decoders on identical held-out cases.
- Collect cases where the LM helped and cases where it hurt; a lower WER does not prove safety.
- Measure decoder-only time on cached acoustic outputs separately from fresh audio-to-transcript latency.
#### Required Output
['- `src/asr/decoding.py`', '- `tests/test_asr_decoding.py`', '- `configs/decoding.yaml`', '- `data/lm_text_manifest.csv`', '- `results/day14_decoding_comparison.csv`', '- `docs/day14_harmful_lm_changes.md`', '- `app/audio_lab.py`']
#### Completion Check
> All three decoders are measured on the same held-out cases with separated decoder and end-to-end timing, and the LM's harmful changes are documented as first-class evidence.

---

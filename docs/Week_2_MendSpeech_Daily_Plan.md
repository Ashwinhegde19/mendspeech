# Week 2

> **Days 08–14**
> **Navigation:** [← Index](INDEX.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Executive Plan](REVISED_EXECUTION_PLAN.md)

---

> [!IMPORTANT]
> **Week theme:** Data protocol, confidence, editor contract, and the early end-to-end baseline
> Establish scoring, data roles, the conservative editor contract, and a measured end-to-end baseline before any optimization.

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 08** | Frame sequence to transcript | `Modal L4 optional, CPU acceptable for
small runs` | CORE | [Open Day 08](days/day_08.md) |
| **Day 09** | CTC from first principles | `Local CPU` | CORE | [Open Day 09](days/day_09.md) |
| **Day 10** | WER/CER, data roles, and evaluation protocol | `Local CPU` | CORE | [Open Day 10](days/day_10.md) |
| **Day 11** | Confidence definitions and failure cases | `Local CPU` | CORE | [Open Day 11](days/day_11.md) |
| **Day 12** | Transcript timestamps and correlated event tracing | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 12](days/day_12.md) |
| **Day 13** | Conservative editor contract and evaluation data | `Local CPU` | CORE | [Open Day 13](days/day_13.md) |
| **Day 14** | Greedy, beam and one small LM comparison | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 14](days/day_14.md) |

---

## Daily Detailed Operating Plans

### DAY 08: Frame sequence to transcript
- **Compute:** Modal L4 optional, CPU acceptable for
small runs
- **Dedicated Daily File:** [`docs/days/day_08.md`](days/day_08.md)

> **STATUS: CORE**

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

#### Required Output Artifacts
- `src/asr/baseline.py`
- `infra/modal_asr.py`
- `results/day08_baseline_transcripts.csv`
  setup/retry evidence, `feasible` or `deferred`; no overwritten historical results)

#### Completion Check
> You can draw the path from features to encoder states to token probabilities to text,
> and launch the same baseline locally or on Modal with a documented command.
> The bounded comparator check has an evidence-backed status; a documented deferral
> is sufficient for this external branch, but is not successful inpainting.

---

### DAY 09: CTC from first principles
- **Compute:** Local CPU
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

#### Required Output Artifacts
- `src/asr/ctc_decode.py`
- `tests/test_ctc_decode.py`
- `docs/ctc_explained.md`

#### Completion Check
> You can explain why a blank is needed and correctly decode repeated characters.

---

### DAY 10: WER/CER, data roles, and evaluation protocol
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_10.md`](days/day_10.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 09](days/day_09.md)
> **Effort:** 2–3 focused hours.

#### Learn
- Substitution/deletion/insertion counts, normalization and empty-reference conventions.
- Group-level leakage, repeated corruptions versus independent samples.

#### Build in MendSpeech
- Implement/test scoring and named-entity/number/negation slices without adding a model-based judge.
- Audit frozen benchmark roles and source/speaker separation. Record immutable checksums and separate training, calibration, validation and final-test roles before downstream tuning.
- Declare diagnostic primary metrics and failure thresholds in experiments/protocol.md; new editor data is a separate manifest, not a changed core set.

#### Experiment and Measure
- Hand-check small examples and empty/identical inputs; run validation diagnostics by damage/severity.
- Report actual corpus counts/roles; insufficient training or calibration data blocks training, not permission to reuse test.

#### Required Output Artifacts
- `src/metrics/wer.py`
- `tests/test_wer.py`
- `results/day10_wer_by_damage.csv`
- `results/day10_error_types.csv`
- `reports/data_roles.md`
- `experiments/protocol.md`

#### Completion Check
> Scoring conventions and frozen data roles are explicit and tested; development uses validation, not repeated test selection.

---

### DAY 11: Confidence definitions and failure cases
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_11.md`](days/day_11.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 10](days/day_10.md)
> **Effort:** 2–3 focused hours.

#### Learn
- Acoustic frame score, emitted-token confidence and word correctness are different quantities.
- Blank-heavy averages can hide errors; confidence is not calibrated merely because it lies in [0,1].

#### Build in MendSpeech
- Build token/word score records with model/head/tokenizer provenance; inspect the current baseline averaging semantics rather than trusting its docstring.
- Define label alignment and valid/missing states, with empty/blank/repeat tests.

#### Experiment and Measure
- On validation clips compare score versus correctness by corruption. Collect confident errors when present; report the observed count, never invent a required ten.
- Keep raw softmax and later calibrated probability distinct.

#### Required Output Artifacts
- `src/asr/confidence.py`
- `tests/test_confidence.py`
- `results/day11_confidence_by_damage.csv`
- `docs/day11_confident_wrong.md`

#### Completion Check
> Every score has a tested definition and correctness target; no raw confidence is called calibrated.

---

### DAY 12: Transcript timestamps and correlated event tracing
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_12.md`](days/day_12.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 10](days/day_10.md), [Day 11](days/day_11.md)
> **Effort:** 2–4 focused hours.

#### Learn
- Audio frame offsets versus wall-clock events; forced-alignment limitations.
- Monotonic clocks, request/stream IDs and event ordering.

#### Build in MendSpeech
- Implement transcript timestamp records plus a shared trace schema; separate word alignment from service-clock events.
- Test synthetic frame-index mapping, real short manually aligned speech, empty output and resampling; a click/tone is not a word-timestamp ground truth.

#### Experiment and Measure
- Report ms error on annotated validation speech and corruption cases with uncertainty; do not demand unchanged word timing when the recognized words differ.
- Round-trip trace IDs and start/end ordering for a replayed request.

#### Required Output Artifacts
- `src/asr/timestamps.py`
- `tests/test_timestamps.py`
- `src/bench/tracing.py`
- `tests/test_tracing.py`
- `results/day12_timestamp_error.csv`

#### Completion Check
> Word timing is validated against relevant annotations and trace clocks are explicit; the two are not conflated.

---

### DAY 13: Conservative editor contract and evaluation data
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_13.md`](days/day_13.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 10](days/day_10.md), [Day 11](days/day_11.md), [Day 12](days/day_12.md)
> **Effort:** 3–5 focused hours.

#### Learn
- Formatting versus rewriting and protected content.
- Training/validation/test group splits and proposed versus delivered output.

#### Build in MendSpeech
- Implement/test identity and deterministic whitespace/casing/punctuation baseline under docs/EDITOR_AND_RL_CONTRACT.md. Define protected spans and fail-closed output guard.
- Create data/editor_manifest.jsonl with licensed/consented source groups, annotated allowed outputs and sealed held-out roles. Audit duplicates and instruction-like input.
- Declare pilot/full-set counts and human review protocol; exact wording/negation/numbers cannot be silently changed.

#### Experiment and Measure
- Score identity and deterministic formatting on validation, including already-correct and needs-edit slices.
- Exercise deleted negation, altered entities/numbers, blank/long output, prompt injection and punctuation-induced meaning changes; log manual reviewer limits.

#### Required Output Artifacts
- `src/llm/contracts.py`
- `src/llm/deterministic.py`
- `tests/test_editor_contract.py`
- `data/editor_manifest.jsonl`
- `reports/editor_data_audit.md`
- `results/day13_editor_baselines.csv`

#### Completion Check
> Editing rules and independent quality metrics exist before LLM training; the small pilot and full evaluation data are separated and incomplete annotation is reported.

---

### DAY 14: Greedy, beam and one small LM comparison
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_14.md`](days/day_14.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 10](days/day_10.md), [Day 11](days/day_11.md), [Day 12](days/day_12.md)
> **Effort:** 3–5 focused hours.

#### Learn
- CTC collapse versus transducer search and external LM fusion.
- Decoder-only cached cost versus fresh end-to-end latency.

#### Build in MendSpeech
- Verify the current acoustic checkpoint/head and one supported decoder backend before comparing greedy, beam-only and beam+one n-gram LM. No second acoustic model or custom search engine.
- Record LM corpus license/hash/deduplication and exclude evaluation references; select a small predeclared beam/LM-weight list on validation only.
- Test token/blank/repeat mapping and alignment; initialize the shared repeatable benchmark harness with trace IDs, warm/cold and CPU worker metadata.

#### Experiment and Measure
- On identical validation cases report WER/CER, names/numbers, helpful and harmful changes and cached decoder versus fresh timing.
- Freeze selection for the later final test. Unsupported backend leaves decoding incomplete; independent baseline work may proceed with that blocker.

#### Required Output Artifacts
- `src/asr/decoding.py`
- `tests/test_asr_decoding.py`
- `configs/decoding.yaml`
- `data/lm_text_manifest.csv`
- `src/bench/benchmark_asr.py`
- `src/bench/environment.py`
- `tests/test_benchmark_asr.py`
- `results/day14_decoding_comparison.csv`
- `docs/day14_harmful_lm_changes.md`

#### Completion Check
> Three decoder conditions have controlled evidence and provenance; offline decoding is not claimed as live streaming and LM scores are not calibrated confidence.

---

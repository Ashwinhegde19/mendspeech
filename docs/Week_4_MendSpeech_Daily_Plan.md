# Week 4: FastConformer and Efficient Encoder Behavior

> **Days 22 to 28**  
> **Navigation:** [← Week 3](Week_3_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Week 5 →](Week_5_MendSpeech_Daily_Plan.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Measure why FastConformer is efficient and freeze a reproducible baseline.
>
> **v2 gate evidence:** Follow Gate 3 in [the execution plan](REVISED_EXECUTION_PLAN.md), shared with Week 3. This week has **5 build sessions: Days 23, 24, 25, 26, and 28**; the combined encoder block has **9 build sessions**, with completion based on evidence rather than a calendar target.
> These are base specification slots. The new decoder/LM experiment adds work
> within Days 24/26/28; estimate it after compatibility checks, not by assuming
> it fits the old session count. An incomplete decoder comparison needs a scope
> review before Gate 3 closes, even if independent streaming work proceeds.

---

## Week Map

| Day | Focus | Minimum Evidence / Artifact | Compute | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 22** | LEARN-ONLY: why FastConformer exists | Explain the efficiency choices; comparison checklist and estimates are absorbed into Day 23. | `Local CPU` | [Open Day 22](days/day_22.md) |
| **Day 23** | CORE: temporal subsampling; absorbs Day 22 | Quantify sequence length and attention cost, with the FastConformer comparison checklist. | `Modal L4 useful` | [Open Day 23](days/day_23.md) |
| **Day 24** | CORE: pretrained baseline and capability check | Cache/context/export/language and decoder/head/LM backend compatibility; no second acoustic model. | `Modal L4` | [Open Day 24](days/day_24.md) |
| **Day 25** | Context and attention limits | You can explain exactly why future context creates algorithmic latency. | `Modal L4` | [Open Day 25](days/day_25.md) |
| **Day 26** | Efficiency harness and greedy/beam/LM decoding | Held-out WER/CER, names/numbers, helpful/harmful changes; decoder-only and fresh end-to-end timings. | `Modal L4` | [Open Day 26](days/day_26.md) |
| **Day 27** | MERGED into Day 28: failure casebook | Top three repeatable failure patterns, recorded within Day 28. | `Modal L4 — within Day 28` | [Open Day 27](days/day_27.md) |
| **Day 28** | CORE: shared audio lab integration; absorbs Day 27 | One app/report/casebook with decoder provenance; offline LM evidence is not streaming support. | `Modal L4` | [Open Day 28](days/day_28.md) |

---

## v2 Compression Map (Gate Evidence)

| Day | v2 Status | Note |
| :--- | :--- | :--- |
| **Day 22** | LEARN-ONLY — merged into Day 23 | Paper notes and compute estimates only; no session |
| **Day 23** | CORE — absorbs Day 22 | Also cover the FastConformer-vs-Conformer comparison checklist |
| **Day 24** | CORE | Frozen baseline plus decoder/LM compatibility and existing checkpoint capabilities; no model hunt |
| **Day 25** | CORE | Context and attention limits |
| **Day 26** | CORE | Extend efficiency harness with three-way decoding; one small LM, validation-only tuning, blocked comparison incomplete |
| **Day 27** | MERGED into Day 28 | Keep only the top-3 failure patterns |
| **Day 28** | CORE — absorbs Day 27 | Shared app + top-3 casebook; decoder, alignment and calibration provenance, verified streaming path retained |

---

## Reference Spine
- Rekesh et al., FastConformer with Linearly Scalable Attention for Efficient Speech Recognition\nNVIDIA NeMo FastConformer documentation and model cards\nPyTorch profiler and benchmark documentation

---

## Daily Detailed Operating Plans

### DAY 22: Why FastConformer exists (LEARN-ONLY; absorbed into Day 23)
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_22.md`](days/day_22.md)

#### Learn
- Sequence length as an attention cost driver.
- Subsampling before expensive encoder blocks.
- Depthwise separable convolution.
- Local and limited context attention.

#### Build in MendSpeech
- No standalone build. Day 23 incorporates the FastConformer comparison checklist and diagram.

#### Experiment and Measure
- Day 23 incorporates attention-matrix estimates before and after temporal subsampling.

#### Required Output
- None for this learn-only session; the retained notes and estimate paths are produced within Day 23.

#### Completion Check
> You can explain FastConformer as a set of concrete efficiency choices, not just a
faster model name.

---

### DAY 23: Temporal subsampling experiment
- **Compute:** `Modal L4 useful`
- **Dedicated Daily File:** [`docs/days/day_23.md`](days/day_23.md)

> **v1 STATUS: CORE — absorbs Day 22.** Also cover Day 22's FastConformer-vs-Conformer comparison checklist and compute estimates in this session.

#### Learn
- Convolutional subsampling.
- Temporal resolution.
- Information loss versus compute reduction.

#### Build in MendSpeech
- Implement a small subsampling front end or isolate one from a framework.
- Track frames per second before and after each stage.
- Incorporate Day 22's comparison checklist and diagram of FastConformer efficiency choices.

#### Experiment and Measure
- Compare 2x, 4x, and 8x temporal reduction on tensor length, runtime, and rough output behavior.
- Estimate attention-matrix size before and after subsampling; record the absorbed Day 22 evidence.

#### Required Output
- `src/models/subsampling.py`
- `results/day23_subsampling.csv`
- `docs/day22_fastconformer_notes.md` — absorbed Day 22 evidence
- `results/day22_compute_estimates.csv` — absorbed Day 22 evidence

#### Completion Check
> You can quantify how subsampling changes sequence length and downstream
attention cost.

---

### DAY 24: Pretrained FastConformer baseline

- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_24.md`](days/day_24.md)

> **v2 STATUS: CORE — baseline and early capability check.** Verify decoding/LM, streaming, and export support on the selected checkpoint; unsupported capabilities are documented, not replaced by a model search or new architecture.

#### Learn
- Model checkpoint loading.
- Tokenizer and decoder configuration.
- Batch versus single utterance inference.
- Checkpoint-specific cache-aware inference, right-context controls, export support, and tokenizer language coverage.
- CTC versus transducer search, token/blank mapping, and external LM fusion.

#### Build in MendSpeech
- Run a current NeMo FastConformer checkpoint on your clean and damaged sets.
- Record model revision and all inference settings.
- Pin the acoustic head and tokenizer expected by Day 26's greedy/beam/LM
  experiment. Verify a supported beam backend, optional dependencies and
  small n-gram LM format against the pinned framework; do not assume these
  extras are installed or that a CTC recipe works with a transducer head.
  Record offline versus incremental support separately. Use this same acoustic
  checkpoint, not another model just to obtain a decoder comparison.
- In `configs/model_baseline.yaml`, record capability status and evidence for cache-aware inference, supported right-context values and units, runtime context switching, the intended export path, and tokenizer language support for the planned evaluation languages. Distinguish verified, unsupported, and unverified behavior.
- Check the pinned model/framework documentation and use minimal supported smoke checks where feasible. Record failures and limitations early; do not start a model hunt, retrain an encoder, or add custom infrastructure to manufacture support.

#### Experiment and Measure
- Benchmark WER, latency, and GPU memory by damage type.
- Carry the capability record into Days 25, 31–35 and later export work. Unsupported adaptive switching defers the adaptive claim, not unrelated gate evidence; unsupported cache-aware inference or export remains an explicit dependency issue, not a completed requirement.
- Include decoder compatibility smoke evidence and estimate Day 26's text
  preparation/integration effort. A blocked required comparison remains
  incomplete pending scope review; no scratch backend or open-ended model hunt.

#### Required Output
- `src/asr/fastconformer_runner.py`
- `results/day24_fastconformer_baseline.csv`
- `configs/model_baseline.yaml`

#### Completion Check
> You have a reproducible baseline with model, data, hardware, and settings fixed,
> plus an evidence-backed capability record covering cache-aware inference,
> right context, export, tokenizer language support, and decoder/head/LM backend
> compatibility. Unsupported or unverified
> capabilities and their downstream implications are explicit.

---

### DAY 25: Context and attention limits
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_25.md`](days/day_25.md)

#### Learn
- Full context attention.
- Limited context attention.
- Left and right context.
- Accuracy versus latency intuition.

#### Build in MendSpeech
- Inspect context settings in the model configuration.
- Create a visual timeline explaining visible past and future context.

#### Experiment and Measure
- If supported, compare at least two context settings on the same subset.

#### Required Output
- `docs/day25_context_timeline.md`
- `results/day25_context_compare.csv`

#### Completion Check
> You can explain exactly why future context creates algorithmic latency.

---

### DAY 26: Efficiency benchmark harness

- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_26.md`](days/day_26.md)

> **v2 STATUS: CORE — efficiency and three-way offline decoding.** Extend the shared harness; one acoustic model and one small n-gram LM. Extra integration work is not assumed to fit the original session slot.

#### Learn
- Warmup runs.
- Synchronized GPU timing.
- Median and percentile latency.
- Real time factor.
- Peak memory.
- Greedy search versus beam search; LM weight, insertion terms, and pruning.
- External LM preferences can alter content: lower WER alone is not proof that
  repair decisions are safer. Fused search scores are not calibrated probabilities.

#### Build in MendSpeech
- Create one benchmark function used by every later experiment.
- Log environment and model metadata automatically.
- Use the compatible acoustic head/backend verified in Day 24. Compare greedy,
  beam without an external LM, and beam plus one small n-gram LM in
  `src/asr/decoding.py`; no second acoustic checkpoint or scratch search engine.
- Freeze text provenance, permitted use, source IDs, normalization, hashes,
  deduplication and split roles in `data/lm_text_manifest.csv`. Exclude all
  evaluation references/duplicates from LM training; document unknown base-model
  pretraining overlap. Keep raw corpora and LM binaries in ignored storage.
- Freeze one small LM order and corpus budget plus a small validation-only
  beam-width/LM-weight candidate list in `configs/decoding.yaml`. Keep lexicon,
  insertion bonuses, tokenizer and normalization controlled; beam-only disables
  the external LM. Select once on validation, never using test errors as prompts
  for vocabulary additions, tuning, or retraining. No broad parameter sweep.
- Reuse the same acoustic outputs when the selected head/backend permits it;
  otherwise rerun identical frozen acoustic settings and label cache use. Test
  empty inputs, token mapping, and LM influence in `tests/test_asr_decoding.py`;
  CTC tests include blank/repeated-token collapse. Test timing/alignment mapping
  before decoded hypotheses can enter repair. Document backend score semantics.
- Keep offline decoding separate from the streaming runner. If compatibility is
  blocked, retain the greedy baseline and report the failed required comparison
  for scope review, not completed beam/LM integration.

#### Experiment and Measure
- Run repeated inference and calculate variance.
- Detect and discard obviously invalid cold start comparisons.
- Report WER/CER, predeclared names/numbers errors, and both helpful and harmful
  transcript changes on identical held-out clean/damaged cases. A null or worse
  LM result is valid; invented improvement or omitted failed cases are not.
- Measure decoder-only elapsed time using cached acoustic outputs separately
  from fresh audio-to-transcript latency, RTF and peak memory. Use repeated warm
  runs on fixed L4/batch/precision, separate cold measurements, and record CPU
  decoder host, thread/worker counts, cache status, timing boundaries and counts.
  Cached decoder speed is not live ASR latency; no streaming LM claim yet.

#### Required Output
- `src/bench/benchmark_asr.py`
- `src/bench/environment.py`
- `results/day26_repeatability.csv`
- `src/asr/decoding.py`
- `tests/test_asr_decoding.py`
- `configs/decoding.yaml`
- `data/lm_text_manifest.csv`
- `results/day26_decoding_comparison.csv`
- `docs/day26_decoding.md` (reproduction, tuning/split audit, score semantics,
  backend/head compatibility, help/hurt cases, and streaming limitations)

#### Completion Check
> Repeated runs support reproducible greedy/beam/LM accuracy and cost comparisons
> with validation-only selection and tests. The evidence distinguishes cached
> decoder time from fresh end-to-end latency and search scores from confidence.
> A blocked beam or LM branch leaves this comparison incomplete; independent
> work may continue, but Gate 3 requires evidence or an approved scope revision.

---

### DAY 27: FastConformer failure casebook (MERGED into Day 28)
- **Compute:** `Modal L4 — within Day 28`
- **Dedicated Daily File:** [`docs/days/day_27.md`](days/day_27.md)

#### Learn
- Error slicing by corruption type and severity.
- Short versus long utterance effects.
- Confidence versus error.

#### Build in MendSpeech
- Within Day 28, capture only the top three repeatable failure patterns; no standalone session.
- Link each case to audio, transcript, confidence, and damage metadata.

#### Experiment and Measure
- Look for systematic error patterns rather than isolated anecdotes.

#### Required Output
- `results/fastconformer_failure_casebook.md` — produced within Day 28

#### Completion Check
> You can name at least three repeatable failure patterns and propose a testable
reason for each.

---

### DAY 28: Week 4 integration

- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_28.md`](days/day_28.md)

> **v2 STATUS: CORE — absorbs Day 27.** Integration plus the top-3 failure casebook in one session, using the shared `app/audio_lab.py` entrypoint.

#### Learn
- Review efficiency choices and baseline results.

#### Build in MendSpeech
- Replace the generic ASR runner in MendSpeech with the reproducible FastConformer path.
- Extend `app/audio_lab.py`, the single app entrypoint, to expose latency, RTF, WER when reference text exists, and GPU memory. Do not create a versioned demo app.
- Capture Day 27's top three repeatable failure patterns with transcript, confidence, and damage metadata in the retained casebook.
- Link Day 26's decoding comparison with head/tokenizer/LM/config provenance
  in `reports/week4_fastconformer.md`. Show offline beam/LM output only as
  offline evidence. Retain a verified streaming-compatible decoder (greedy if
  needed); do not replace it with an offline-only backend or treat LM-fused
  scores as confidence. Changes to text/timestamps require alignment checks
  and Day 41 decoder-specific calibration before calibrated repair claims.

#### Experiment and Measure
- Run the same ten reference clips through the full Week 2 uncertainty policy using FastConformer.
- Record the actual decoder for every run. Include helpful/harmful Day 26
  examples; preserve the original greedy results instead of overwriting them.

#### Required Output
- `app/audio_lab.py`
- `results/fastconformer_failure_casebook.md` — absorbed Day 27 evidence
- `reports/week4_fastconformer.md`

#### Completion Check
> The shared audio lab exposes a measured, inspectable FastConformer recognition
> core, and the report links the top-three failure casebook and completed
> three-way decoding evidence. Offline and streaming capabilities, score
> semantics and pending calibration are explicit; blocked decoding needs scope
> review rather than a completed Gate 3 label.

---

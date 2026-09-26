# Week 4: FastConformer and Efficient Encoder Behavior

> **Days 22 to 28**  
> **Navigation:** [← Week 3](Week_3_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Week 5 →](Week_5_MendSpeech_Daily_Plan.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Measure why FastConformer is efficient and freeze a reproducible baseline.
>
> **v2 gate evidence:** Follow Gate 3 in [the execution plan](REVISED_EXECUTION_PLAN.md), shared with Week 3. This week has **5 build sessions: Days 23, 24, 25, 26, and 28**; the combined encoder block has **9 build sessions**, with completion based on evidence rather than a calendar target.

---

## Week Map

| Day | Focus | Minimum Evidence / Artifact | Compute | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 22** | LEARN-ONLY: why FastConformer exists | Explain the efficiency choices; comparison checklist and estimates are absorbed into Day 23. | `Local CPU` | [Open Day 22](days/day_22.md) |
| **Day 23** | CORE: temporal subsampling; absorbs Day 22 | Quantify sequence length and attention cost, with the FastConformer comparison checklist. | `Modal L4 useful` | [Open Day 23](days/day_23.md) |
| **Day 24** | CORE: pretrained baseline and capability check | Reproducible baseline plus cache-aware, right-context, export, and tokenizer language support evidence. | `Modal L4` | [Open Day 24](days/day_24.md) |
| **Day 25** | Context and attention limits | You can explain exactly why future context creates algorithmic latency. | `Modal L4` | [Open Day 25](days/day_25.md) |
| **Day 26** | Efficiency benchmark harness | Repeated runs produce stable enough numbers to support comparisons. | `Modal L4` | [Open Day 26](days/day_26.md) |
| **Day 27** | MERGED into Day 28: failure casebook | Top three repeatable failure patterns, recorded within Day 28. | `Modal L4 — within Day 28` | [Open Day 27](days/day_27.md) |
| **Day 28** | CORE: shared audio lab integration; absorbs Day 27 | `app/audio_lab.py`, retained report, and top-three failure casebook. | `Modal L4` | [Open Day 28](days/day_28.md) |

---

## v2 Compression Map (Gate Evidence)

| Day | v2 Status | Note |
| :--- | :--- | :--- |
| **Day 22** | LEARN-ONLY — merged into Day 23 | Paper notes and compute estimates only; no session |
| **Day 23** | CORE — absorbs Day 22 | Also cover the FastConformer-vs-Conformer comparison checklist |
| **Day 24** | CORE | Frozen baseline plus early checkpoint capability record; document unsupported behavior, no model hunt |
| **Day 25** | CORE | Context and attention limits |
| **Day 26** | CORE | Efficiency benchmark harness |
| **Day 27** | MERGED into Day 28 | Keep only the top-3 failure patterns |
| **Day 28** | CORE — absorbs Day 27 | Shared `app/audio_lab.py` integration + top-3 failure casebook |

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

> **v2 STATUS: CORE — baseline and early capability check.** Verify the selected checkpoint's supported behavior before downstream streaming and export work; unsupported capabilities are documented, not replaced by a model search or new architecture.

#### Learn
- Model checkpoint loading.
- Tokenizer and decoder configuration.
- Batch versus single utterance inference.
- Checkpoint-specific cache-aware inference, right-context controls, export support, and tokenizer language coverage.

#### Build in MendSpeech
- Run a current NeMo FastConformer checkpoint on your clean and damaged sets.
- Record model revision and all inference settings.
- In `configs/model_baseline.yaml`, record capability status and evidence for cache-aware inference, supported right-context values and units, runtime context switching, the intended export path, and tokenizer language support for the planned evaluation languages. Distinguish verified, unsupported, and unverified behavior.
- Check the pinned model/framework documentation and use minimal supported smoke checks where feasible. Record failures and limitations early; do not start a model hunt, retrain an encoder, or add custom infrastructure to manufacture support.

#### Experiment and Measure
- Benchmark WER, latency, and GPU memory by damage type.
- Carry the capability record into Days 25, 31–35 and later export work. Unsupported adaptive switching defers the adaptive claim, not unrelated gate evidence; unsupported cache-aware inference or export remains an explicit dependency issue, not a completed requirement.

#### Required Output
- `src/asr/fastconformer_runner.py`
- `results/day24_fastconformer_baseline.csv`
- `configs/model_baseline.yaml`

#### Completion Check
> You have a reproducible baseline with model, data, hardware, and settings fixed,
> plus an evidence-backed capability record covering cache-aware inference,
> right context, export, and tokenizer language support. Unsupported or unverified
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

#### Learn
- Warmup runs.
- Synchronized GPU timing.
- Median and percentile latency.
- Real time factor.
- Peak memory.

#### Build in MendSpeech
- Create one benchmark function used by every later experiment.
- Log environment and model metadata automatically.

#### Experiment and Measure
- Run repeated inference and calculate variance.
- Detect and discard obviously invalid cold start comparisons.

#### Required Output
- `src/bench/benchmark_asr.py`
- `src/bench/environment.py`
- `results/day26_repeatability.csv`

#### Completion Check
> Repeated runs produce stable enough numbers to support comparisons.

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

#### Experiment and Measure
- Run the same ten reference clips through the full Week 2 uncertainty policy using FastConformer.

#### Required Output
- `app/audio_lab.py`
- `results/fastconformer_failure_casebook.md` — absorbed Day 27 evidence
- `reports/week4_fastconformer.md`

#### Completion Check
> The shared audio lab exposes a measured, inspectable FastConformer recognition
> core, and the report links the top-three failure casebook.

---

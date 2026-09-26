# Week 3

> **Days 15–21**
> **Navigation:** [← Index](INDEX.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Executive Plan](REVISED_EXECUTION_PLAN.md)

---

> [!IMPORTANT]
> **Week theme:** Streaming capability, session loop, endpointing, and calibration
> Build and verify the streaming chunk loop, endpointing, and calibrated triage.

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 15** | Early ASR-to-editor baseline and resource pilot | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 15](days/day_15.md) |
| **Day 16** | Early SFT and GRPO feasibility on the text editor | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 16](days/day_16.md) |
| **Day 17** | Attention and Conformer architecture reading | `Local CPU` | LEARN-ONLY | [Open Day 17](days/day_17.md) |
| **Day 18** | Streaming checkpoint and decoder capability gate | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 18](days/day_18.md) |
| **Day 19** | Shape and mask checks inside the streaming capability gate | `Within Day18` | MERGED | [Open Day 19](days/day_19.md) |
| **Day 20** | Scratch production-encoder comparison outside release | `None` | DROPPED | [Open Day 20](days/day_20.md) |
| **Day 21** | Correct chunk loop and per-stream state | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 21](days/day_21.md) |

---

## Daily Detailed Operating Plans

### DAY 15: Early ASR-to-editor baseline and resource pilot
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_15.md`](days/day_15.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 13](days/day_13.md), [Day 14](days/day_14.md)
> **Effort:** 3–5 focused hours.

#### Learn
- Causal-LM generation, prompt/output token limits and KV memory.
- Server/client TTFT versus completion and validated delivery.

#### Build in MendSpeech
- Select one small causal editor candidate via docs/EDITOR_AND_RL_CONTRACT.md, pin permitted weights/tokenizer/template and isolated optional environment; do not add unverified packages to core dependencies.
- Build src/llm/polish.py and an app/audio_lab.py file/replayed-audio path using the existing ASR, deterministic/identity/LLM choices, guard and bypass. Label it offline/replayed, not completed live streaming.
- Pilot one-L4 ASR+editor co-residency with fixed token caps, serialized work and combined peak memory. ASR text is data, not instructions; no tools or agents.

#### Experiment and Measure
- Measure validation quality of raw/deterministic/prompt-only outputs, fallback, names/numbers/negation and context-free correction risks.
- Capture correlated stage events, server TTFT/completion, startup and combined memory. Record CPU threads/topology; a failed co-residency test requests scope review, not a hidden second GPU.

#### Required Output Artifacts
- `src/llm/polish.py`
- `tests/test_llm_polish.py`
- `configs/llm.yaml`
- `infra/editor/requirements.txt`
- `docs/editor_model_card.md`
- `app/audio_lab.py`
- `results/day15_e2e_baseline.csv`
- `results/day15_resource_pilot.csv`

#### Completion Check
> One real ASR→guarded-editor baseline runs before optimization; quality, TTFT/completion, memory and topology evidence are recorded without a sub-500ms promise.

---

### DAY 16: Early SFT and GRPO feasibility on the text editor
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_16.md`](days/day_16.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 13](days/day_13.md), [Day 15](days/day_15.md)
> **Effort:** 4–6 focused hours.

#### Learn
- Completion-only SFT, LoRA gradients, group-relative advantages and reference KL.
- Why CTC acoustic outputs cannot be passed to a causal-LM GRPO trainer.

#### Build in MendSpeech
- Use the selected Day15 editor and a pinned compatible optional Transformers/PEFT/TRL environment. Implement training/editor_pilot.py with the exact bounded SFT/GRPO route in the editor contract, no scratch PPO or reward model.
- Implement provisional reward component tests; one pilot group uses fresh current-policy generations. Check names/counts of trainable adapters, completion masks/EOS and group/batch divisibility.
- Declare <=10 SFT and <=5 GRPO steps, group2, fixed token caps, wall-time/spend and nonfinite/OOM stop conditions. Unload ASR while training.

#### Experiment and Measure
- Show finite loss/logprobs/gradients and adapter weight changes, reload checkpoint and inspect completions. Measure policy/reference/optimizer/rollout memory, step time and cost.
- Log reward variance/zero-variance groups and KL; failed update or insufficient rollout diversity is blocked, not a null RL result. Forecast later training cost before approval.

#### Required Output Artifacts
- `training/editor_pilot.py`
- `configs/editor_pilot.yaml`
- `src/rl/reward.py`
- `tests/test_reward.py`
- `reports/day16_training_feasibility.md`
- `results/day16_training_pilot.csv`

#### Completion Check
> An actual tiny SFT and GRPO update on the causal text editor is verified, or the full training track remains explicitly blocked before further training expenditure.

---

### DAY 17: Attention and Conformer architecture reading
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_17.md`](days/day_17.md)

> **STATUS: LEARN-ONLY**
> **Prerequisites:** [Day 09](days/day_09.md)
> **Effort:** 0–0 focused hours.

#### Learn
- Attention tensor shapes, local convolution, residual/normalization placement and context limits.

#### Build in MendSpeech
- No standalone implementation. Use existing tested CTC/audio examples and the model documentation needed for Day18.

#### Experiment and Measure
- Explain state/context costs during Day18; optional scratch exercises do not gate release.

#### Required Output Artifacts
- None; learning is included in the absorbing Day18 estimate.

#### Completion Check
> Concepts support Day18 model selection; no extra artifact or build session is counted.

---

### DAY 18: Streaming checkpoint and decoder capability gate
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_18.md`](days/day_18.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 14](days/day_14.md), [Day 15](days/day_15.md)
> **Effort:** 2–4 focused hours.

#### Learn
- Cache-aware versus buffered inference, CTC/RNN-T head compatibility and chunk lookahead.

#### Build in MendSpeech
- Select one streaming-capable ASR checkpoint compatible with the planned runtime; record revision/head/tokenizer, cache signatures, chunk/right context and inference/export support in configs/model_baseline.yaml.
- Smoke one utterance offline and via the documented streaming interface; model/decoder changes invalidate old confidence thresholds. Verify the combined Day15 editor memory with the new checkpoint.

#### Experiment and Measure
- Record output/length/context evidence, offline/streaming and LM support separately. Do not rewrite a framework to force missing support.
- Compare supported baseline transcripts on validation and classify dependency blockers before writing optimizations.

#### Required Output Artifacts
- `src/asr/streaming_runner.py`
- `tests/test_streaming_runner.py`
- `configs/model_baseline.yaml`
- `results/day18_capability_check.csv`
- `docs/day18_capabilities.md`

#### Completion Check
> Pinned streaming/cache/head behavior and combined-resource feasibility are verified; unsupported live inference is a blocker, not a filename-based success.

---

### DAY 19: Shape and mask checks inside the streaming capability gate
- **Compute:** Within Day18
- **Dedicated Daily File:** [`docs/days/day_19.md`](days/day_19.md)

> **STATUS: MERGED**
> **Prerequisites:** [Day 18](days/day_18.md)
> **Effort:** 0–0 focused hours.

#### Learn
- Feature length, cache shape, padding and valid output lengths.

#### Build in MendSpeech
- Checks are owned by Day18 tests/test_streaming_runner.py; no separate tiny encoder.

#### Experiment and Measure
- Use Day18 short/long/padded fixture results.

#### Required Output Artifacts
- None; evidence belongs to Day18.

#### Completion Check
> No standalone work; merge is explicit and does not duplicate artifacts.

---

### DAY 20: Scratch production-encoder comparison outside release
- **Compute:** None
- **Dedicated Daily File:** [`docs/days/day_20.md`](days/day_20.md)

> **STATUS: DROPPED**
> **Prerequisites:** [Day 18](days/day_18.md)
> **Effort:** 0–0 focused hours.

#### Learn
- Optional architecture study only.

#### Build in MendSpeech
- No build.

#### Experiment and Measure
- No release experiment.

#### Required Output Artifacts
- None.

#### Completion Check
> Excluded from release scope; not a completion claim.

---

### DAY 21: Correct chunk loop and per-stream state
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_21.md`](days/day_21.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 18](days/day_18.md)
> **Effort:** 3–5 focused hours.

#### Learn
- Chunk clocks, partial/final decoding, cache ownership and last-chunk flush.

#### Build in MendSpeech
- Implement src/streaming/session.py with start/push/finish/reset, bounded state, variable final chunks and sequence validation.
- Add deterministic audio replay preserving input cadence; test isolated/interleaved streams and flush/reset exactly once.

#### Experiment and Measure
- Verify repeated single-stream replay agrees with isolated interleaved streams under identical model context. Offline full-context text need not match streaming text; compare the documented same-context reference.
- Log first partial, final timestamps and state sizes. Future-context dependence cannot be called causal.

#### Required Output Artifacts
- `src/streaming/session.py`
- `tests/test_streaming_session.py`
- `experiments/replay_audio.py`
- `results/day21_streaming_correctness.csv`

#### Completion Check
> An explicit tested chunk/session loop exists before profiling, with valid finals and no cross-stream state leakage.

---

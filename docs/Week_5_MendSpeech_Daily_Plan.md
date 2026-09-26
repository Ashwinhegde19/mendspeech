# Week 5

> **Days 29–35**
> **Navigation:** [← Index](INDEX.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Executive Plan](REVISED_EXECUTION_PLAN.md)

---

> [!IMPORTANT]
> **Week theme:** Profiling, compile/graph capture, batching, precision, and the scorecard
> Optimize only what the profile shows, and record what does not help.

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 29** | Batching, concurrency and queueing | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 29](days/day_29.md) |
| **Day 30** | Precision and export parity | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 30](days/day_30.md) |
| **Day 31** | Streaming fast path and cache-failure evidence | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 31](days/day_31.md) |
| **Day 32** | Optimization scorecard and selection | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 32](days/day_32.md) |
| **Day 33** | Rebuild and revalidate the end-to-end pipeline | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 33](days/day_33.md) |
| **Day 34** | RL reward definition and falsifiability | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 34](days/day_34.md) |
| **Day 35** | Editor SFT and continued-SFT control | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 35](days/day_35.md) |

---

## Daily Detailed Operating Plans

### DAY 29: Batching, concurrency and queueing
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_29.md`](days/day_29.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 27](days/day_27.md)
> **Effort:** 2–4 focused hours.

#### Learn
- Static versus dynamic batching, queue wait versus service time, batch-size latency/throughput knee.

#### Build in MendSpeech
- Add bounded batching and per-stream queues in src/serve/batching.py; keep the co-residency pilot constraint in force.

#### Experiment and Measure
- Sweep batch size/concurrency within budget; report throughput, per-stream p50/p95/p99, queue wait, achieved concurrency and memory.
- Verify no cross-stream state contamination and report starvation/fairness; batching is not required to be selected.

#### Required Output Artifacts
- `src/serve/batching.py`
- `tests/test_batching.py`
- `results/day29_batch_sweep.csv`
- `docs/day29_queueing.md`

#### Completion Check
> Throughput and single-stream latency are reported separately with the knee and fairness behavior.

---

### DAY 30: Precision and export parity
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_30.md`](days/day_30.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 27](days/day_27.md)
> **Effort:** 2–4 focused hours.

#### Learn
- FP16/INT8 (dynamic and static), calibration sets, exported-versus-original parity.

#### Build in MendSpeech
- Smoke-check export/precision support for the selected backend in an isolated pinned environment; document in docs/day30_quant_notes.md.
- Verify original-versus-exported parity first; then apply supported FP16/INT8 with a calibration slice from training/calibration only, never the frozen test.

#### Experiment and Measure
- Measure WER/CER, latency percentiles, RTF, memory and confidence/logit shifts per precision; recheck Day25 calibration binding.
- State explicitly if a precision is slower or degrades accuracy; blocked precision leaves the required comparison incomplete with recorded blocker.

#### Required Output Artifacts
- `src/asr/quantized_runner.py`
- `docs/day30_quant_notes.md`
- `results/day30_quantization_tradeoffs.csv`
- `app/audio_lab.py`

#### Completion Check
> Parity is verified before any precision claim, and every supported precision has measured accuracy/latency/memory or a documented blocker.

---

### DAY 31: Streaming fast path and cache-failure evidence
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_31.md`](days/day_31.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 21](days/day_21.md), [Day 24](days/day_24.md), [Day 28](days/day_28.md), [Day 30](days/day_30.md)
> **Effort:** 2–4 focused hours.

#### Learn
- State reuse versus recomputation across chunks and reset/truncation failure modes.

#### Build in MendSpeech
- Add a streaming fast path in src/streaming/fast_path.py reusing the best supported variant, with state equivalence tests.
- Break the cache deliberately at chosen boundaries via src/streaming/cache_stress.py to characterize failure.

#### Experiment and Measure
- Report steady-state per-chunk latency separate from first chunk; verify cached vs uncached transcripts.
- Record WER changes and whether errors cluster or propagate at reset points; this failure evidence is required release material.

#### Required Output Artifacts
- `src/streaming/fast_path.py`
- `tests/test_fast_path_parity.py`
- `src/streaming/cache_stress.py`
- `results/day31_cache_failures.md`
- `results/day31_fast_path.csv`

#### Completion Check
> A measured streaming fast path with parity and a concrete, reproducible cache-state failure.

---

### DAY 32: Optimization scorecard and selection
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_32.md`](days/day_32.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 26](days/day_26.md), [Day 28](days/day_28.md), [Day 29](days/day_29.md), [Day 30](days/day_30.md), [Day 31](days/day_31.md)
> **Effort:** 2–3 focused hours.

#### Learn
- Multi-objective selection, Pareto frontiers and honest negative reporting.

#### Build in MendSpeech
- Build the scorecard generator in src/bench/scorecard.py over all measured variants.
- Record a what_did_not_help section; select one shipping candidate and mark the selection provisional pending final-stack revalidation.

#### Experiment and Measure
- Tabulate WER, latency percentiles, RTF, memory and calibration status per variant.
- Select on measured grounds and identify the largest remaining bottleneck for the end-to-end path.

#### Required Output Artifacts
- `src/bench/scorecard.py`
- `results/day32_optimization_scorecard.csv`
- `docs/day32_optimization_report.md`

#### Completion Check
> A defensible provisional shipping configuration with Pareto evidence including the techniques that failed.

---

### DAY 33: Rebuild and revalidate the end-to-end pipeline
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_33.md`](days/day_33.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 25](days/day_25.md), [Day 26](days/day_26.md), [Day 32](days/day_32.md)
> **Effort:** 2–4 focused hours.

#### Learn
- Provisional optimization results must be re-checked after pipeline integration.

#### Build in MendSpeech
- Rebuild src/streaming, src/serve and app/audio_lab.py around the selected candidate with the Day25 calibration and guard contract intact.
- Re-verify streaming parity, confidence binding and the joint memory/interference pilot on the selected stack.

#### Experiment and Measure
- Re-measure Day26 baseline quality/latency/memory end-to-end; deltas vs provisional are explained.
- If the candidate is infeasible jointly, record the blocker and scope-review options instead of forcing the stack.

#### Required Output Artifacts
- `results/day33_rebuild_revalidation.csv`
- `docs/day33_rebuild_notes.md`

#### Completion Check
> The optimized end-to-end pipeline is revalidated, and any change in behavior versus the provisional baseline is explained.

---

### DAY 34: RL reward definition and falsifiability
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_34.md`](days/day_34.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 13](days/day_13.md), [Day 16](days/day_16.md), [Day 25](days/day_25.md)
> **Effort:** 3–5 focused hours.

#### Learn
- Reward hacking, faithful rewards, and a bounded group-relative RL objective on the text editor.

#### Build in MendSpeech
- Design the conservative reward on the Day13 contract: edit/format fidelity, protected-span safety, length and fluency penalties, with a per-example safety floor.
- Show a short-falsifiable prediction before training: which validation failures the reward should reduce and which must not increase.
- Use the pilot path from Day16; do not train on the CTC acoustic model and do not handcraft a per-example rewrite.

#### Experiment and Measure
- Construct adversarial cases (empty/truncated output, negation removal, entity edits, prompt-echo, runaway length) and verify each is penalized.
- Confirm the reward is computable offline on validation; log reward variance and any zero-variance groups. A reward that cannot be gamed by refusal earns nothing on needs-edit cases.

#### Required Output Artifacts
- `src/rl/reward.py`
- `configs/editor_reward.yaml`
- `tests/test_reward.py`
- `docs/day34_reward_design.md`

#### Completion Check
> A written falsifiable prediction plus reward tests that demonstrate the failure modes the reward is designed to penalize.

---

### DAY 35: Editor SFT and continued-SFT control
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_35.md`](days/day_35.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 16](days/day_16.md), [Day 34](days/day_34.md)
> **Effort:** 3–5 focused hours.

#### Learn
- Supervised fine-tuning for constrained text editing and the compute-matched continued-SFT control.

#### Build in MendSpeech
- Run editor SFT via training/editor_sft.py on the Day13 train split; freeze the SFT checkpoint as the RL reference.
- Run a continued-SFT control matched to the future RL wall-time/token budget, so extra compute is not mistaken for the RL algorithm.

#### Experiment and Measure
- Evaluate SFT and continued-SFT on validation (protected-content violations, formatting accuracy, identity vs needs-edit, risk-coverage).
- Log trainable-parameter names/counts, memory, step time and cost; artifacts for both arms.

#### Required Output Artifacts
- `training/editor_sft.py`
- `configs/editor_sft.yaml`
- `results/day35_sft_vs_continued.csv`
- `docs/day35_sft_notes.md`

#### Completion Check
> An SFT editor and a compute-matched continued-SFT control exist so any later RL gain cannot be explained by extra training alone.

---

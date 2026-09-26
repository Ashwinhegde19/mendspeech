# Week 6

> **Days 36–42**
> **Navigation:** [← Index](INDEX.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Executive Plan](REVISED_EXECUTION_PLAN.md)

---

> [!IMPORTANT]
> **Week theme:** Editor reward, SFT and compute-matched control, GRPO, and ASR robustness adaptation
> Post-train the text editor with explicit controls, and adapt the recognizer for acoustic robustness.

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 36** | Training pipeline anatomy for the editor | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 36](days/day_36.md) |
| **Day 37** | ASR robustness adaptation dataset and leakage audit | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 37](days/day_37.md) |
| **Day 38** | ASR robustness fine-tuning (not personalization) | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 38](days/day_38.md) |
| **Day 39** | Augmentation ablation on corrupted audio | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 39](days/day_39.md) |
| **Day 40** | Bounded GRPO post-training on the editor | `Modal L4; spend/spend-capped` | CORE | [Open Day 40](days/day_40.md) |
| **Day 41** | Personalization-adjacent robustness and error analysis | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 41](days/day_41.md) |
| **Day 42** | Editor personalization feasibility check (scope, not claim) | `Local CPU` | CORE | [Open Day 42](days/day_42.md) |

---

## Daily Detailed Operating Plans

### DAY 36: Training pipeline anatomy for the editor
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_36.md`](days/day_36.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 35](days/day_35.md)
> **Effort:** 2–3 focused hours.

#### Learn
- Loss curves, overfitting detection, tokenizer/label masking and checkpoint reproducibility.

#### Build in MendSpeech
- Diagnose the Day35 SFT loss/eval curves and add reproducibility checks in training/editor_sft.py.
- Document data mix, label masking and checkpoint reload determinism.

#### Experiment and Measure
- Verify checkpoint reload reproduces validation numbers.
- Relate loss/overfit behaviour to editor data size so RL budgets are set on evidence.

#### Required Output Artifacts
- `docs/day36_editor_training_diagnosis.md`
- `results/day36_editor_loss_curves.csv`

#### Completion Check
> Editor training behaviour is diagnosable and reproducible, giving evidence-based budgets for the RL run.

---

### DAY 37: ASR robustness adaptation dataset and leakage audit
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_37.md`](days/day_37.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 10](days/day_10.md), [Day 18](days/day_18.md)
> **Effort:** 2–3 focused hours.

#### Learn
- Acoustic corruption manifests and the frozen benchmark invariant.

#### Build in MendSpeech
- Build data/train_manifest.jsonl and val_manifest.jsonl from separate non-frozen source audio; keep data/test_manifest.jsonl byte-identical to the frozen set.
- Audit source/speaker leakage and corruption provenance; document in reports/data_audit.md.

#### Experiment and Measure
- Prove no source/speaker crosses splits; report severity distribution.
- The frozen test set is not used for training, tuning or checkpoint selection in this phase.

#### Required Output Artifacts
- `data/train_manifest.jsonl`
- `data/val_manifest.jsonl`
- `data/test_manifest.jsonl`
- `reports/data_audit.md`

#### Completion Check
> The adaptation dataset is leakage-free and the frozen evaluation set is provably untouched.

---

### DAY 38: ASR robustness fine-tuning (not personalization)
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_38.md`](days/day_38.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 24](days/day_24.md), [Day 25](days/day_25.md), [Day 37](days/day_37.md)
> **Effort:** 3–5 focused hours.

#### Learn
- Transfer learning, frozen versus trainable layers, mixed precision. This is acoustic robustness, not user personalization.

#### Build in MendSpeech
- Fine-tune the pinned streaming checkpoint with training/asr_finetune.py under configs/asr_finetune.yaml (steps, LR, seed, sampling frozen).
- Bind and re-validate Day25 calibration for the adapted checkpoint before it informs triage.

#### Experiment and Measure
- Compare base vs adapted on the frozen test set per corruption/severity with clean-speech regression.
- An adaptation that helps damaged speech but harms clean speech is a documented trade-off; no personalization claim is made.

#### Required Output Artifacts
- `training/asr_finetune.py`
- `configs/asr_finetune.yaml`
- `results/day38_base_vs_adapted.csv`
- `reports/day38_robustness_adaptation.md`

#### Completion Check
> Acoustic robustness is measured with clean-speech regression on the frozen set, and is reported as adaptation rather than personalization.

---

### DAY 39: Augmentation ablation on corrupted audio
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_39.md`](days/day_39.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 38](days/day_38.md)
> **Effort:** 2–3 focused hours.

#### Learn
- SpecAugment, room impulse-response augmentation and training-time confounds.

#### Build in MendSpeech
- Run one controlled augmentation arm with identical steps/seed via experiments/augmentation_ablation.py.
- Test augmentation strength and label-preserving transforms; never alter the frozen set.

#### Experiment and Measure
- Compare no-augmentation vs augmentation at equal budget, then give the extra steps to the unaugmented baseline.
- Record the gain (or its absence) and per-corruption effect.

#### Required Output Artifacts
- `experiments/augmentation_ablation.py`
- `results/day39_augmentation.csv`

#### Completion Check
> The effect of augmentation is separated from the effect of extra training time.

---

### DAY 40: Bounded GRPO post-training on the editor
- **Compute:** Modal L4; spend/spend-capped
- **Dedicated Daily File:** [`docs/days/day_40.md`](days/day_40.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 16](days/day_16.md), [Day 34](days/day_34.md), [Day 35](days/day_35.md), [Day 36](days/day_36.md)
> **Effort:** 4–8 focused hours.

#### Learn
- Group-relative policy optimization, KL to the SFT reference, reward variance, rollout cost.

#### Build in MendSpeech
- Run the bounded GRPO run using training/editor_rl.py with the Day34 reward and Day35 SFT reference; default to LoRA-scale updates, modest steps, and a declared stop/spend budget.
- Log reward curves, KL to reference, group reward variance, completion length, adapter grad norms and refusals.

#### Experiment and Measure
- Evaluate against the SFT and continued-SFT controls on validation: protected-content violations, formatting accuracy, identity vs needs-edit, risk-coverage.
- Stop on nonfinite loss, repeated OOM, or safety violations rising >2pp above the SFT baseline at two consecutive evals; a valid null is kept, a failed run is blocked not disguised.

#### Required Output Artifacts
- `training/editor_rl.py`
- `configs/editor_rl.yaml`
- `results/day40_rl_vs_sft.csv`
- `docs/day40_rl_findings.md`
- `results/day40_reward_curve.csv`

#### Completion Check
> A bounded, controlled GRPO run on the text editor with SFT/continued-SFT comparison and explicit stop criteria, or a documented blocked/null outcome.

---

### DAY 41: Personalization-adjacent robustness and error analysis
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_41.md`](days/day_41.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 38](days/day_38.md), [Day 39](days/day_39.md), [Day 40](days/day_40.md)
> **Effort:** 2–4 focused hours.

#### Learn
- Acoustic robustness adaptation versus user personalization; error taxonomy for the shipped path.

#### Build in MendSpeech
- Build the failure casebook in reports/casebook.md for the robustness-adapted checkpoint plus the RL editor across corruption/severity.
- Explicitly label Day38 as robustness adaptation; define the personalization evaluation this release does NOT claim.

#### Experiment and Measure
- Rank failure modes by frequency and severity across the frozen matrix.
- Verify each failure mode has an owner stage (ASR, triage, editor, guard, or endpoint) so the report can attribute causality.

#### Required Output Artifacts
- `reports/casebook.md`
- `results/day41_failure_frequency.csv`

#### Completion Check
> A ranked, stage-attributed failure casebook, and an honest statement that this release measures acoustic robustness rather than user personalization.

---

### DAY 42: Editor personalization feasibility check (scope, not claim)
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_42.md`](days/day_42.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 13](days/day_13.md), [Day 41](days/day_41.md)
> **Effort:** 2–3 focused hours.

#### Learn
- User-specific vocabulary/corrections would be personalization; this session sizes it without claiming it.

#### Build in MendSpeech
- Write a feasibility memo on what per-user enrollment, correction history and a personal lexicon would require in data, time and budget.
- Compare to the released scope; recommend keep, defer or drop with reasons.

#### Experiment and Measure
- No model training. Report effort estimates and dependency blockers.
- The memo must state that a personalization claim is not made unless this work is separately approved and executed.

#### Required Output Artifacts
- `docs/day42_personalization_feasibility.md`

#### Completion Check
> A written feasibility memo decides the scope of personalization without pretending it was achieved.

---

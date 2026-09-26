# Week 6: Robustness, Fine Tuning, RNNT, and Calibration

> **Days 36 to 42**  
> **Navigation:** [← Week 5](Week_5_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Week 7 →](Week_7_MendSpeech_Daily_Plan.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Adapt the recognizer to damaged speech while learning training and calibration discipline.
>
> **v2 evidence gate:** Days 36–42 remain CORE. Gate 5 requires one ASR
> adaptation experiment with clean regression, validation-based calibration and
> the robustness report in the one app. Day 40 retains the quantization lab with
> compatibility/parity checks: measured supported precision or explicit blocked
> optimization status, never an invented speedup. No calendar deadline applies.

---

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 36** | Training pipeline anatomy | `Modal L4` | CORE | [Open Day 36](days/day_36.md) |
| **Day 37** | Adaptation dataset and leakage audit | `Local CPU` | CORE | [Open Day 37](days/day_37.md) |
| **Day 38** | Fine-tune for damaged-speech robustness | `Modal L4` | CORE | [Open Day 38](days/day_38.md) |
| **Day 39** | Augmentation ablation | `Modal L4` | CORE | [Open Day 39](days/day_39.md) |
| **Day 40** | RL reward design | `Modal L4` | CORE | [Open Day 40](days/day_40.md) |
| **Day 41** | RL post-training run | `Modal L4` | CORE | [Open Day 41](days/day_41.md) |
| **Day 42** | Personalization comparison and robustness milestone | `Modal L4` | CORE | [Open Day 42](days/day_42.md) |

---

## Phase Focus

Personalization, fine-tuning, and RL post-training

---

## Daily Detailed Operating Plans
### DAY 36: Training pipeline anatomy
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_36.md`](days/day_36.md)

> **v3 STATUS: CORE** Phase P5 begins. You cannot fine-tune responsibly if you cannot read a loss curve.
#### Learn
- Manifest format.
- Batching variable-duration audio.
- Loss curves.
- Learning rate, validation split, checkpointing.
#### Build in MendSpeech
- Create one reproducible training configuration in `configs/train_smoke.yaml`.
- Implement the training loop in `training/train.py` with checkpointing and validation hooks.
#### Experiment and Measure
- Run a short smoke job and verify the loss decreases.
- Deliberately use a bad learning rate and record the failure signature in `results/day36_training_smoke.csv`.
- Verify checkpoints reload and reproduce the same validation number.
#### Required Output
['- `configs/train_smoke.yaml`', '- `training/train.py`', '- `tests/test_train_loop.py`', '- `results/day36_training_smoke.csv`']
#### Completion Check
> You can diagnose whether a run is learning, diverging, or overfitting from basic evidence, and a checkpoint reloads reproducibly.

---

### DAY 37: Adaptation dataset and leakage audit
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_37.md`](days/day_37.md)

> **v3 STATUS: CORE** Leaked data makes every later number meaningless, so this session precedes training.
#### Learn
- Train, validation, and test separation.
- Speaker leakage.
- Synthetic corruption sampling.
- Balanced severity distribution.
#### Build in MendSpeech
- Build manifests pairing clean transcripts with corrupted audio for one adaptation experiment in `data/`.
- Preserve the already-frozen test membership and speaker-separated splits.
- Audit source duplicates and speaker leakage, and write the audit to `reports/data_audit.md`.
#### Experiment and Measure
- Prove no source or speaker appears in more than one split, including via corrupted copies.
- Report the severity distribution and correct any imbalance before training.
#### Required Output
['- `data/train_manifest.jsonl`', '- `data/val_manifest.jsonl`', '- `data/test_manifest.jsonl`', '- `reports/data_audit.md`']
#### Completion Check
> The evaluation set cannot appear in training through clean or corrupted duplicates, and the audit shows how you know.

---

### DAY 38: Fine-tune for damaged-speech robustness
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_38.md`](days/day_38.md)

> **v3 STATUS: CORE** The personalization experiment: base versus adapted, measured with a clean-speech regression check.
#### Learn
- Transfer learning.
- Frozen versus trainable layers.
- Mixed precision.
- Gradient accumulation.
#### Build in MendSpeech
- Fine-tune the Day 24 checkpoint on the Day 37 dataset using `training/finetune.py`.
- Freeze trainable layers, learning rate, steps, seed, and sampling in `configs/finetune.yaml`.
#### Experiment and Measure
- Compare base and adapted models on the frozen test set, reporting WER per corruption and severity.
- Measure clean-speech regression explicitly; an adaptation that helps damaged speech but harms clean speech is a documented tradeoff, not a win.
- Record actual training time and cost.
#### Required Output
['- `training/finetune.py`', '- `configs/finetune.yaml`', '- `reports/day38_adaptation.md`', '- `results/day38_base_vs_adapted.csv`']
#### Completion Check
> You can state exactly what improved, what did not, and whether clean speech regressed.

---

### DAY 39: Augmentation ablation
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_39.md`](days/day_39.md)

> **v3 STATUS: CORE** Confound control: augmentation must be separated from extra training time.
#### Learn
- Time masking.
- Frequency masking.
- Data augmentation as invariance training.
#### Build in MendSpeech
- Add one augmentation intervention to a controlled short run in `experiments/specaugment_ablation.py`.
#### Experiment and Measure
- Compare no augmentation versus selected augmentation with the same seed and the same step budget.
- Report whether the gain survives when the extra steps are given to the unaugmented baseline.
#### Required Output
['- `experiments/specaugment_ablation.py`', '- `results/day39_augmentation.csv`']
#### Completion Check
> You can separate the effect of augmentation from the effect of extra training time.

---

### DAY 40: RL reward design
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_40.md`](days/day_40.md)

> **v3 STATUS: CORE** The reward must be falsifiable, or the run proves nothing. This is the most novel session in the plan.
#### Learn
- Policy-gradient and PPO intuition for sequence output.
- Reward hacking: what a model does when the reward is exploitable.
- Designing a reward that is falsifiable in advance.
#### Build in MendSpeech
- Define the reward in `src/rl/reward.py`: penalize fluent output that the acoustics do not support.
- Write the falsifiable prediction in `configs/rl.yaml` BEFORE running anything.
- Implement a minimal policy-gradient or PPO-style update in `src/rl/ppo.py`.
#### Experiment and Measure
- Show the reward can be gamed: construct at least one input where a naive reward rewards a wrong transcript.
- Verify the reward is computable offline from cached logits before spending GPU time.
- Unit-test reward components in `tests/test_reward.py`.
#### Required Output
['- `src/rl/reward.py`', '- `src/rl/ppo.py`', '- `configs/rl.yaml`', '- `tests/test_reward.py`', '- `docs/day40_reward_design.md`']
#### Completion Check
> You have a written falsifiable prediction, a reward shown to be gameable in at least one case, and a tested implementation.

---

### DAY 41: RL post-training run
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_41.md`](days/day_41.md)

> **v3 STATUS: CORE** Base versus fine-tuned versus RL, on the same held-out data. A null result here is still a result.
#### Learn
- Reward/advantage computation.
- KL regularization against the reference model.
- Why RL can degrade a well-calibrated model.
#### Build in MendSpeech
- Run the bounded RL post-training from `training/rl_train.py` using the Day 38 checkpoint as reference.
- Track reward, KL, and held-out WER together; reward rising while WER worsens is the key diagnostic.
#### Experiment and Measure
- Compare base, fine-tuned, and RL variants on held-out data.
- Re-run the Day 13 risk-coverage analysis for the RL model; improved WER does not imply improved triage safety.
- Record total GPU cost against the declared budget in `results/day41_rl_vs_baseline.csv`.
#### Required Output
['- `training/rl_train.py`', '- `results/day41_rl_vs_baseline.csv`', '- `docs/day41_rl_findings.md`', '- `results/day41_risk_coverage.csv`']
#### Completion Check
> You can state whether RL helped, did nothing, or hurt, with evidence, and you checked triage safety rather than WER alone.

---

### DAY 42: Personalization comparison and robustness milestone
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_42.md`](days/day_42.md)

> **v3 STATUS: CORE** One table answers the personalization question and closes Phase P5.
#### Learn
- Separating adaptation effects from training-time effects.
- Reporting a null result without overclaiming.
#### Build in MendSpeech
- Produce the final base/fine-tuned/RL comparison in `reports/day42_personalization.md`.
- Extend `app/audio_lab.py` to switch between the base, fine-tuned, and RL checkpoints.
#### Experiment and Measure
- Report WER per corruption and severity for all three checkpoints, plus clean-speech regression.
- Report the risk-coverage curve for each checkpoint.
- State plainly which checkpoint ships and why the choice rests on measured evidence.
#### Required Output
['- `reports/day42_personalization.md`', '- `results/day42_personalization_matrix.csv`', '- `app/audio_lab.py`']
#### Completion Check
> The personalization question is answered with a table and a shipping recommendation, including any null results.

---

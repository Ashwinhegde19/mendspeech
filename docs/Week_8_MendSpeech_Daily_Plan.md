# Week 8

> **Days 50–56**
> **Navigation:** [← Index](INDEX.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Executive Plan](REVISED_EXECUTION_PLAN.md)

---

> [!IMPORTANT]
> **Week theme:** Frozen evaluation, report, and release
> Freeze the protocol, run the matrix and ablations, and write the report and reproduction guide.

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 50** | Freeze the evaluation protocol and claims | `Local CPU` | CORE | [Open Day 50](days/day_50.md) |
| **Day 51** | Release SpeechDamageBench v1 and freeze the evaluation set | `Local CPU` | CORE | [Open Day 51](days/day_51.md) |
| **Day 52** | Robustness matrix on the frozen set | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 52](days/day_52.md) |
| **Day 53** | Optimization, serving and editor ablations on the frozen harness | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 53](days/day_53.md) |
| **Day 54** | Adaptation and RL final comparison | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 54](days/day_54.md) |
| **Day 55** | Technical report and reproduction guide | `Local CPU` | CORE | [Open Day 55](days/day_55.md) |
| **Day 56** | Final demo, clean reproduction and release | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 56](days/day_56.md) |

---

## Daily Detailed Operating Plans

### DAY 50: Freeze the evaluation protocol and claims
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_50.md`](days/day_50.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 32](days/day_32.md), [Day 40](days/day_40.md), [Day 46](days/day_46.md), [Day 48](days/day_48.md)
> **Effort:** 1–2 focused hours.

#### Learn
- Pre-registration, null outcomes and the claims this release will not make.

#### Build in MendSpeech
- Freeze code/model/data revisions, hardware, corruption configs, editor prompt/reward, and metrics in configs/frozen.yaml.
- Write experiments/protocol.md: baselines, statistical caveat, null outcomes, failure criteria and excluded claims.

#### Experiment and Measure
- Run a dry run confirming every required field has a measurement or explicit status.
- No test-set inspection after this point; new experiments get new configs, never a new test set.

#### Required Output Artifacts
- `configs/frozen.yaml`
- `experiments/protocol.md`
- `docs/day50_protocol.md`

#### Completion Check
> The evaluation protocol, baselines and claim limits are frozen before the final measurement phase.

---

### DAY 51: Release SpeechDamageBench v1 and freeze the evaluation set
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_51.md`](days/day_51.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 37](days/day_37.md), [Day 50](days/day_50.md)
> **Effort:** 1–2 focused hours.

#### Learn
- Severity grids, speaker-separated evaluation, seed control and checksum verification.

#### Build in MendSpeech
- Finalize the standalone package and lock manifest checksums under benchmarks/.
- Document a one-command reproduction example in speechdamagebench/README.md.

#### Experiment and Measure
- Reinstall in a clean environment; regenerate a sample and verify its checksum.
- Verify clean references are byte-identical after regeneration.

#### Required Output Artifacts
- `speechdamagebench/CHANGELOG.md`
- `benchmarks/manifest.csv`
- `benchmarks/README.md`

#### Completion Check
> A clean environment reproduces a benchmark item from the manifest and clean references are provably unchanged.

---

### DAY 52: Robustness matrix on the frozen set
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_52.md`](days/day_52.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 38](days/day_38.md), [Day 39](days/day_39.md), [Day 51](days/day_51.md)
> **Effort:** 2–3 focused hours.

#### Learn
- Sliced evaluation, multiple comparisons and variance.

#### Build in MendSpeech
- Run the full corruption x severity x decoder grid in src/bench/run_matrix.py on the frozen set.
- Report the robustness-adapted checkpoint and the base checkpoint in the same matrix.

#### Experiment and Measure
- Report WER/CER, names/numbers and confidence behaviour per cell.
- Estimate variance on a representative subset; identify the worst cell.

#### Required Output Artifacts
- `src/bench/run_matrix.py`
- `results/day52_robustness_matrix.csv`
- `results/day52_robustness_matrix.png`

#### Completion Check
> The complete robustness matrix is measured on the frozen set, with the worst cell and variance identified.

---

### DAY 53: Optimization, serving and editor ablations on the frozen harness
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_53.md`](days/day_53.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 32](days/day_32.md), [Day 45](days/day_45.md), [Day 46](days/day_46.md), [Day 48](days/day_48.md), [Day 52](days/day_52.md)
> **Effort:** 2–3 focused hours.

#### Learn
- Pareto frontiers; holding inputs fixed so comparisons mean something.

#### Build in MendSpeech
- Run every optimization variant, serving configuration and editor variant on the identical frozen subset in src/bench/run_ablations.py.
- Keep live measurements separate from simulated estimates.

#### Experiment and Measure
- Plot WER vs p99 latency and mark Pareto-efficient points.
- Report editor variants by quality/latency and state the shipped configuration.

#### Required Output Artifacts
- `src/bench/run_ablations.py`
- `results/day53_ablations.csv`
- `results/day53_pareto.png`

#### Completion Check
> A measured ablation set with Pareto frontiers and a stated shipping configuration.

---

### DAY 54: Adaptation and RL final comparison
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_54.md`](days/day_54.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 38](days/day_38.md), [Day 40](days/day_40.md), [Day 41](days/day_41.md), [Day 53](days/day_53.md)
> **Effort:** 2–3 focused hours.

#### Learn
- Separating acoustic robustness adaptation from text-editor post-training.

#### Build in MendSpeech
- Run base, robustness-adapted, SFT-editor and RL-editor conditions through the frozen harness in src/bench/run_personalization.py.
- Report each condition on its own axis: ASR WER/robustness for the checkpoint; editor quality for the editor.

#### Experiment and Measure
- Report the frozen-test numbers for all conditions, including clean-speech regression.
- State whether adaptation and RL each earned their place; a null result is reported, not hidden.

#### Required Output Artifacts
- `src/bench/run_personalization.py`
- `results/day54_conditions_final.csv`
- `reports/day54_conditions_final.md`

#### Completion Check
> The final conditions are compared on frozen evidence, with adaptation and editor quality kept distinct and nulls reported.

---

### DAY 55: Technical report and reproduction guide
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_55.md`](days/day_55.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 50](days/day_50.md), [Day 52](days/day_52.md), [Day 53](days/day_53.md), [Day 54](days/day_54.md)
> **Effort:** 2–4 focused hours.

#### Learn
- Observation versus causal claim; honest negative results; reproducibility.

#### Build in MendSpeech
- Write REPORT.md with reproduction commands and environment capture; write REPRODUCE.md.
- Include the latency budget, robustness matrix, adaptation/RL results and editor quality as dedicated sections.

#### Experiment and Measure
- Audit every major claim against a table, figure or experiment.
- List every blocked, deferred or partial capability in docs/limitations_and_claims.md and soften unsupported conclusions.

#### Required Output Artifacts
- `REPORT.md`
- `REPRODUCE.md`
- `results/final_figures/`
- `docs/limitations_and_claims.md`

#### Completion Check
> A technical reader understands the contribution, trade-offs and limitations without opening the source.

---

### DAY 56: Final demo, clean reproduction and release
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_56.md`](days/day_56.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 55](days/day_55.md)
> **Effort:** 2–3 focused hours.

#### Learn
- Honest demonstration including failure modes; reproducible release.

#### Build in MendSpeech
- Extend only app/audio_lab.py: live/replayed audio, partial/final transcripts, confidence, triage, guarded edit, latency budget, one failure case.
- Reproduce one frozen benchmark from a fresh environment and tag a release.

#### Experiment and Measure
- Verify every public chart regenerates from saved result files.
- Confirm displayed numbers match committed artifacts; show a failure case, not only the success path.

#### Required Output Artifacts
- `app/audio_lab.py`
- `REPRODUCE.md`
- `demos/final_demo.mp4`
- `docs/architecture.md`

#### Completion Check
> A new user can run, evaluate and reproduce the system, and every number shown traces to a committed artifact.

---

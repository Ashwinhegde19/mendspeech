# Week 8: Research Capstone: Controlled Repair Comparisons

> **Days 50 to 56**  
> **Navigation:** [← Week 7](Week_7_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Index →](INDEX.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Freeze the benchmark, run controlled ablations, compare architectures, and publish a reproducible result.
>
> **v2 evidence gate:** Gate 7 requires frozen comparisons, controlled ablations,
> a technical report, and clean reproduction in the one `app/audio_lab.py`.
> The compression map groups work; it guarantees neither session count nor
> compute cost. One external restoration comparator is conditional on Week 2
> feasibility; unavailable or unsupported capabilities are not tested results.

---

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 50** | Freeze the evaluation protocol | `Local CPU with Modal L4 dry run` | CORE | [Open Day 50](days/day_50.md) |
| **Day 51** | Release SpeechDamageBench v1 and freeze the evaluation set | `Local CPU` | CORE | [Open Day 51](days/day_51.md) |
| **Day 52** | Robustness matrix on the frozen set | `Modal L4` | CORE | [Open Day 52](days/day_52.md) |
| **Day 53** | Optimization and serving ablations | `Modal L4` | CORE | [Open Day 53](days/day_53.md) |
| **Day 54** | Personalization comparison on the frozen harness | `Modal L4` | CORE | [Open Day 54](days/day_54.md) |
| **Day 55** | Technical report and reproduction guide | `Local CPU` | CORE | [Open Day 55](days/day_55.md) |
| **Day 56** | Final demo, clean reproduction, and release | `Modal L4 plus local interface` | CORE | [Open Day 56](days/day_56.md) |

---

## Phase Focus

Frozen evaluation, report, and release

---

## Daily Detailed Operating Plans
### DAY 50: Freeze the evaluation protocol
- **Compute:** `Local CPU with Modal L4 dry run`
- **Dedicated Daily File:** [`docs/days/day_50.md`](days/day_50.md)

> **v3 STATUS: CORE** Phase P7 begins. Freeze before measuring, or the measurement decides the protocol.
#### Learn
- What makes an evaluation protocol reproducible.
- Pre-registering claims so results cannot be reinterpreted afterwards.
#### Build in MendSpeech
- Freeze code, model, and data revisions, hardware, corruption configs, and metrics in `configs/frozen.yaml`.
- Define baselines and claims you will NOT make in `experiments/protocol.md`.
- Define null outcomes and failure criteria in advance.
#### Experiment and Measure
- Run a dry run to confirm every required field has a measurement or an explicit status.
- Scope every claim to the benchmark scale and state the statistical caveat.
#### Required Output
['- `configs/frozen.yaml`', '- `experiments/protocol.md`', '- `docs/day50_protocol.md`']
#### Completion Check
> Another engineer can reproduce the supported comparisons and knows exactly which claims are out of scope.

---

### DAY 51: Release SpeechDamageBench v1 and freeze the evaluation set
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_51.md`](days/day_51.md)

> **v3 STATUS: CORE** The frozen set is the project's anchor; new experiments get new configs, never a new test set.
#### Learn
- Severity grids.
- Speaker-separated evaluation.
- Seed control and deterministic manifests.
- Package versioning and checksum verification.
#### Build in MendSpeech
- Finalize the standalone package and lock manifest checksums in `benchmarks/`.
- Document a one-command example that reproduces one benchmark item in `speechdamagebench/README.md`.
#### Experiment and Measure
- Reinstall the package in a clean environment.
- Regenerate a sample from the manifest and verify its checksum.
- Verify clean references are byte-identical after regeneration.
#### Required Output
['- `speechdamagebench/CHANGELOG.md`', '- `benchmarks/manifest.csv`', '- `benchmarks/README.md`']
#### Completion Check
> A clean environment reproduces a benchmark item from the manifest, and clean references are provably unchanged.

---

### DAY 52: Robustness matrix on the frozen set
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_52.md`](days/day_52.md)

> **v3 STATUS: CORE** The full corruption x severity x decoder grid, on data nobody can now change.
#### Learn
- Why a full matrix beats spot checks.
- Multiple-comparison discipline when slicing results.
#### Build in MendSpeech
- Run the full matrix through the frozen harness in `src/bench/run_matrix.py`.
#### Experiment and Measure
- Report WER/CER and confidence behaviour for every corruption, severity, and decoder combination.
- Identify the corruption/decoder pair with the worst risk-coverage behaviour.
- Repeat enough runs to estimate variance on a representative subset.
#### Required Output
['- `src/bench/run_matrix.py`', '- `results/day52_robustness_matrix.csv`', '- `results/day52_robustness_matrix.png`']
#### Completion Check
> The complete matrix is measured and the worst cell is identified, with variance estimated on a subset.

---

### DAY 53: Optimization and serving ablations
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_53.md`](days/day_53.md)

> **v3 STATUS: CORE** Fixed inputs, one variable at a time, and Pareto frontiers rather than a single winner.
#### Learn
- Pareto frontiers: when no configuration dominates.
- Holding inputs fixed so comparisons mean something.
#### Build in MendSpeech
- Run every optimization variant and serving configuration on the identical frozen subset in `src/bench/run_ablations.py`.
#### Experiment and Measure
- Plot WER against p99 latency and mark Pareto-efficient points in `results/day53_pareto.png`.
- Report serving configurations separately from model-level optimizations.
- Keep live measurements separate from any simulated estimate.
#### Required Output
['- `src/bench/run_ablations.py`', '- `results/day53_ablations.csv`', '- `results/day53_pareto.png`']
#### Completion Check
> You can say which configuration to ship and which trade-offs are unavoidable, with measured frontiers.

---

### DAY 54: Personalization comparison on the frozen harness
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_54.md`](days/day_54.md)

> **v3 STATUS: CORE** Base, fine-tuned, and RL, measured once on data frozen before any of them ran.
#### Learn
- Why the final comparison must use the frozen set, not a convenient one.
- Reporting regression as carefully as improvement.
#### Build in MendSpeech
- Run the three checkpoints through the frozen harness in `src/bench/run_personalization.py`.
#### Experiment and Measure
- Report WER, risk-coverage, and clean-speech regression for base, fine-tuned, and RL.
- Report what RL cost in GPU time against what it bought.
- State plainly whether personalization earned its place in the pipeline.
#### Required Output
['- `src/bench/run_personalization.py`', '- `results/day54_personalization_final.csv`', '- `reports/day54_personalization.md`']
#### Completion Check
> The personalization decision is made on frozen evidence, including the case where it did not pay off.

---

### DAY 55: Technical report and reproduction guide
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_55.md`](days/day_55.md)

> **v3 STATUS: CORE** Every claim points at a table. Every omission appears as a limitation.
#### Learn
- Separating observation from causal claim.
- Reporting a mixed or negative result honestly.
- Why a report nobody can reproduce is not evidence.
#### Build in MendSpeech
- Write `REPORT.md` with exact reproduction commands and environment capture.
- Write `REPRODUCE.md` and verify it from a clean checkout.
#### Experiment and Measure
- Audit every major claim against a concrete table, figure, or experiment.
- Remove or soften any conclusion not directly supported by frozen evidence.
- Add a limitations section listing every blocked or deferred capability.
#### Required Output
['- `REPORT.md`', '- `REPRODUCE.md`', '- `results/final_figures/`', '- `docs/limitations_and_claims.md`']
#### Completion Check
> A technical reader understands the contribution, the trade-offs, and the limitations without opening the source.

---

### DAY 56: Final demo, clean reproduction, and release
- **Compute:** `Modal L4 plus local interface`
- **Dedicated Daily File:** [`docs/days/day_56.md`](days/day_56.md)

> **v3 STATUS: CORE** Release gate. The demo must show measured numbers, not a scripted success path.
#### Learn
- Demonstrating a system honestly, including its failure modes.
- Releasing with a stable, reproducible artifact.
#### Build in MendSpeech
- Extend only `app/audio_lab.py` with live or prerecorded audio, partial/final transcripts, confidence, triage actions, and the measured latency budget.
- Reproduce one frozen benchmark from a fresh environment and tag a stable release.
#### Experiment and Measure
- Verify every public chart regenerates from saved result files.
- Demonstrate at least one failure case, not only the success path.
- Confirm the demo's displayed numbers match the committed result files.
#### Required Output
['- `app/audio_lab.py`', '- `REPRODUCE.md`', '- `demos/final_demo.mp4`', '- `docs/architecture.md`']
#### Completion Check
> A new user can run, evaluate, and reproduce the system, and every number shown traces to a committed artifact.

---

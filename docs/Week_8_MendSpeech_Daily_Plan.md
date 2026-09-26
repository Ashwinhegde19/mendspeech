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

## Week Map

| Day | Focus | Minimum Evidence / Artifact | Compute | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 50** | Freeze questions and baselines | Four internal comparisons plus one feasible external comparator; support/deferral and oracle/live labels frozen | `Local CPU; L4 dry run` | [Open Day 50](days/day_50.md) |
| **Day 51** | Release SpeechDamageBench v1 and freeze evaluation | Independently installable, deterministic, versioned benchmark | `Local CPU` | [Open Day 51](days/day_51.md) |
| **Day 52** | Recognition and context ablations | Measured live effects versus separately labeled simulation; no fictitious runtime gains | `Modal L4` | [Open Day 52](days/day_52.md) |
| **Day 53** | Cascaded repair and seam ablations | Fixed predicted text/spans isolate stitching effects; null/worse outcomes valid | `Modal L4` | [Open Day 53](days/day_53.md) |
| **Day 54** | Capability-scoped direct restoration comparison | Supported measured rows, unsupported/deferred conditions, and honest internal-only fallback | `Modal L4; local analysis` | [Open Day 54](days/day_54.md) |
| **Day 55** | Technical report and reproducibility guide | Claims trace to measurements; unsupported/deferred capabilities explicit | `Local CPU` | [Open Day 55](days/day_55.md) |
| **Day 56** | One final app and clean reproduction | `app/audio_lab.py`, tested abstention, reproducible charts/report, evidence-based Gate 7 | `Modal L4; local interface` | [Open Day 56](days/day_56.md) |

---

## v2 Compression Map (Evidence Gates)

| Day | v2 Status | Note |
| :--- | :--- | :--- |
| **Day 50** | CORE | Freeze research questions and baselines |
| **Day 51** | CORE | SpeechDamageBench v1 release + frozen evaluation |
| **Day 52 + Day 53** | Day 52 CORE — absorbs Day 53 MERGED | Combined frozen recognition/context and repair/seam evidence; live versus simulated labels |
| **Day 54** | CORE | One capability-scoped comparator if feasible; otherwise internal comparisons and explicit external/inpainting deferral |
| **Day 55 → Day 56** | MERGED → CORE | Technical report plus existing `app/audio_lab.py`; clean reproduction, no new app |

Core comparisons are raw damaged audio, full resynthesis, naive selective
repair, and boundary-matched selective repair. Preserve / Balanced / Rescue
remain policy settings, not new model projects. The one selected external
comparator consumes `docs/baseline_install_notes.md` from Week 2; do not
model-hunt or train a restoration fallback from scratch. No inpainting or mask
support is assumed. Deferred training and unsupported conditions stay explicit.

---

## Reference Spine
- Frozen protocol and prior results
- The one Week 2 restoration comparator's model card and capability record
- Primary papers only when needed to interpret a measured result

---

## Daily Detailed Operating Plans

### DAY 50: Freeze research questions and baselines
- **Compute:** `Local CPU for planning, Modal L4 for
dry run`
- **Dedicated Daily File:** [`docs/days/day_50.md`](days/day_50.md)

> **v2 STATUS: CORE — freeze evidence and capability limits, not a calendar.** One external restoration comparator at most; consume Week 2's bounded feasibility decision.

#### Learn
- Primary question: can selective semantic repair improve intelligibility while preserving more original speech than full resynthesis?
- Secondary question: can uncertainty guided context allocation improve the latency versus accuracy operating point?
- Architecture question: on supported conditions, how does cascaded ASR plus
  TTS compare with the one selected pretrained direct restoration comparator?
  Denoising/enhancement is not evidence of mask-aware missing-span inpainting.
- Scope every claim to the frozen benchmark scale (≥30 utterances, ≥5
  speakers, typically ~5 at this lab) and state the statistical caveat
  explicitly — do not claim population-level generalization.
- Define null outcomes, failure criteria, and claims you will not make.

#### Build in MendSpeech
- Freeze code revision, model revisions, datasets, hardware, corruption configs, and metrics.
- Freeze raw damaged audio, full resynthesis, naive selective repair, and
  boundary-matched selective repair, with predicted text as the normal path.
  Hold text/spans fixed for stitching comparisons; segregate oracle rows.
- Reuse `docs/baseline_install_notes.md` from Week 2: record selected checkpoint,
  revision/license, feasible/deferred state, supported corruptions, mask
  capability, resampling, and preservation semantics. Implement only one
  adapter, `src/baselines/direct_audio_restore.py`, if feasible; selection
  alone is not tested support. Do not assume mask input or inpainting ability.
- If unavailable, freeze the four internal comparisons above and explicitly
  defer external restoration/inpainting. No model hunting, second comparator,
  scratch-restoration fallback, or claim that the external method was tested.
- Freeze fixed/adaptive context conditions with `execution_mode=live` or
  `simulated`. Only implemented live control with same-L4 measurements can
  support runtime-gain claims; cached/oracle scheduling is not deployed speedup.
- Keep Day 49 abstention active, and record whether the TTS checkpoint is base
  or adapted plus Day 46's measured/deferred status. No required positive result.

#### Experiment and Measure
- Run a tiny dry run to verify every required field has a measurement or
  explicit status/reason. Unsupported/deferred conditions have missing metrics,
  not fabricated zeros; they are excluded from measured rankings and plots.

#### Required Output
- `experiments/capstone_protocol.md`
- `configs/capstone_frozen.yaml`
- `docs/baseline_definitions.md`

#### Completion Check
> Another engineer can reproduce the supported comparisons and distinguish
> selected from tested support, external/inpainting deferral, oracle diagnostics,
> and live versus simulated context results without inventing missing evidence.

---

### DAY 51: Release SpeechDamageBench v1 and freeze evaluation
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_51.md`](days/day_51.md)

#### Learn
- Severity grids.
- Speaker separated evaluation.
- Seed control and deterministic manifests.
- Package versioning and reproducibility.
- Clean regression cases that must remain untouched.

#### Build in MendSpeech
- Finalize the independent SpeechDamageBench package with noise, clipping, bandwidth, dropout, and reverberation presets.
- Generate the frozen test matrix and lock manifest checksums.
- Add an installation command and a one command example that reproduces one benchmark item.

#### Experiment and Measure
- Reinstall the package in a clean environment.
- Regenerate a sample from the manifest and verify its checksum.
- Validate that clean references remain unchanged.

#### Required Output
- `speechdamagebench/`
- `speechdamagebench/README.md`
- `speechdamagebench/CHANGELOG.md`
- `benchmarks/speechdamagebench_manifest.csv`
- `benchmarks/README.md`

#### Completion Check
> SpeechDamageBench is independently installable, deterministic, versioned, and
usable without MendSpeech.

---

### DAY 52: Run recognition and context ablations
- **Compute:** `Modal L4, keep hardware fixed`
- **Dedicated Daily File:** [`docs/days/day_52.md`](days/day_52.md)

> **v2 STATUS: CORE — absorbs [Day 53](days/day_53.md).** Run recognition/context and repair/seam ablations together on the frozen harness; evidence, not elapsed sessions, closes the gate.

#### Learn
- Fixed lookahead comparison.
- Adaptive context policy.
- WER, latency, RTF, memory, confidence behavior.

#### Build in MendSpeech
- Run every streaming condition on the exact same benchmark subset.
- Repeat timing runs enough to estimate variance.
- Record GPU type and environment automatically through the Modal runner.
- Record `execution_mode=live|simulated` for every context policy. A live
  adaptive claim requires runtime context changes in the recognizer, not
  cached-output selection, offline scheduling, or an oracle decision rule.
- Keep simulated-policy cost estimates separate from measured latency/RTF;
  do not count cached reuse or a hypothetical context reduction as runtime gains.

#### Experiment and Measure
- Plot WER versus measured latency and mark Pareto efficient live points.
  Label simulated analyses separately with their assumptions; do not mix
  estimates into a measured frontier. Null or worse adaptive outcomes are valid.

#### Required Output
- `results/capstone_streaming.csv`
- `results/streaming_pareto.png`

#### Completion Check
> You can say whether implemented adaptive context helped, hurt, or made no
> meaningful difference. If only simulation is available, state that limitation
> and defer live runtime claims rather than fabricate speedups.

---

### DAY 53: Run cascaded repair and seam ablations
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_53.md`](days/day_53.md)

> **v2 STATUS: MERGED into [Day 52](days/day_52.md) — single combined ablation session.** Produce all repair/seam controls and artifacts within Day 52; no standalone session.

#### Learn
- Repair threshold.
- Repair span padding.
- Preserve percentage.
- Full resynthesis baseline.
- Boundary energy matching, crossfade choice, and seam artifact rate.

#### Build in MendSpeech
- Run Preserve, Balanced, Rescue, full resynthesis, naive selective stitching, and boundary matched selective stitching.
- Record original waveform retained, repair percentage, end to end latency, speaker similarity proxy, and seam metrics.

#### Experiment and Measure
- Test whether repairing more audio always helps intelligibility.
- Test whether boundary matching reduces seam artifacts without materially increasing latency.
- Keep recognition outputs fixed for the stitching comparison so only the repair method changes.

#### Required Output
- `results/capstone_cascaded_repair.csv`
- `results/repair_tradeoff.png`
- `results/seam_ablation.png`

#### Completion Check
> You have a defensible result for the cascaded selective repair path and can separate
recognition, reconstruction, and stitching effects.

---

### DAY 54: Capability-scoped direct restoration comparison
- **Compute:** `Modal L4 for comparisons; local CPU for analysis`
- **Dedicated Daily File:** [`docs/days/day_54.md`](days/day_54.md)

> **v2 STATUS: CORE — one conditional external comparator, no model hunting.** Week 2 feasibility bounds apply; unavailable external restoration/inpainting is explicitly deferred.

#### Learn
- Why text is an information bottleneck for prosody and acoustic continuity.
- Direct audio inpainting in latent or codec token spaces at a conceptual level.
- Fair baseline design when systems have different latency and compute profiles.
- Failure taxonomy across semantic correctness, speaker similarity, prosody, seam quality, and compute.

#### Build in MendSpeech
- Consume the one selected comparator and bounded feasible/deferred decision
  in Week 2's `docs/baseline_install_notes.md`. Selection is not a claim of
  tested support. Do not search for substitutes or train restoration from scratch.
- If feasible, implement only `src/baselines/direct_audio_restore.py` behind
  the shared benchmark interface. Record checkpoint/revision/license, sample
  rate, supported damage conditions, mask support, and whether output changes
  samples outside a requested interval. Do not pass masks unless supported.
  This plan chooses one restoration adapter, not a separate inpainting
  adapter; an inpainting claim requires verified mask-aware
  missing-span reconstruction, not a suggestive filename or denoising output.
- Feed identical supported SpeechDamageBench cases, label out-of-scope
  conditions `unsupported`, and exclude them from aggregate comparisons.
  A model available only outside L4 is not an L4 efficiency comparison;
  defer it rather than silently changing hardware or extending the budget.
- If unavailable, still compare raw damaged audio, full resynthesis, naive
  selective repair, and boundary-matched selective repair. Record external
  restoration/inpainting as `deferred`, not tested; no second project/fallback.
- Reuse the tested `src/controller/abstain.py` from Day 49. Keep abstention
  active when inferred content, conditioning consent, or seam safety is weak.

#### Experiment and Measure
- Compare the four internal paths and only the supported external conditions
  on the same cases. Use predicted text for normal TTS paths; separately label
  oracle text/spans and live versus simulated context policies.
- Select at least ten worst or most revealing cases and inspect them manually.
- In `results/capstone_architecture_compare.csv`, include method/checkpoint,
  corruption, mask capability, text/span source, execution mode, condition
  status, and reason. Measured rows may show improvement, equality, or harm;
  `unsupported`/`deferred` rows have missing metrics, never invented numbers.
- Create a failure casebook and tradeoff plot from measured evidence only.
  If no supported external run exists, title the plot as internal comparisons
  and state the deferral; do not imply both architectures were evaluated.

#### Required Output
- Feasible branch only: `src/baselines/direct_audio_restore.py` (one adapter
  with capability metadata; no placeholder implementation if deferred)
- `results/capstone_architecture_compare.csv`
- `results/capstone_failure_casebook.md`
- `results/architecture_tradeoff.png`
- Reuse, do not postpone: `src/controller/abstain.py` (required by Day 49)

#### Completion Check
> Supported conditions have reproducible measured comparisons and explicit
> limitations; unsupported conditions are not fabricated. If the comparator
> is unavailable, the internal capstone plus external/inpainting deferral is
> complete, but an external or mask-aware comparison is not claimed as tested.

---

### DAY 55: Write the research report and reproducibility guide
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_55.md`](days/day_55.md)

> **v2 STATUS: MERGED into [Day 56](days/day_56.md).** Write the technical report alongside the single `app/audio_lab.py` demo; completion is evidence-based.

#### Learn
- Abstract, motivation, hypotheses, method, baselines, metrics, results, limitations, ethics, and future work.
- Difference between observation and causal claim.
- How to report a negative or mixed architectural comparison honestly.
- Benchmark scale and its statistical limits: never claim population-level generalization from a ~5-speaker lab set.

#### Build in MendSpeech
- Write the complete report.
- Add exact reproduction commands and environment capture.
- Include a dedicated internal-versus-external comparison section with the
  one comparator's selected, supported, measured, unsupported, and deferred
  conditions. If it could not run, report internal comparisons and explicit
  external/inpainting deferral, not an invented architectural result.
- Document seam limitations, prosody loss, consent, abstention, and supported
  conditions where either method is stronger, unchanged, or worse.
- State Day 46's base/adapted evidence or training deferral, gold-text/oracle
  exclusions, and live versus simulated context labels. Simulation cannot
  establish measured runtime gains; blocked training is not measured adaptation.
- Reproduce the existing `app/audio_lab.py`; do not introduce a second app.
- Include plots with captions that state what changed and what stayed fixed.

#### Experiment and Measure
- Audit every major claim against a concrete table, figure, or experiment result.
- Remove or soften any conclusion that is not directly supported by frozen evidence.
- Verify that the report distinguishes measured facts from hypotheses and future work.

#### Required Output
- `REPORT.md`
- `REPRODUCE.md`
- `results/final_figures/`
- `docs/limitations_and_claims.md`

#### Completion Check
> A technical reader can understand the contribution, the architectural tradeoff, and the
limitations without opening the source code first.

---

### DAY 56: Final product, demo, and clean reproduction
- **Compute:** `Modal L4 for inference, local CPU for
interface and analysis`
- **Dedicated Daily File:** [`docs/days/day_56.md`](days/day_56.md)

> **v2 STATUS: CORE — absorbs Day 55.** Gate 7 closes on report, artifact, and reproduction evidence, not a date or guaranteed session count.

#### Learn
- Review the complete path from waveform and controlled corruption to streaming encoder, uncertainty, repair policy, cascaded reconstruction, direct audio baseline, and evaluation.

#### Build in MendSpeech
- Extend only `app/audio_lab.py` with upload or consented microphone input,
  controlled damage, transcript, uncertainty heatmap, Preserve / Inspect /
  Repair / Abstain, before/after playback, and measured metrics. Reuse Day 49
  abstention; do not create a separate final or voice-agent app.
- Label live input/control, prerecorded benchmark playback, simulated context,
  and oracle diagnostics distinctly. Expose only supported external conditions
  in benchmark playback; show unavailable comparator/inpainting as deferred.
- Show measured changed/preserved samples, including crossfade margins. Do not
  claim a full-waveform restoration model preserved everything outside a mask.
- Reproduce one frozen benchmark from a fresh environment and tag a stable release.

#### Experiment and Measure
- Record a concise demo and create a final architecture diagram.
- Reproduce one benchmark end to end from the documented command.
- Verify that every public chart can be regenerated from saved result files.

#### Required Output
- `app/audio_lab.py`
- `README.md`
- `demos/final_demo.mp4`
- `docs/architecture.png`
- `release_notes.md`
- `results/reproduction_check.txt`

#### Completion Check
> A new user can reproduce MendSpeech and SpeechDamageBench, evaluate the
> internal baselines and any supported external comparison, and distinguish
> measured results from unsupported/deferred capabilities. The one app and
> technical report agree on abstention, consent, and live/simulated labels.

---

## Add-on C — Separate Indic & Code-Mixed Speech Evaluation

Prepare the separate verified manifest after Gate 2; measure each capability
when its pipeline is available and finish the report with Gate 7. Reuse the
single ASR adaptation path only if compatible language/data support exists;
otherwise keep this evaluation-only. Never change the frozen core benchmark.

### Session 1 — evaluation slice

- Create a consented or public manifest containing one Indian language the
  researcher can verify, Indian English, and code-mixed speech.
- Include multiple speakers, names, numerals, transliterations, and matched
  clean/noisy/dropout conditions.
- Verify checkpoint/tokenizer support and record normalization and transcript
  verification before evaluation. Use the one open pipeline without adding
  hosted backend comparisons. Unsupported slices remain explicitly incomplete.

### Session 2 — failure analysis

- Record offline WER/CER and entity errors first where feasible; add
  VAD/endpointing errors, first-partial and p50/p95 latency after streaming
  exists. Missing capabilities have reasons, not fabricated metrics.
- Inspect code-switch boundaries, named entities, normalization, accent, and
  endpointing failures manually.
- State the small-set limitation and do not make population-level claims.

### Required Output

- `data/indic_codemix_manifest.csv`
- `results/addon_c_indic_codemix.csv`
- `reports/addon_c_speech_readiness.md`

### Completion Check

> You can explain at least three multilingual or code-mixing failure modes,
> identify whether recognition, endpointing, normalization, or synthesis caused
> each one, and defend the limits of the extension set. If coverage is
> insufficient, report the missing evidence instead of inventing failure modes.

---

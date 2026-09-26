# Week 2: ASR, CTC, Confidence, and Repair Localization

> **Days 08 to 14**  
> **Navigation:** [← Week 1](Week_1_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Week 3 →](Week_3_MendSpeech_Daily_Plan.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Build the recognition and uncertainty layer, plus a reusable Modal execution path.
>
> **v2 evidence gate:** Days 08–14 remain CORE. Gate 2 requires ASR, WER/CER,
> confidence/timing, safe policy decisions and reproducible model/run metadata.
> Record the bounded external-comparator status; a documented deferral is not
> inpainting success. Then complete required Add-on A baseline/reference VAD
> measurements. No calendar deadline or timed rebuild determines completion.

---

## Week Map

| Day | Focus | Minimum Evidence / Artifact | Compute | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 08** | Frame sequence to transcript | Reproducible ASR; one external comparator recorded as feasible or deferred, not assumed masked inpainting. | `CPU smoke; L4 comparisons` | [Open Day 08](days/day_08.md) |
| **Day 09** | CTC from first principles | You can explain why a blank is needed and correctly decode repeated characters. | `Local CPU` | [Open Day 09](days/day_09.md) |
| **Day 10** | WER, CER, and error taxonomy | You can calculate WER by hand for a short example and explain each error. | `Local CPU` | [Open Day 10](days/day_10.md) |
| **Day 11** | Token confidence and uncertainty | You understand why low confidence can be useful but cannot be treated as truth. | `Modal L4 recommended` | [Open Day 11](days/day_11.md) |
| **Day 12** | Time alignment and uncertain spans | Reusable uncertainty overlay in the single app. | `Modal L4 recommended` | [Open Day 12](days/day_12.md) |
| **Day 13** | Define selective repair policy v0 | Preserve/inspect/repair/abstain, reason codes and validation-only thresholds. | `Local CPU after caching` | [Open Day 13](days/day_13.md) |
| **Day 14** | Week 2 integration and review | One app, model/policy provenance, matched controls, casebook and comparator status. | `Modal L4 recommended` | [Open Day 14](days/day_14.md) |

**Compression map:** all seven sessions remain CORE; no merge or completion
status is implied by this revision. Add-on A follows Gate 2. Prepare Add-on C's
separate verified manifest after Gate 2; finish streaming metrics later and its
report at Gate 7. Optional systems drills add no quota or release dependency.

---

## Reference Spine
- Graves et al., Connectionist Temporal Classification\nPyTorch CTC loss and TorchAudio ASR documentation\nModal documentation for environment definitions and GPU runs

---

## Daily Detailed Operating Plans

### DAY 08: Frame sequence to transcript
- **Compute:** `Modal L4 optional, CPU acceptable for
small runs`
- **Dedicated Daily File:** [`docs/days/day_08.md`](days/day_08.md)

> **v2 STATUS: CORE.** The external comparator requires a bounded feasibility
> record, not successful neural masked inpainting. This revision does not alter
> historical result evidence or assert that a new check has passed.

#### Learn
- Why acoustic frames outnumber output tokens.
- Encoder outputs, vocabulary logits, and decoding.
- CTC versus transducer versus attention decoder at a high level.

#### Build in MendSpeech
- Run a pretrained ASR model on clean and damaged SpeechDamageBench clips.
- Store transcript, token outputs if available, and timing metadata.
- Add a reusable Modal entry point so the same command can run ASR experiments on an L4 without editing deployment code each day.
- Check the single general-restoration candidate in `docs/baseline_install_notes.md`.
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

#### Required Output
- `src/asr/baseline.py`
- `infra/modal_asr.py`
- `results/day08_baseline_transcripts.csv`
- `docs/baseline_install_notes.md` (one candidate, capability/provenance record,
  setup/retry evidence, `feasible` or `deferred`; no overwritten historical results)

#### Completion Check
> You can draw the path from features to encoder states to token probabilities to text,
and launch the same baseline locally or on Modal with a documented command.
The bounded comparator check has an evidence-backed status; a documented deferral
is sufficient for this external branch, but is not successful inpainting.

---

### DAY 09: CTC from first principles
- **Compute:** `Local CPU`
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

#### Required Output
- `src/asr/ctc_decode.py`
- `tests/test_ctc_decode.py`
- `docs/ctc_explained.md`

#### Completion Check
> You can explain why a blank is needed and correctly decode repeated characters.

---

### DAY 10: WER, CER, and error taxonomy
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_10.md`](days/day_10.md)

#### Learn
- Word error rate: substitutions, deletions, insertions.
- Character error rate and when it helps.
- Why WER alone hides error severity.

#### Build in MendSpeech
- Implement or verify WER and CER calculations.
- Add an error analyzer that labels substitution, deletion, and insertion spans.

#### Experiment and Measure
- Score clean versus every SpeechDamageBench severity.
- Find which corruption type causes deletion errors fastest.

#### Required Output
- `src/metrics/wer.py`
- `results/day10_wer_by_damage.csv`
- `results/day10_error_types.csv`

#### Completion Check
> You can calculate WER by hand for a short example and explain each error.

---

### DAY 11: Token confidence and uncertainty
- **Compute:** `Modal L4 recommended`
- **Dedicated Daily File:** [`docs/days/day_11.md`](days/day_11.md)

#### Learn
- Softmax confidence and why it can be miscalibrated.
- Frame confidence versus token confidence versus word confidence.
- Entropy as an uncertainty signal.
- Confidence calibration intuition.

#### Build in MendSpeech
- Extract confidence or approximate it from model outputs.
- Create a word level confidence timeline aligned to the transcript.

#### Experiment and Measure
- Compare confidence on clean, noisy, clipped, and dropout audio.
- Find confident but wrong examples and document them.

#### Required Output
- `src/asr/confidence.py`
- `results/day11_confidence_cases.csv`
- `docs/confidence_failure_modes.md`

#### Completion Check
> You understand why low confidence can be useful but cannot be treated as truth.

---

### DAY 12: Time alignment and uncertain spans
- **Compute:** `Modal L4 recommended`
- **Dedicated Daily File:** [`docs/days/day_12.md`](days/day_12.md)

> **v2 STATUS: CORE.** Reuse the uncertainty overlay inside the single application.

#### Learn
- Frame time conversion.
- Token timestamps and word timestamps.
- Alignment boundaries around corrupted regions.

#### Build in MendSpeech
- Map low confidence tokens back to audio time spans.
- Overlay uncertain intervals on waveform and spectrogram.
- Keep `app/uncertainty_overlay.py` as a reusable visualization module imported
  by `app/audio_lab.py`, not a separately maintained runnable application.

#### Experiment and Measure
- Inject known 100 ms and 250 ms dropouts and test whether uncertainty overlaps them.

#### Required Output
- `src/asr/alignment.py`
- `app/uncertainty_overlay.py` (reusable module for `app/audio_lab.py`)
- `results/day12_overlap_metrics.csv`

#### Completion Check
> The shared UI can highlight an uncertain audio interval and show the associated
word or token without creating another application.

---

### DAY 13: Define selective repair policy v0
- **Compute:** `Local CPU after ASR outputs are
cached`
- **Dedicated Daily File:** [`docs/days/day_13.md`](days/day_13.md)

> **v2 STATUS: CORE.** Define safe action semantics now; exercise synthesis-time
> abstention by Day 49, not first at the final comparison.

#### Learn
- Threshold policies.
- Hysteresis to avoid rapid toggling.
- Minimum repair span and padding.
- False repair versus missed repair tradeoff.
- Uncertainty indicates a need for evidence, not permission to invent content.

#### Build in MendSpeech
- Keep Preserve, Balanced, and Rescue as sensitivity presets, not action labels.
  Each returns timed decisions with an action and reason code:
  - `preserve`: reliable audio remains unchanged.
  - `inspect`: flag uncertain content for review; do not synthesize it.
  - `repair`: propose a bounded edit only when content evidence, speaker-use
    permission, alignment and supported synthesis constraints are sufficient.
  - `abstain`: an unsafe or unsupported repair is refused; retain original audio
    and disclose why no reconstruction was produced.
- Missing evidence or unavailable synthesis support cannot silently become a
  repair. Week 2 tests decisions without claiming generated audio. Carry these
  semantics into `src/controller/abstain.py` and exercise them on Day 49.

#### Experiment and Measure
- Sweep thresholds on speaker-separated validation data only; log selected
  values in `configs/repair_modes.yaml` and freeze them before test scoring.
- Measure proposed repair coverage and overlap with known damage, false repair
  on clean speech, missed repair, and inspect/abstain rates. Ground-truth damage
  masks score decisions; they are not policy inputs in normal evaluation.
- Include reliable clean audio, uncertain text, missing permission, missing
  capability and invalid alignment cases; verify all non-repair actions leave
  audio unchanged. Raw confidence remains provisional until Day 41 calibration.

#### Required Output
- `src/controller/policy.py`
- `configs/repair_modes.yaml`
- `results/day13_policy_sweep.csv`

#### Completion Check
> You can explain and demonstrate preserve/inspect/repair/abstain decisions,
including refusal to synthesize unsupported content. Thresholds come only from
validation; false repairs and abstentions are visible rather than hidden.

---

### DAY 14: Week 2 integration and review
- **Compute:** `Modal L4 recommended`
- **Dedicated Daily File:** [`docs/days/day_14.md`](days/day_14.md)

> **v2 STATUS: CORE — Gate 2 advances on evidence, not a date.** Extend
> `app/audio_lab.py`; external-comparator deferral does not block the core ASR work.

#### Learn
- Review CTC, WER, confidence, timestamp alignment, and repair decisions.

#### Build in MendSpeech
- Extend `app/audio_lab.py`: damaged audio to transcript to confidence to timed
  preserve/inspect/repair/abstain proposals. Reuse the Day 12 overlay; do not
  create a separate milestone app or claim synthesis before it exists.
- Add clean JSON output for every run: source/corruption/seed, model revision,
  policy preset/version, thresholds, intervals, actions and reason codes.
- Verify the Modal wrapper records model revision, GPU type, software versions, and run id automatically.
- Carry forward `docs/baseline_install_notes.md`: one comparator's `feasible`
  or `deferred` status and verified capabilities. A blocker report is enough
  for this conditional branch; it must not be labeled masked-inpainting success.

#### Experiment and Measure
- Run at least twenty corrupted utterances with matched clean/raw-damaged
  controls and fixed validation-selected thresholds; inspect false repair,
  missed repair, inspect and abstain cases. Preserve model-version provenance.
- Report ASR WER/CER, uncertainty overlap, proposed repair coverage and clean
  false repairs; do not imply generated-audio improvement. Keep L4 comparisons
  separate from functional CPU smoke runs.

#### Required Output
- `app/audio_lab.py`
- `infra/modal_asr.py`
- `results/week2_casebook.md`
- `reports/week2_asr_uncertainty.md`

#### Completion Check
> The one app shows what the ASR heard and the exact proposed actions, with
unchanged audio for inspect/abstain. The casebook/report retain controls, model
and policy versions, errors and comparator feasibility status. Gate 2 does not
require a successful external neural restoration model.

---

## Gate 2 Add-on A — VAD and Endpointing Baseline

This required add-on budgets approximately two sessions for a deterministic
baseline and reference comparison. Completion depends on evidence, not a stopwatch.

### Session 1 — deterministic baseline

- Fix a labeled 30–50-file evaluation subset without changing the frozen core
  benchmark. Select detector thresholds on validation, not the test subset.
- Implement deterministic framing, timestamp conversion and a small explainable
  energy or spectral decision rule. Record frame/hop units and endpoint settings.
- Add tests for silence, all-speech input, short clips, frame boundaries, and
  deterministic evaluation.

### Session 2 — comparison and diagnosis

- Compare the baseline with one local reference VAD, such as WebRTC VAD, on
  identical clean/damaged inputs. No separate denoising or diarization branch.
- Measure precision, recall, F1, false alarms, missed speech, onset/offset
  boundary error in milliseconds, and CPU RTF by corruption type.
- Explain observed failure modes and carry the measured detector/endpointing
  choice into Day 35 and Add-on B. Keep CPU VAD RTF separate from GPU comparisons.

### Required Output

- `src/vad/baseline.py`
- `tests/test_vad.py`
- `results/addon_a_vad_benchmark.csv`
- `docs/addon_a_notes.md`

### Completion Check

> Framing/timestamp tests pass, baseline/reference measurements are reproducible,
> and onset/offset errors and detection tradeoffs are explained. No timed rebuild
> or optional drill is required.

---

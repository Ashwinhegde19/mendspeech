# MendSpeech Documentation Index

Session specifications and engineering rules for the MendSpeech project.

> **Generated navigation.** Day numbers are specification identifiers, not
> calendar promises. Status banners override day bodies; optional drills are
> not release gates. Regenerate with `python scripts/plan_docs.py --write`.

---

## Controlling documents

- [Revised Execution Plan](REVISED_EXECUTION_PLAN.md) — phases, gates, scope, and compute rules.
- [Project Blueprint](MendSpeech_Project_Blueprint.md) — architecture, metrics, and definition of done.
- [Latency and Quality Contract](LATENCY_AND_QUALITY_CONTRACT.md) — timing boundaries, workloads, and fair comparison rules.
- [Editor and RL Contract](EDITOR_AND_RL_CONTRACT.md) — conservative editing limits, data, model route, and reward design.
- [Plan manifest](plan_manifest.json) — machine-readable statuses, prerequisites, and effort ranges.
- [Optional systems drills](SPEECH_ML_SYSTEMS_DRILLS.md) — study material, not release gates.

---

## 8-week / 56-specification progression

| Week | Focus | Milestone | Weekly Plan | Daily Files |
| :--- | :--- | :--- | :--- | :--- |
| **Week 1** | Audio, Degradation, & Measurement Foundations | Build the audio laboratory and release `SpeechDamageBench` as a standalone package. | [Week 1 Guide](Week_1_MendSpeech_Daily_Plan.md) | [Day 01](days/day_01.md) • [Day 02](days/day_02.md) • [Day 03](days/day_03.md) • [Day 04](days/day_04.md) • [Day 05](days/day_05.md) • [Day 06](days/day_06.md) • [Day 07](days/day_07.md) |
| **Week 2** | Recognition, Editor Contract, & End-to-End Baseline | Scoring, data roles, confidence, the conservative editor contract, decoder comparison, and the first ASR-to-editor baseline. | [Week 2 Guide](Week_2_MendSpeech_Daily_Plan.md) | [Day 08](days/day_08.md) • [Day 09](days/day_09.md) • [Day 10](days/day_10.md) • [Day 11](days/day_11.md) • [Day 12](days/day_12.md) • [Day 13](days/day_13.md) • [Day 14](days/day_14.md) |
| **Week 3** | Streaming Capability, Session Loop, & Endpointing | Verify streaming and cache support, build the chunk loop, and measure endpointing. | [Week 3 Guide](Week_3_MendSpeech_Daily_Plan.md) | [Day 15](days/day_15.md) • [Day 16](days/day_16.md) • [Day 17](days/day_17.md) • [Day 18](days/day_18.md) • [Day 19](days/day_19.md) • [Day 20](days/day_20.md) • [Day 21](days/day_21.md) |
| **Week 4** | Context, Calibration, Triage, & Latency Baseline | Fixed-lookahead trade-off, fitted calibration, triage policy, and a full pipeline latency baseline. | [Week 4 Guide](Week_4_MendSpeech_Daily_Plan.md) | [Day 22](days/day_22.md) • [Day 23](days/day_23.md) • [Day 24](days/day_24.md) • [Day 25](days/day_25.md) • [Day 26](days/day_26.md) • [Day 27](days/day_27.md) • [Day 28](days/day_28.md) |
| **Week 5** | Inference Optimization | Profile first, then compile/graph capture, batching, precision parity, cache-failure evidence, and a scorecard. | [Week 5 Guide](Week_5_MendSpeech_Daily_Plan.md) | [Day 29](days/day_29.md) • [Day 30](days/day_30.md) • [Day 31](days/day_31.md) • [Day 32](days/day_32.md) • [Day 33](days/day_33.md) • [Day 34](days/day_34.md) • [Day 35](days/day_35.md) |
| **Week 6** | Editor Post-Training & ASR Robustness | Reward design, SFT and compute-matched control, bounded GRPO, and acoustic robustness adaptation. | [Week 6 Guide](Week_6_MendSpeech_Daily_Plan.md) | [Day 36](days/day_36.md) • [Day 37](days/day_37.md) • [Day 38](days/day_38.md) • [Day 39](days/day_39.md) • [Day 40](days/day_40.md) • [Day 41](days/day_41.md) • [Day 42](days/day_42.md) |
| **Week 7** | Serving, Load, & Correlated Latency Budget | One endpoint, load to saturation, editor selection under load, and the per-stage budget. | [Week 7 Guide](Week_7_MendSpeech_Daily_Plan.md) | [Day 43](days/day_43.md) • [Day 44](days/day_44.md) • [Day 45](days/day_45.md) • [Day 46](days/day_46.md) • [Day 47](days/day_47.md) • [Day 48](days/day_48.md) • [Day 49](days/day_49.md) |
| **Week 8** | Frozen Evaluation, Report, & Release | Freeze the protocol, run the matrix and ablations, and publish the report and reproduction guide. | [Week 8 Guide](Week_8_MendSpeech_Daily_Plan.md) | [Day 50](days/day_50.md) • [Day 51](days/day_51.md) • [Day 52](days/day_52.md) • [Day 53](days/day_53.md) • [Day 54](days/day_54.md) • [Day 55](days/day_55.md) • [Day 56](days/day_56.md) |

---

## Session totals

- **P1** (days 01–09): 9 core sessions, 0–0 focused hours — Audio lab, deterministic damage suite, frozen labeled benchmark, CTC and ASR baseline.
- **P2** (days 10–16): 7 core sessions, 19–31 focused hours — Data roles and protocol, confidence, timestamps, conservative editor contract, decoder comparison, early ASR-to-editor baseline, and SFT/GRPO feasibility.
- **P3** (days 18–26): 6 core sessions, 15–27 focused hours — Streaming capability gate, chunk/session loop, endpointing, fixed-context frontier, calibration and triage, and an early end-to-end pipeline with a latency baseline.
- **P4** (days 27–33): 7 core sessions, 15–27 focused hours — Profiling, compile/graph capture, batching, precision parity, cache-failure evidence, scorecard, and pipeline revalidation.
- **P5** (days 34–42): 9 core sessions, 23–39 focused hours — Editor reward, SFT and compute-matched control, GRPO run, failure casebook, and a personalization feasibility decision.
- **P6** (days 43–49): 7 core sessions, 14–26 focused hours — Serving contract, async service, load to saturation, editor selection under load, correlated latency budget, one optimization round, and a progress review.
- **P7** (days 50–55): 6 core sessions, 10–17 focused hours — Frozen protocol, evaluation freeze, robustness matrix, ablations, final condition comparison, technical report, and reproduction guide.
- **P8** (days 56–56): 1 core sessions, 2–3 focused hours — Final demo, clean reproduction, and tagged release.

Total core sessions: 52 (days 01–09 complete). Remaining effort: 98–170 focused hours, including days 01–09. Estimate the remaining days from observed throughput.

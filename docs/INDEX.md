# MendSpeech Documentation Index

Product-oriented documentation for MendSpeech.

## Controlling documents

- [Project blueprint](MendSpeech_Project_Blueprint.md) — architecture, metrics, and definition of done
- [Revised execution plan](REVISED_EXECUTION_PLAN.md) — scope, phases, and release gates
- [Latency and quality contract](LATENCY_AND_QUALITY_CONTRACT.md) — timing boundaries, workloads, and fair comparisons
- [Editor and RL contract](EDITOR_AND_RL_CONTRACT.md) — editing limits, data, model route, and reward design
- [Metric limitations](metric_limitations.md) — known limitations of the evaluated metrics
- [CTC from first principles](ctc_explained.md) — technical reference for `src/asr/ctc_decode.py`

## Evaluation documentation

- [Hinglish evaluation policy](hinglish_evaluation_policy.md) — speaker roles, held-out reserve rule, and frozen-pool immutability
- [Data roles and leakage audit](../reports/data_roles.md) — frozen benchmark roles and leakage checks
- [Evaluation protocol](../experiments/protocol.md) — metrics, data rules, failure thresholds, and reproducibility

## Preserved learning material

Earlier learning session specs, 56-day plans, notebooks, and day-specific
measurements were removed from this product branch. They remain available on
the `research/learning` branch.
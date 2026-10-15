# Results Index

This repository is product-oriented. Current runtime code lives under `src/` and
`speechdamagebench/`; experiment scripts are under `scripts/`.

Historical learning artifacts and day-by-day measurements are preserved on the
`research/learning` branch. New product experiments should add a functional
filename and one row to this index in the same change.

## Reusable experiment entry points

- `scripts/run_day08_experiment.py` — clean versus corrupted ASR baseline
- `scripts/run_day10_experiment.py` — WER/CER and protected-content slices on the frozen benchmark
- `scripts/run_day11_experiment.py` — confidence versus correctness analysis

## Naming rules

- Reusable source and test modules use functional names.
- Result files should be named for their evaluation scope, not for a day number.
- Audio, checkpoints, and run logs remain gitignored; only manifests, tables, figures, and notes are tracked.
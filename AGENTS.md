# Engineering guidelines

Rules for everyone changing this repository: human contributors and automated
assistants alike. Where these rules conflict with a tool's defaults, these
rules win.

---

## 1. Project

MendSpeech is a voice-AI product focused on meaning-preserving dictation:
damage-robust streaming ASR, bounded transcript editing, calibrated confidence,
and correlated latency measurements.

Two products matter:

1. **MendSpeech** — the end-to-end system (`src/`).
2. **SpeechDamageBench** — a standalone, versioned damage-generation
   package that must remain usable without MendSpeech.

Architecture, metrics, and the definition of done live in
[`docs/MendSpeech_Project_Blueprint.md`](docs/MendSpeech_Project_Blueprint.md).
Scope, phases, and gates live in
[`docs/REVISED_EXECUTION_PLAN.md`](docs/REVISED_EXECUTION_PLAN.md).

Work is tracked through focused branches, issues, and pull requests. Legacy
learning plans, day specifications, and notebooks are preserved on
`research/learning` and must not be mixed into product changes.

---

## 2. Critical rules

1. **This repository is public.** Personal notes stay out of tracked
   files. `PRIVATE_PLAN_NOTES.md` and `PRIVATE_LEARNING_WITH_AGENTS.md`
   are gitignored — never stage them, never commit them, never remove
   their ignore entries. Review `git diff --cached` before every commit.
2. **Never fake completion.** A change is complete only when relevant tests or
   manual checks pass and the claimed output exists. If work is incomplete,
   say so.
3. **Never force-push or rewrite pushed history on `main` or `research/learning`.**
4. **Do not expand scope unilaterally.** Propose changes before applying them.

---

## 3. Layout

```text
AGENTS.md                 # These guidelines
README.md                 # Public project face
pyproject.toml            # Package: mendspeech (Python >=3.10)
src/                      # Product subsystems
  audio/                  # Waveform I/O, STFT, log-Mel
  asr/                    # CTC decoding and confidence helpers
  metrics/                # WER/CER, normalization, data roles, slices
speechdamagebench/        # Standalone damage-generation package
tests/                    # pytest suite
data/                     # Manifests only; audio is gitignored
docs/                     # Architecture, contracts, evaluation docs
infra/                    # Cloud execution helpers
app/                      # Gradio comparison console
scripts/                  # Reproducible experiment entry points
results/README.md         # Index for committed measured artifacts
pdfs/                     # Archived originals — never modify
```

---

## 4. Environment

- Python **>= 3.10**. Install: `pip install -e .`
- Use the project venv: `.venv/bin/python -m pytest`.
- Tests: `pytest` from repository root.
- Imports: `from src.audio.loader import load_audio`.
- Cloud measurements use Modal L4 unless stated otherwise.

---

## 5. Product development workflow

1. Create `feat/*`, `fix/*`, `chore/*`, or `release/*` branches from `main`.
2. Keep each pull request focused on one logical change.
3. Run `pytest` before committing or opening a pull request.
4. Update documentation links when removing tracked Markdown files.
5. Preserve all frozen benchmark data and leakage checks.
6. Follow conventional commits.

---

## 6. Coding standards

Match `src/audio/`:

- Type hints on public functions; use `typing` forms (`Optional`, `Tuple`,
  `Union`, `Dict`) as currently used.
- Module docstrings plus Google-style `Args:` / `Returns:` on public
  functions. State units (Hz, dB, seconds).
- Dataclasses for structured records (`AudioMetadata`, `LogMelMetadata`).
- **Determinism is a product feature.** Damage and corruption code takes an
  explicit seed; generated artifacts record corruption, severity, seed,
  source ID, parameters, and package version. No unseeded randomness.
- Tests in `tests/test_*.py`, written in the same change as the public
  function. Add shape and gradient assertions for tensor code.
- Keep NumPy <-> torch conversions explicit. Prefer torch at module
  boundaries; use NumPy inside measurement utilities.
- Notebooks are for exploration. Reusable logic belongs in `src/` with tests.

---

## 7. Data and artifacts

- Never commit audio (`*.wav`, `*.mp3`, `*.flac`, `*.ogg`), checkpoints
  (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.nemo`), or run logs.
- Commit manifests (`data/*.csv`, `data/*.json`), metric tables, figures,
  notes, code, and tests.
- Every committed result gets a row in `results/README.md` in the same
  change.
- Once frozen, the evaluation set is immutable. New experiments use new
  corruption configurations, not a new test set.

---

## 8. Git

- Conventional Commits: `type(scope): imperative subject`.
- Subject <= 72 characters, lowercase, no trailing period. Body wrapped at
  72 characters: what and why.
- One logical change per commit. Do not mix unrelated plan and feature code.
- Never force-push, rebase pushed commits, or amend pushed history.

---

## 9. Compute

- Honor documented hardware constraints for latency, RTF, and memory work.
- Default cloud tier for ASR measurements: Modal L4.
- L40S is not valid for latency, RTF, or memory comparisons.
- Cache ASR outputs. Budget is roughly $15 to $30 for the whole project.

---

## 10. Boundaries

- **Touch:** `src/`, `tests/`, `notebooks/`, `results/`, `data/` manifests,
  `docs/`, and `AGENTS.md` when rules change.
- **Never touch:** `pdfs/`, private note files, their `.gitignore` entries,
  the frozen evaluation set, or previously frozen benchmark results.
- **When uncertain:** follow the repository contracts and flag the question.
  The smallest correct change beats a broad refactor.
- Partial work is committed as partial, with an honest commit message.
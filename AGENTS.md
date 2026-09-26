# Engineering guidelines

Rules for everyone changing this repository: human
contributors and automated assistants alike. Where these rules conflict
with a tool's defaults, these rules win.

---

## 1. Project

**MendSpeech** targets meaning-preserving dictation: damage-robust streaming
ASR, calibrated uncertainty, conservative transcript editing, and correlated
latency measurements. Acoustic fine-tuning is robustness adaptation; bounded
SFT/RL applies to the text editor, not an unspecified CTC policy update.
These are target capabilities, not claims of completed implementation.

Two products matter:

1. **MendSpeech** — the end-to-end system (`src/`).
2. **SpeechDamageBench** — a standalone, versioned damage-generation
   package that must remain usable without MendSpeech.

Architecture, metrics, and the definition of done live in
[`docs/MendSpeech_Project_Blueprint.md`](docs/MendSpeech_Project_Blueprint.md).
Pacing and gates live in
[`docs/REVISED_EXECUTION_PLAN.md`](docs/REVISED_EXECUTION_PLAN.md).
On any calendar or scope conflict, the execution plan wins.

Work is scoped to one session spec at a time (`docs/days/day_NN.md`).
`docs/plan_manifest.json` records statuses, prerequisites and effort ranges.
After changing these or day specs, run `scripts/plan_docs.py --write`, then
`scripts/plan_docs.py --check` and the documentation tests. Generated week,
compiled, and navigation views must not be edited independently.
Do not redesign the build order, add experiments, or change scope unless
the maintainer asks.

---

## 2. Critical rules

1. **This repository is public.** Personal notes stay out of tracked
   files. `PRIVATE_PLAN_NOTES.md` and `PRIVATE_LEARNING_WITH_AGENTS.md`
   are gitignored — never stage them, never commit them, never remove
   their ignore entries. Review `git diff --cached` before every commit.
2. **Never fake completion.** A session is done only when its spec's
   Completion Check passes and its Required Output artifacts exist. If
   work is incomplete, say so. Do not mark it finished to protect a date.
3. **Never force-push or rewrite pushed history on `main`.**
4. **Do not redesign the plan unilaterally.** Propose scope changes;
   do not apply them.

---

## 3. Layout

```text
AGENTS.md                 # These guidelines
README.md                 # Public project face
pyproject.toml            # Package: mendspeech (Python >=3.10)
src/                      # One package per subsystem
  audio/                  # EXISTS: waveform I/O, STFT, log-Mel
  asr/ streaming/ controller/ vad/ llm/ rl/ serve/ metrics/ bench/
                          # created when that subsystem is reached
tests/                    # pytest suite (test_*.py), mirrors src
notebooks/                # Exploration and figures only
results/                  # Measured CSVs, PNGs, tables
data/                     # Manifests only; audio is gitignored
docs/                     # Blueprint, execution plan, session specs
pdfs/                     # Archived originals — never modify
```

---

## 4. Environment

- Python **>= 3.10**. Install: `pip install -e .`
- Use the project venv: `.venv/bin/python -m pytest` (system Python
  does not have the dependencies).
- Tests: `pytest` from repo root (`pythonpath = [".", "speechdamagebench"]`,
  `testpaths = ["tests", "speechdamagebench/tests"]`). Documentation tests
  inspect public paths only, never private notes or dependency caches.
- Imports: `from src.audio.loader import load_audio`.
- Week 1 is local CPU. Modal L4 is allowed from the first ASR session
  and is required for any latency, RTF, or memory comparison.

---

## 5. Session specs

When implementing from `docs/days/day_NN.md`:

1. Read the status banner under the title. It overrides the file body:
   - `CORE` — full session: learn, build, measure, artifacts.
   - `LEARN-ONLY` — theory only; no build, no artifacts, no code commit.
   - `MERGED` — work happens inside the absorbing session; do not run it
     standalone.
   - `DROPPED` — not scheduled.
2. Do not start the build until the Learn content is understood well
   enough to defend. Print one concrete example after coding (shape,
   units, seed, path). “The files exist” is not the Completion Check.
3. Build exactly what the spec asks — minimum viable, then measure.
   SpeechDamageBench is a nested package with its **own**
   `pyproject.toml`. Never overwrite the repo-root `mendspeech`
   `pyproject.toml`.
4. Write artifacts at the exact paths the spec names. Paths are
   contracts.
5. Honor the Completion Check honestly.
6. Commit that session's code and artifacts (see §8).

If a session's status changes, update the day-file banner, the week
guide's compression map, and `docs/REVISED_EXECUTION_PLAN.md` in the
same commit.

---

## 6. Coding standards

Match `src/audio/`:

- Type hints on public functions; `typing` forms (`Optional`, `Tuple`,
  `Union`, `Dict`) as currently used.
- Module docstring plus Google-style `Args:` / `Returns:` on public
  functions. State units (Hz, dB, seconds).
- Dataclasses for structured records (`AudioMetadata`, `LogMelMetadata`).
- **Determinism is a product feature.** Every damage, corruption, or
  augmentation takes an explicit seed. Every generated artifact records
  corruption name, severity, seed, source ID, parameters, and package
  version. No unseeded randomness.
- Tests in `tests/test_*.py`, written in the same session as the public
  function. Shape and gradient assertions for tensor code.
- Keep numpy ↔ torch conversions explicit. Prefer torch at module
  boundaries; numpy inside measurement utilities.
- Notebooks are for figures. Reusable logic lives in `src/` with tests.

---

## 7. Data and artifacts

- Never commit audio (`*.wav` / `mp3` / `flac` / `ogg`), checkpoints
  (`*.pt` / `*.pth` / `*.ckpt` / `*.onnx` / `*.nemo`), or wandb/run logs.
- Do commit manifests (`data/*.csv|json`), metric tables, figures,
  notes, and code/tests.
- Result files are named `dayNN_<what>.<ext>` when a session spec
  names that path.
- Every committed result gets a row in `results/README.md` in the same
  session (day → artifact → one-line finding).
- Once frozen, the evaluation set (≥30 transcripted utterances, ≥5
  speakers, speaker-separated splits) is immutable. New experiments get
  new corruption configs, not a new test set.

---

## 8. Git

- Conventional Commits: `type(scope): imperative subject`. Types:
  `feat`, `fix`, `docs`, `test`, `chore`, `refactor`. Scopes follow
  the subsystem (`audio`, `asr`, `bench`, `plan`).
- Subject ≤ 72 chars, lowercase, no trailing period. Body wrapped ≤ 72
  chars: what and why.
- One logical change per commit. Do not mix plan edits with feature
  code.
- `pytest` green before committing code changes.
- Never force-push, rebase pushed commits, or amend pushed history.

---

## 9. Compute

- Honor the `Compute Target` on the session spec. Default cloud tier:
  **Modal L4**.
- **L40S is banned** for latency, RTF, or memory comparisons. Measured
  efficiency numbers come from L4 only.
- Prefer the cheapest tier that fits. Cache ASR outputs. Budget is
  roughly $15–30 for the whole project.
- A hardware or batch-size change invalidates a comparison. If you must
  switch, say so in the results file. Do not mix tiers silently.

---

## 10. Boundaries

- **Touch:** `src/`, `tests/`, `notebooks/`, `results/`, `data/`
  manifests, and — when explicitly asked — planning docs in `docs/`.
- **Never touch:** `pdfs/`, `PRIVATE_PLAN_NOTES.md`,
  `PRIVATE_LEARNING_WITH_AGENTS.md`, `.gitignore`'s private-notes
  entries, the frozen evaluation set, or a `results/` file from a
  previously frozen benchmark run.
- **When uncertain:** follow the session spec literally and flag the
  question. Smallest correct change beats a clever refactor.
- Never mark a session complete when it is not. Partial work is
  committed as partial, with an honest message.

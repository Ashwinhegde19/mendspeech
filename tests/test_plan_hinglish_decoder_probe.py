"""Offline tests for the Phase 7.1 decoder-probe plan freeze builder.

Every fixture is synthetic: invented clip IDs, dummy WAV bytes, and
synthetic failure intelligence are built in ``tmp_path``; the tests never
touch the real private archive except for one clearly guarded integration
test that runs the builder's dry run against the real frozen artifacts and
skips cleanly when they are absent.

The pre-registered mandate (rules R1-R4, counts 8 ceiling-word-loop /
3 ceiling-char-loop / 2 early-loop / 3 control) is exercised in code exactly
as it is enforced against the real 30-clip development cohort, so both the
synthetic happy path and the real-data integration test share the same
verification logic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "plan_hinglish_decoder_probe.py"

PINNED_CREATED_UTC = "2026-10-18T00:00:00Z"

# Pre-registered rule R3 mandates these exact early-loop IDs; the synthetic
# world therefore reuses them (they are identifiers, never content).
EARLY_LOOP_IDS = (
    "304558_c8Ba68BmVqNaBudE_0000",
    "399287_1TuGgwSGEBVVydVA_0000",
)
CONTROL_PREFIXES = ("111627_", "136706_", "415128_")

WORD_LOOP_INSERTIONS = [132, 74, 73, 64, 62, 60, 58, 56, 56, 43, 42, 40]
WORD_LOOP_RUNTIMES = [3.695, 3.714, 3.687, 3.734, 3.706, 3.721, 3.733, 3.727, 3.699, 3.712, 3.697, 3.709]
WORD_LOOP_DURATIONS = [8.55, 6.55, 3.55, 6.55, 6.55, 5.55, 5.55, 7.55, 4.55, 6.55, 3.55, 6.55]

EXPECTED_TOP8_PREFIXES = ["cwl00", "cwl01", "cwl02", "cwl03", "cwl04", "cwl05", "cwl06", "cwl07"]


def _load_builder(module_scope="plan_hinglish_decoder_probe_script"):
    """Import the freeze-builder script in-process (fast, echo of repo style)."""
    spec = importlib.util.spec_from_file_location(module_scope, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the 3.13 dataclass machinery resolves InitVar
    # types through ``sys.modules[module].__dict__`` at class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> Any:
    """Loaded freeze-builder module shared by all tests in this file."""
    return _load_builder()


class SynthWorld:
    """Paths and expected values for one synthetic fixture world."""

    def __init__(
        self,
        root: Path,
        *,
        word_loop_count: int = 12,
        early_runtimes: Tuple[float, float] = (1.054, 1.2),
        control_runtimes: Tuple[float, float, float] = (1.562, 0.667, 0.17),
        early_classes: Tuple[str, str] = ("repetitive_generation_early_stop",) * 2,
        reserve_leaks_selected_speaker: bool = False,
    ) -> None:
        self.root = root
        self.word_loop_count = word_loop_count
        self.early_runtime_overrides = list(early_runtimes)
        self.control_runtime_overrides = list(control_runtimes)
        self.early_class_overrides = list(early_classes)
        self.audio_dir = root / "audio"
        self.manifest = root / "frozen_manifest.csv"
        self.reserve = root / "reserve.json"
        self.raw_output = root / "raw_modal_output.json"
        self.intelligence = root / "intelligence.json"
        self.private_dir = root / "freeze"
        self.plan_out = root / "results" / "hinglish" / "decoder_probe_plan.json"

        self.word_loop_ids = [
            f"cwl{index:02d}_zzsynth00_0000" for index in range(word_loop_count)
        ]
        self.char_loop_ids = [
            f"cch{index:02d}_zzsynth00_0000" for index in range(3)
        ]
        self.early_ids = list(EARLY_LOOP_IDS)
        self.control_ids = [prefix + "zzsynth00_0000" for prefix in CONTROL_PREFIXES]
        self.all_ids = (
            self.word_loop_ids + self.char_loop_ids + self.early_ids + self.control_ids
        )

        self.word_loop_insertions = WORD_LOOP_INSERTIONS[:word_loop_count]
        self.word_loop_runtimes = WORD_LOOP_RUNTIMES[:word_loop_count]
        self.word_loop_durations = WORD_LOOP_DURATIONS[:word_loop_count]

        self._write_audio()
        self._write_manifest(reserve_leaks_selected_speaker)
        self._write_reserve()
        self._write_raw()
        self._write_intelligence()

    # ------------------------------------------------------------------ audio
    def _audio_bytes(self, clip_id: str, index: int) -> bytes:
        """Distinct dummy WAV-shaped bytes per clip (never real audio)."""
        return b"RIFF" + str(index).rjust(4, "0").encode() + clip_id.encode() * 3

    def _write_audio(self) -> None:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        for index, clip_id in enumerate(self.all_ids):
            (self.audio_dir / f"{clip_id}.wav").write_bytes(self._audio_bytes(clip_id, index))

    # --------------------------------------------------------------- manifest
    def _write_manifest(self, reserve_leaks: bool) -> None:
        speakers = []
        for index, clip_id in enumerate(self.all_ids):
            if reserve_leaks and index == 0:
                speakers.append("spk_reserve")
            else:
                speakers.append(f"spk{index:02d}")
        lines = ["clip_id,speaker_id"] + [
            f"{clip_id},{speaker}" for clip_id, speaker in zip(self.all_ids, speakers)
        ]
        self.manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.manifest_speakers = dict(zip(self.all_ids, speakers))

    # ---------------------------------------------------------------- reserve
    def _write_reserve(self) -> None:
        payload = {
            "pool_version": "test.reserve.v0",
            "speakers": [
                {"speaker_id": "spk_reserve"},
                {"speaker_id": "spk_reserve2"},
            ],
        }
        self.reserve.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------- raw output
    def _write_raw(self) -> None:
        records: List[Dict[str, Any]] = []
        for clip_id in self.all_ids:
            if clip_id.startswith("cwl"):
                index = self.word_loop_ids.index(clip_id)
                runtime = self.word_loop_runtimes[index]
                duration = self.word_loop_durations[index]
            elif clip_id.startswith("cch"):
                slot = int(clip_id[3:5])
                runtime = [3.72, 3.73, 3.70][slot]
                duration = [7.55, 1.55, 6.55][slot]
            elif clip_id in self.early_ids:
                slot = self.early_ids.index(clip_id)
                runtime = self.early_runtime_overrides[slot]
                duration = 6.55 if clip_id.startswith("304558_") else 7.55
            else:
                slot = self.control_ids.index(clip_id)
                runtime = self.control_runtime_overrides[slot]
                duration = [7.275, 5.55, 1.55][slot]
            records.append(
                {
                    "clip_id": clip_id,
                    "runtime_sec": runtime,
                    "audio_duration_sec": duration,
                    "transcript": "synthetic",
                }
            )
        text = "Launching synthetic...\n" + json.dumps(records, indent=1)
        self.raw_output.write_text(text, encoding="utf-8")
        self.expected_runtimes = {record["clip_id"]: record["runtime_sec"] for record in records}
        self.expected_durations = {
            record["clip_id"]: record["audio_duration_sec"] for record in records
        }

    # ------------------------------------------------------------ intelligence
    def _write_intelligence(self) -> None:
        records: List[Dict[str, Any]] = []
        for index, clip_id in enumerate(self.word_loop_ids):
            records.append(
                {
                    "clip_id": clip_id,
                    "speaker_id": self.manifest_speakers[clip_id],
                    "failure_class": "generation_cap_with_repetition",
                    "insertions": self.word_loop_insertions[index],
                    "has_word_loop": True,
                    "has_char_loop": False,
                    "heuristic_flag": True,
                    "cap_hit": True,
                }
            )
        # One ceiling clip carries both loop kinds; the word-loop rule wins.
        # (Only on full-cohort worlds: "cwl10" exists when word_loop_count >= 11.)
        if self.word_loop_count >= 11:
            double_loop = next(
                r for r in records if r["clip_id"].startswith("cwl10")
            )
            double_loop["has_char_loop"] = True
        for clip_id in self.char_loop_ids:
            records.append(
                {
                    "clip_id": clip_id,
                    "speaker_id": self.manifest_speakers[clip_id],
                    "failure_class": "generation_cap_with_repetition",
                    "insertions": 2 - int(clip_id[3:5]),
                    "has_word_loop": False,
                    "has_char_loop": True,
                    "heuristic_flag": False,
                    "cap_hit": True,
                }
            )
        for offset, clip_id in enumerate(self.early_ids):
            records.append(
                {
                    "clip_id": clip_id,
                    "speaker_id": self.manifest_speakers[clip_id],
                    "failure_class": self.early_class_overrides[offset],
                    "insertions": [16, 9][offset],
                    "has_word_loop": True,
                    "has_char_loop": False,
                    "heuristic_flag": [True, False][offset],
                    "cap_hit": False,
                }
            )
        control_classes = ("script_mismatch_dominant", "script_mismatch_dominant", "recognition_difficulty")
        control_insertions = [0, 0, 1]
        for offset, clip_id in enumerate(self.control_ids):
            records.append(
                {
                    "clip_id": clip_id,
                    "speaker_id": self.manifest_speakers[clip_id],
                    "failure_class": control_classes[offset],
                    "insertions": control_insertions[offset],
                    "has_word_loop": False,
                    "has_char_loop": False,
                    "heuristic_flag": False,
                    "cap_hit": False,
                }
            )
        payload = {
            "report_version": "test.intelligence.v0",
            "per_clip": records,
        }
        self.intelligence.write_text(
            json.dumps(payload, indent=1), encoding="utf-8"
        )

    # ---------------------------------------------------------------- expected
    def expected_selection(self) -> List[Tuple[str, str]]:
        """The pre-registered derivation for this world: (clip_id, category)."""
        top8 = [f"{prefix}_zzsynth00_0000" for prefix in EXPECTED_TOP8_PREFIXES]
        ordered = [(clip_id, "ceiling_word_loop") for clip_id in top8]
        ordered += [(clip_id, "ceiling_char_loop") for clip_id in self.char_loop_ids]
        ordered += [(clip_id, "early_loop") for clip_id in sorted(self.early_ids)]
        ordered += [(clip_id, "control") for clip_id in self.control_ids]
        return ordered

    def cli_base(self) -> List[str]:
        """CLI arguments pointing every input at this world."""
        return [
            f"--intelligence-json={self.intelligence}",
            f"--audio-dir={self.audio_dir}",
            f"--manifest={self.manifest}",
            f"--reserve-json={self.reserve}",
            f"--raw-output-json={self.raw_output}",
            f"--private-dir={self.private_dir}",
            f"--plan-out={self.plan_out}",
            f"--report-out={self.root / 'plan_report.json'}",
        ]


def _run_main(builder: Any, argv: Sequence[str]) -> Tuple[int, str, str]:
    """Invoke main() in-process and return (exit code, stdout, stderr)."""
    import io
    import contextlib

    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = builder.main(list(argv))
    return code, stdout.getvalue(), stderr.getvalue()


@pytest.fixture()
def _build_world(tmp_path: Path) -> Any:
    def factory(name: str = "world", **kwargs: Any) -> SynthWorld:
        return SynthWorld(tmp_path / name, **kwargs)

    return factory


# ---------------------------------------------------------------------------
# Happy path: rule compliance on synthetic fixtures with the REAL mandate.
# ---------------------------------------------------------------------------
def test_unit_selection_enforces_mandate(builder: Any, _build_world: Any) -> None:
    """R1-R3 derivation selects exactly the mandated 16 clips in order."""
    world = _build_world("unit")
    intel, intelligence_sha = builder.load_intelligence(world.intelligence)
    manifest, manifest_sha = builder.load_manifest(world.manifest)
    raw, _raw_sha = builder.parse_raw_output(world.raw_output)
    reserve, _reserve_sha = builder.load_reserve_speakers(world.reserve)

    features, counts = builder.derive_categories(intel, raw, manifest)
    assert counts == {
        "ceiling_word_loop": 12,
        "ceiling_char_loop": 3,
        "early_loop": 2,
        "control": 3,
    }
    selected = builder.build_selection(features, counts)
    builder.validate_membership(selected, manifest, reserve)
    selected = builder.attach_audio_hashes(selected, world.audio_dir)

    expected = world.expected_selection()
    assert [(c.clip_id, c.category) for c in selected] == expected
    assert len(selected) == 16

    # Tie-break check: cwl07 and cwl08 both carry 56 insertions; the smaller
    # clip ID (cwl07) fills the eighth slot and cwl08 is excluded.
    assert selected[7].clip_id.startswith("cwl07")
    assert selected[7].insertions == 56

    # Word-loop rule precedence over char-loop on the double-loop clip:
    # cwl10 carries both loop kinds; the cohort counts prove it was counted
    # as a word-loop ceiling clip (12, with cwl10 not in the top-8).
    double = features["cwl10_zzsynth00_0000"]
    assert double.has_word_loop and double.has_char_loop
    assert counts["ceiling_word_loop"] == 12

    # R1 agreement covered every cohort clip.
    assert len(features) == 20

    # Placeholder hash fields are replaced by real digests.
    for clip in selected:
        assert clip.audio_sha256 == hashlib.sha256(
            (world.audio_dir / f"{clip.clip_id}.wav").read_bytes()
        ).hexdigest()

    del intelligence_sha, manifest_sha  # provenance hashing covered elsewhere


def test_cli_dry_run_prints_and_writes_nothing(
    builder: Any, _build_world: Any, capsys: Any
) -> None:
    """Default mode is a dry run: full derivation printed, zero writes."""
    world = _build_world("dry")
    code, out, err = _run_main(builder, world.cli_base())
    assert (code, err) == (0, "")
    assert "mode: dry-run (nothing written)" in out
    for clip_id, _category in world.expected_selection():
        assert clip_id in out
    assert "config inference seconds: C0=" in out
    assert "wall = " in out and "estimated_gpu_cost = " in out
    assert "$0.06888" in out
    assert not world.plan_out.exists()
    assert not world.private_dir.exists()


def test_cli_dry_run_in_real_defaults_when_artifacts_present(
    builder: Any, capsys: Any
) -> None:
    """Integration (skipped cleanly when private artifacts are absent).

    Dry run only - the test must never write into the repository. It pins
    the pre-registered 16-clip frozen set and the committed estimate
    numbers against the real frozen cohort (30 clips, R1 agreement, R4
    membership, and the archived audio hashes).
    """
    defaults_present = all(
        path.exists()
        for path in (
            builder.DEFAULT_INTELLIGENCE_JSON,
            builder.DEFAULT_AUDIO_DIR,
            builder.DEFAULT_MANIFEST,
            builder.DEFAULT_RESERVE_JSON,
            builder.DEFAULT_RAW_OUTPUT_JSON,
        )
    )
    if not defaults_present:
        pytest.skip(
            "real frozen cohort artifacts (private archive + committed "
            "intelligence) are not present in this environment"
        )
    code, out, err = _run_main(builder, [])
    assert (code, err) == (0, "")
    assert "R1 ceiling agreement: 30/30 clips agree" in out
    assert "mode: dry-run (nothing written)" in out

    intel, _ = builder.load_intelligence(builder.DEFAULT_INTELLIGENCE_JSON)
    manifest, _ = builder.load_manifest(builder.DEFAULT_MANIFEST)
    raw, _ = builder.parse_raw_output(builder.DEFAULT_RAW_OUTPUT_JSON)
    reserve, _ = builder.load_reserve_speakers(builder.DEFAULT_RESERVE_JSON)
    assert len(manifest) == 30  # the real freeze mandates the 30-clip cohort
    features, counts = builder.derive_categories(intel, raw, manifest)
    selected = builder.build_selection(features, counts)
    builder.validate_membership(selected, manifest, reserve)
    selected = builder.attach_audio_hashes(selected, builder.DEFAULT_AUDIO_DIR)
    estimate = builder.build_estimate(selected)

    expected_real = {
        "420389_9vdn8rISVcWlbojw_0000": "ceiling_word_loop",
        "154712_txbpWebPgY4aS0sN_0000": "ceiling_word_loop",
        "296876_8NVl0ViOXGYOiW60_0000": "ceiling_word_loop",
        "193949_kNy6anmVvg5h7xmA_0000": "ceiling_word_loop",
        "166257_RtAxi9ScmHHUqbCx_0000": "ceiling_word_loop",
        "350956_AdztwNlAtCku4h6b_0000": "ceiling_word_loop",
        "344267_aZSahmWizlD17OIp_0000": "ceiling_word_loop",
        "208711_z98W1r9OPyYYWZsZ_0000": "ceiling_word_loop",
        "180696_GSUP9H9fbPzTvUJf_0000": "ceiling_char_loop",
        "246662_fLxNwT4sz4MQwqGE_0000": "ceiling_char_loop",
        "396461_myiAlAOYzbquNUxR_0000": "ceiling_char_loop",
        "304558_c8Ba68BmVqNaBudE_0000": "early_loop",
        "399287_1TuGgwSGEBVVydVA_0000": "early_loop",
        "111627_bYPRSx2sjXooUdv8_0000": "control",
        "136706_PMLT41UVHAjHIiJW_0000": "control",
        "415128_riwQQ9Di2QY1QXwx_0000": "control",
    }
    assert [(c.clip_id, c.category) for c in selected] == list(expected_real.items())
    assert estimate["config_inference_seconds"]["C0"] == 45.477
    assert estimate["config_inference_seconds"]["C1"] == 26.653
    assert estimate["config_inference_seconds"]["C2"] == 60.0
    assert estimate["total_wall_sec"] == 282.13
    assert estimate["estimated_gpu_cost_usd"] == 0.046269
    assert estimate["timeout_worst_case_usd"] == 0.06888
    # Archived-hash spot check against the archive's own checksum ledger.
    spot = hashlib.sha256(
        (builder.DEFAULT_AUDIO_DIR / "111627_bYPRSx2sjXooUdv8_0000.wav").read_bytes()
    ).hexdigest()
    assert spot == "5d3be137b5cf315d07aca8a0892d78bc0d6625b18dd81f0ec736b403ca1804bb"
    ledger = json.loads(
        (builder.DEFAULT_AUDIO_DIR.parent / "audio_checksums.json").read_text()
    )
    assert ledger["111627_bYPRSx2sjXooUdv8_0000"]["sha256"] == spot


# ---------------------------------------------------------------------------
# Freeze determinism, overwrite refusal, self-check.
# ---------------------------------------------------------------------------
def test_freeze_is_deterministic_and_selfchecked(
    builder: Any, _build_world: Any, tmp_path: Path
) -> None:
    """Two pinned freezes produce identical bytes (modulo created_utc)."""
    world_a = _build_world("det_a")
    world_b = _build_world("det_b")
    pin = f"--created-utc={PINNED_CREATED_UTC}"

    code_a, out_a, err_a = _run_main(builder, world_a.cli_base() + ["--freeze", pin])
    code_b, out_b, err_b = _run_main(builder, world_b.cli_base() + ["--freeze", pin])
    assert (code_a, code_b) == (0, 0)
    assert not err_a and not err_b
    assert "FROZEN BEFORE EXECUTION" in out_a

    plan_a = world_a.plan_out.read_bytes()
    plan_b = world_b.plan_out.read_bytes()
    assert plan_a == plan_b  # committed content is absolute-path-free and pinned

    private_a = json.loads((world_a.private_dir / "decoder_probe_plan.json").read_text())
    private_b = json.loads((world_b.private_dir / "decoder_probe_plan.json").read_text())
    for payload in (private_a, private_b):
        payload["private_sources"] = {"normalized": "for-test"}
    assert private_a == private_b

    # A different pinned timestamp changes exactly the created_utc field.
    world_c = _build_world("det_c")
    other_pin = "2026-10-19T00:00:00Z"
    _run_main(builder, world_c.cli_base() + ["--freeze", f"--created-utc={other_pin}"])
    normalized_a = json.loads(plan_a.decode("utf-8"))
    normalized_c = json.loads((world_c.plan_out).read_text(encoding="utf-8"))
    assert normalized_a["created_utc"] == PINNED_CREATED_UTC
    assert normalized_c["created_utc"] == other_pin
    normalized_a["created_utc"] = normalized_c["created_utc"] = None
    assert normalized_a == normalized_c

    # Committed content carries no absolute/private paths.
    assert b"/home/" not in plan_a
    assert b"backups/" not in plan_a

    # Private freeze additionally carries private-only source pointers.
    committed = json.loads(plan_a.decode("utf-8"))
    assert set(private_a) - set(committed) == {"private_sources"}

    # SHA256SUMS.txt verifies every written file byte for byte.
    sums = (world_a.private_dir / "SHA256SUMS.txt").read_text().splitlines()
    assert sums
    for line in sums:
        sha, path_text = line.split("  ", 1)
        path = Path(path_text)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_freeze_refuses_overwrite_without_force_but_force_rewrites(
    builder: Any, _build_world: Any, tmp_path: Path
) -> None:
    """Second freeze refuses without --force; --force rewrites cleanly."""
    world = _build_world("ow")
    pin = f"--created-utc={PINNED_CREATED_UTC}"
    code_ok, _, err_ok = _run_main(builder, world.cli_base() + ["--freeze", pin])
    assert (code_ok, err_ok) == (0, "")
    original = world.plan_out.read_bytes()

    code_refus, _, err_refus = _run_main(builder, world.cli_base() + ["--freeze", pin])
    assert code_refus == 4
    assert "refusing to overwrite" in err_refus
    assert world.plan_out.read_bytes() == original

    code_force, _, err_force = _run_main(
        builder, world.cli_base() + ["--freeze", pin, "--force"]
    )
    assert (code_force, err_force) == (0, "")
    assert world.plan_out.read_bytes() == original  # same content again


def test_freeze_report_written_and_hash_recorded(builder: Any, _build_world: Any) -> None:
    """The machine-readable report records outputs and estimate numbers."""
    world = _build_world("rep")
    code, _, err = _run_main(
        builder, world.cli_base() + ["--freeze", f"--created-utc={PINNED_CREATED_UTC}"]
    )
    assert (code, err) == (0, "")
    report_path = world.root / "plan_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "freeze"
    assert report["created_utc"] == PINNED_CREATED_UTC
    assert report["mandate"]["category_counts"] == {
        "ceiling_word_loop": 8,
        "ceiling_char_loop": 3,
        "early_loop": 2,
        "control": 3,
    }
    assert len(report["clips"]) == 16
    assert set(report["sources"]) == {
        "manifest_sha256",
        "reserve_sha256",
        "intelligence_json_sha256",
        "raw_output_sha256",
        "archive_note",
    }
    assert report["outputs"]["force"] is False

    committed = json.loads(world.plan_out.read_text(encoding="utf-8"))
    assert committed["provenance"]["manifest_sha256"] == hashlib.sha256(
        world.manifest.read_bytes()
    ).hexdigest()
    assert committed["provenance"]["reserve_sha256"] == hashlib.sha256(
        world.reserve.read_bytes()
    ).hexdigest()
    assert committed["provenance"]["intelligence_json_sha256"] == hashlib.sha256(
        world.intelligence.read_bytes()
    ).hexdigest()
    assert committed["provenance"]["raw_output_sha256"] == hashlib.sha256(
        world.raw_output.read_bytes()
    ).hexdigest()
    assert report["sources"]["manifest_sha256"] == committed["provenance"]["manifest_sha256"]


# ---------------------------------------------------------------------------
# Safety: the committed plan can never carry private or transcript content.
# ---------------------------------------------------------------------------
def test_committed_plan_safe_content(builder: Any, _build_world: Any) -> None:
    """Key sets, no Devanagari, no private paths in the committed plan."""
    world = _build_world("safe")
    code, _, _ = _run_main(
        builder, world.cli_base() + ["--freeze", f"--created-utc={PINNED_CREATED_UTC}"]
    )
    assert code == 0
    committed = json.loads(world.plan_out.read_text(encoding="utf-8"))
    assert set(committed) == set(builder.ALLOWED_TOP_LEVEL_KEYS)
    for clip in committed["clips"]:
        assert set(clip) == set(builder.ALLOWED_CLIP_KEYS)
    assert committed["probe_version"] == "mendspeech.hinglish_decoder_probe.v1"
    assert committed["model"] == {
        "id": "openai/whisper-small",
        "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
    }
    assert committed["configs"]["C0"] == {
        "purpose": "baseline decoding + instrumentation",
        "generate_kwargs": {"task": "transcribe", "language": "hi"},
    }
    assert committed["configs"]["C1"] == {
        "purpose": "single-parameter delta max_new_tokens=160",
        "generate_kwargs_overrides": {"max_new_tokens": 160},
    }
    assert committed["configs"]["C2"] == {
        "purpose": "single-parameter delta no_repeat_ngram_size=4",
        "generate_kwargs_overrides": {"no_repeat_ngram_size": 4},
    }
    assert committed["estimate"]["config_inference_seconds"] == {
        "C0": 45.52,
        "C1": 26.653,
        "C2": 60.0,
    }
    # Devanagari or transcript-ish strings are refused by the safety scan.
    with pytest.raises(ValueError):
        builder.assert_committed_plan_safe({**committed, "clips": [{"probe": "\u0915"}]})
    with pytest.raises(ValueError):
        builder.assert_committed_plan_safe(
            {**committed, "archive_note": "/home/ashwin/backups/private.csv"}
        )


def test_selfcheck_detects_divergent_freeze(builder: Any, _build_world: Any) -> None:
    """Mutating the private copy fails the parse-back self-check."""
    world = _build_world("selfchk")
    code, _, _ = _run_main(
        builder, world.cli_base() + ["--freeze", f"--created-utc={PINNED_CREATED_UTC}"]
    )
    assert code == 0
    private_path = world.private_dir / "decoder_probe_plan.json"
    tampered = json.loads(private_path.read_text(encoding="utf-8"))
    tampered["estimate"]["total_wall_sec"] = 99999.0
    private_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
    with pytest.raises(builder.SelfCheckError):
        builder.verify_freeze_selfcheck(
            world.plan_out,
            private_path,
            world.private_dir / "SHA256SUMS.txt",
        )


# ---------------------------------------------------------------------------
# Gates and mandate contradiction paths.
# ---------------------------------------------------------------------------
def test_cost_gate_failure_long_runtimes_exit_2(
    builder: Any, _build_world: Any, capsys: Any
) -> None:
    """Inflated non-ceiling runtimes blow the wall gate; nothing is written."""
    world = _build_world(
        "gate",
        early_runtimes=(100.0, 150.0),
        control_runtimes=(300.0, 250.0, 400.0),
    )
    code, out, err = _run_main(builder, world.cli_base())
    assert code == 2
    assert "GATE FAILED" in err
    assert "exceeds" in err
    assert not world.plan_out.exists()
    assert not world.private_dir.exists()
    del out


def test_gate_checks_raise_on_numbers(builder: Any) -> None:
    """build_estimate raises GateError beyond either hard gate."""
    clips = [
        builder.SelectedClip(
            clip_id="c0",
            category="control",
            baseline_runtime_sec=400.0,
            insertions=0,
            has_word_loop=False,
            has_char_loop=False,
            heuristic_flag=False,
            duration_sec=1.0,
            audio_sha256="0" * 64,
        )
    ]
    with pytest.raises(builder.GateError):
        builder.build_estimate(clips)


def test_mandate_contradiction_missing_word_loop_clips(
    builder: Any, _build_world: Any, capsys: Any
) -> None:
    """Fewer than 8 word-loop ceiling clips contradicts the mandate (exit 3)."""
    world = _build_world("few_cwl", word_loop_count=7)
    code, _, err = _run_main(builder, world.cli_base())
    assert code == 3
    assert "MANDATE CONTRADICTION" in err
    assert "ceiling_word_loop" in err


def test_mandate_contradiction_r1_disagreement(
    builder: Any, _build_world: Any, capsys: Any
) -> None:
    """Runtime-window and failure-class disagreement stops with exit 3."""
    world = _build_world(
        "r1bad", early_classes=("generation_cap_with_repetition",) * 2
    )
    code, _, err = _run_main(builder, world.cli_base())
    assert code == 3
    assert "R1 ceiling agreement failed" in err


def test_mandate_contradiction_reserve_member(
    builder: Any, _build_world: Any, capsys: Any
) -> None:
    """A selected speaker present in the holdout reserve violates R4 (exit 3)."""
    world = _build_world("leak", reserve_leaks_selected_speaker=True)
    code, _, err = _run_main(builder, world.cli_base())
    assert code == 3
    assert "holdout reserve" in err
    del capsys


def test_missing_selected_audio_is_input_error(
    builder: Any, _build_world: Any, capsys: Any
) -> None:
    """A frozen-set clip without a WAV file stops as an input error."""
    world = _build_world("earlydrop")
    (world.audio_dir / f"{world.early_ids[0]}.wav").unlink()
    code, _, err = _run_main(builder, world.cli_base())
    assert code == 4
    assert "missing selected audio file" in err
    del capsys


# ---------------------------------------------------------------------------
# Estimator arithmetic.
# ---------------------------------------------------------------------------
def test_clip_config_seconds_clamps(builder: Any) -> None:
    """Per-clip clamps: C0 exact, C1 <= 2.0, C2 >= 3.75."""
    assert builder.clip_config_seconds(0.5) == (0.5, 0.5, 3.75)
    assert builder.clip_config_seconds(2.0) == (2.0, 2.0, 3.75)
    assert builder.clip_config_seconds(2.5) == (2.5, 2.0, 3.75)
    assert builder.clip_config_seconds(3.75) == (3.75, 2.0, 3.75)
    assert builder.clip_config_seconds(4.0) == (4.0, 2.0, 4.0)
    c0, c1, c2 = builder.clip_config_seconds(1.0)
    assert (c0, c1, c2) == (1.0, 1.0, 3.75)


def test_build_estimate_arithmetic(builder: Any) -> None:
    """Totals add constants; cost scales by the pinned T4 rate."""
    clips = [
        builder.SelectedClip(
            clip_id=f"clip{index}",
            category="control",
            baseline_runtime_sec=measured,
            insertions=0,
            has_word_loop=False,
            has_char_loop=False,
            heuristic_flag=False,
            duration_sec=10.0,
            audio_sha256="0" * 64,
        )
        for index, measured in enumerate((1.0, 4.0))
    ]
    estimate = builder.build_estimate(clips)
    assert estimate["config_inference_seconds"] == {"C0": 5.0, "C1": 3.0, "C2": 7.75}
    assert estimate["decoded_wall_sec"] == 15.75
    assert estimate["model_load_sec"] == 90.0
    assert estimate["container_overhead_sec"] == 60.0
    assert estimate["total_wall_sec"] == 165.75
    assert estimate["estimated_gpu_cost_usd"] == 0.027183
    assert estimate["timeout_worst_case_usd"] == 0.06888
    assert estimate["clips_total_audio_sec"] == 20.0
    assert estimate["per_clip_config_seconds"]["clip0"] == {
        "C0": 1.0,
        "C1": 1.0,
        "C2": 3.75,
    }


def test_fixture_estimate_matches_formulas(builder: Any, _build_world: Any) -> None:
    """The frozen synthetic estimate reproduces the clamped sums exactly."""
    world = _build_world("est")
    intel, _ = builder.load_intelligence(world.intelligence)
    manifest, _ = builder.load_manifest(world.manifest)
    raw, _ = builder.parse_raw_output(world.raw_output)
    reserve, _ = builder.load_reserve_speakers(world.reserve)
    features, counts = builder.derive_categories(intel, raw, manifest)
    selected = builder.attach_audio_hashes(builder.build_selection(features, counts), world.audio_dir)
    assert builder.validate_membership(selected, manifest, reserve) is None
    estimate = builder.build_estimate(selected)
    assert len(selected) == 16
    # The estimate sums only the 16 selected clips' archived runtimes.
    selected_runtimes = [world.expected_runtimes[clip.clip_id] for clip in selected]
    assert estimate["config_inference_seconds"]["C0"] == pytest.approx(
        sum(selected_runtimes), abs=1e-6
    )
    assert estimate["config_inference_seconds"]["C1"] == pytest.approx(
        sum(min(m, 2.0) for m in selected_runtimes), abs=1e-6
    )
    assert estimate["config_inference_seconds"]["C2"] == pytest.approx(
        sum(max(m, 3.75) for m in selected_runtimes), abs=1e-6
    )
    assert estimate["total_wall_sec"] == pytest.approx(
        estimate["decoded_wall_sec"] + 90.0 + 60.0, abs=1e-6
    )
    assert estimate["estimated_gpu_cost_usd"] == pytest.approx(
        round(estimate["total_wall_sec"] * 0.000164, 6), abs=2e-6
    )
    assert not (world.plan_out.exists() or world.private_dir.exists())
    del intel, manifest, raw, reserve, counts


# ---------------------------------------------------------------------------
# Subprocess smoke test of the CLI entry point.
# ---------------------------------------------------------------------------
def test_cli_subprocess_dry_run(_build_world: Any) -> None:
    """python scripts/plan_hinglish_decoder_probe.py --dry-run stays offline."""
    world = _build_world("sub")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *world.cli_base(), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "mode: dry-run (nothing written)" in result.stdout
    assert not world.plan_out.exists()

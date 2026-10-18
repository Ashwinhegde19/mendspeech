"""Tests for the held-out MUCS Hinglish reserve pool freeze script.

Fixtures are fully synthetic. They contain no real MUCS transcripts and no
real speaker or clip identifiers, and they exercise the same code paths as the
private phase-6.x artifacts. CLI end-to-end behavior is covered with a small
number of subprocess runs; validation logic is covered with direct calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reserve_hinglish_holdout_pool import (  # noqa: E402
    POOL_VERSION,
    _compute_state,
    _load_inventory,
    _load_speaker_split,
    _select_pool,
    _validate_state,
    _write_pool,
)

SCRIPT = REPO_ROOT / "scripts" / "reserve_hinglish_holdout_pool.py"

# Synthetic marker so tests can prove transcript values never reach outputs.
FIXTURE_TRANSCRIPT_MARKER = "synthetic-reference-words-marker"

DEFAULT_TRAINING = tuple(f"t{index:03d}" for index in range(1, 7))
DEFAULT_VALIDATION = ("v001", "v002", "v003", "v004", "v005")
DEFAULT_SELECTED = ("v001", "v002", "v003")

EXPECTED = {
    "expected_rows": 33,
    "expected_total_speakers": 11,
    "expected_training_speakers": 6,
    "expected_validation_speakers": 5,
    "expected_selected_speakers": 3,
    "expected_remaining_speakers": 2,
}

EXPECTED_ARGS = [
    "--expected-rows", "33",
    "--expected-total-speakers", "11",
    "--expected-training-speakers", "6",
    "--expected-validation-speakers", "5",
    "--expected-selected-speakers", "3",
    "--expected-remaining-speakers", "2",
]


def _row(
    clip_id: str,
    speaker_id: str,
    english_ratio: float,
    *,
    total_words: int = 10,
) -> dict[str, str]:
    english_count = round(english_ratio * total_words)
    return {
        "segment_id": clip_id,
        "speaker_id": speaker_id,
        "transcript": f"{FIXTURE_TRANSCRIPT_MARKER} {clip_id}",
        "ratio_english_words": str(english_ratio),
        "ratio_hindi_words": str(1.0 - english_ratio),
        "count_english_words": str(english_count),
        "count_dev_words": str(total_words - english_count),
        "segment_part": clip_id.rsplit("_", 1)[-1],
        "shard": "0",
    }


def _fixture_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for speaker in DEFAULT_TRAINING:
        for part in range(3):
            rows.append(_row(f"{speaker}_seg{part:04d}", speaker, 0.2))
    for speaker in ("v001", "v002", "v003"):
        for part in range(3):
            rows.append(_row(f"{speaker}_seg{part:04d}", speaker, 0.2))
    for part in range(3):
        rows.append(_row(f"v004_seg{part:04d}", "v004", 0.5))
    # v005's smallest clip id has invalid metadata and must be skipped.
    rows.append(_row("v005_seg0000", "v005", 1.5))
    rows.append(_row("v005_seg0001", "v005", 0.2))
    rows.append(_row("v005_seg0002", "v005", 0.2))
    return rows


def _write_inventory(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "segment_id",
        "speaker_id",
        "transcript",
        "ratio_english_words",
        "ratio_hindi_words",
        "count_english_words",
        "count_dev_words",
        "segment_part",
        "shard",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_split(
    path: Path,
    *,
    training: tuple[str, ...] = DEFAULT_TRAINING,
    validation: tuple[str, ...] = DEFAULT_VALIDATION,
    selected: tuple[str, ...] = DEFAULT_SELECTED,
    diagnostics: tuple[str, ...] = ("t006",),
) -> None:
    payload = {
        "seed": 42,
        "training_speakers": list(training),
        "validation_speakers": list(validation),
        "selected_validation_speakers": list(selected),
        "diagnostic_speakers_reserved_for_training": list(diagnostics),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_fixture(
    tmp_path: Path,
    *,
    training: tuple[str, ...] = DEFAULT_TRAINING,
    validation: tuple[str, ...] = DEFAULT_VALIDATION,
    selected: tuple[str, ...] = DEFAULT_SELECTED,
    diagnostics: tuple[str, ...] = ("t006",),
) -> tuple[Path, Path]:
    inventory = tmp_path / "inventory.csv"
    split = tmp_path / "speaker_split.json"
    _write_inventory(inventory, _fixture_rows())
    _write_split(
        split,
        training=training,
        validation=validation,
        selected=selected,
        diagnostics=diagnostics,
    )
    return inventory, split


def _fixture_state(tmp_path: Path, **overrides: object) -> dict[str, object]:
    inventory, split_path = _write_fixture(tmp_path, **overrides)
    rows = _load_inventory(inventory)
    split = _load_speaker_split(split_path)
    return _compute_state(rows, split)


def _run(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, arguments)],
        capture_output=True,
        text=True,
    )


def _base_args(inventory: Path, split: Path) -> list[str]:
    return [
        "--inventory", str(inventory),
        "--speaker-split", str(split),
        *EXPECTED_ARGS,
    ]


def test_dry_run_prints_summary_and_writes_nothing(tmp_path: Path) -> None:
    inventory, split = _write_fixture(tmp_path)
    output = tmp_path / "pool.json"
    checksum = tmp_path / "pool.sha256"

    completed = _run(
        *_base_args(inventory, split),
        "--dry-run",
        "--output", output,
        "--checksum", checksum,
    )

    assert completed.returncode == 0, completed.stderr
    assert "dry-run" in completed.stdout
    assert "v004" in completed.stdout
    assert "v005" in completed.stdout
    assert "mixed_code" in completed.stdout
    assert "hindi_dominant" in completed.stdout
    assert "selected=v005_seg0001" in completed.stdout
    assert not output.exists()
    assert not checksum.exists()


def test_freeze_writes_pool_and_checksum(tmp_path: Path) -> None:
    inventory, split = _write_fixture(tmp_path)
    output = tmp_path / "pool.json"

    completed = _run(
        *_base_args(inventory, split),
        "--freeze",
        "--output", output,
    )
    assert completed.returncode == 0, completed.stderr

    checksum = tmp_path / "pool.sha256"  # default sibling path
    assert checksum.exists()
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert checksum.read_text(encoding="utf-8") == f"{digest}  pool.json\n"

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["pool_version"] == POOL_VERSION
    assert payload["source_partition"] == "train"
    assert payload["counts"] == {
        "inventory_rows": 33,
        "total_speakers": 11,
        "training_speakers": 6,
        "validation_pool_speakers": 5,
        "selected_development_speakers": 3,
        "held_out_candidate_speakers": 2,
    }
    assert payload["bucket_distribution"] == {"hindi_dominant": 1, "mixed_code": 1}
    assert payload["dropped_ineligible_speakers"] == []
    assert payload["inventory_sha256"] == hashlib.sha256(
        inventory.read_bytes()
    ).hexdigest()
    assert payload["speaker_split_sha256"] == hashlib.sha256(
        split.read_bytes()
    ).hexdigest()

    speakers = payload["speakers"]
    assert [entry["speaker_id"] for entry in speakers] == ["v004", "v005"]
    assert len({entry["selected_clip_id"] for entry in speakers}) == 2
    by_speaker = {entry["speaker_id"]: entry for entry in speakers}
    assert by_speaker["v004"]["selected_clip_id"] == "v004_seg0000"
    assert by_speaker["v004"]["stratification_bucket"] == "mixed_code"
    assert by_speaker["v005"]["selected_clip_id"] == "v005_seg0001"
    assert by_speaker["v005"]["invalid_metadata_clip_ids"] == ["v005_seg0000"]
    assert by_speaker["v005"]["eligible_clip_count"] == 2
    assert by_speaker["v005"]["total_clip_count"] == 3

    raw = output.read_text(encoding="utf-8")
    assert '"transcript"' not in raw
    assert FIXTURE_TRANSCRIPT_MARKER not in raw


def test_freeze_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    inventory, split = _write_fixture(tmp_path)
    output = tmp_path / "pool.json"
    original = b"sentinel pool content\n"
    output.write_bytes(original)

    refused = _run(*_base_args(inventory, split), "--freeze", "--output", output)
    assert refused.returncode == 1
    assert "refusing to overwrite" in refused.stderr
    assert output.read_bytes() == original

    forced = _run(
        *_base_args(inventory, split),
        "--freeze",
        "--output", output,
        "--force",
    )
    assert forced.returncode == 0, forced.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["pool_version"] == POOL_VERSION


def test_freeze_requires_output_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.reserve_hinglish_holdout_pool import main

    inventory, split = _write_fixture(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--inventory", str(inventory),
                "--speaker-split", str(split),
                "--freeze",
                *EXPECTED_ARGS,
            ]
        )
    assert excinfo.value.code == 2
    assert "--output is required" in capsys.readouterr().err


def test_output_inside_repository_is_refused(tmp_path: Path) -> None:
    from scripts.reserve_hinglish_holdout_pool import main

    inventory, split = _write_fixture(tmp_path)
    in_repo = REPO_ROOT / "holdout_pool_should_not_be_written.json"
    assert not in_repo.exists()

    with pytest.raises(SystemExit, match="inside the repository"):
        main(
            [
                "--inventory", str(inventory),
                "--speaker-split", str(split),
                "--freeze",
                "--output", str(in_repo),
                *EXPECTED_ARGS,
            ]
        )
    assert not in_repo.exists()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("expected_rows", 34, "inventory row mismatch"),
        ("expected_total_speakers", 12, "inventory speaker mismatch"),
        ("expected_training_speakers", 7, "training speaker count mismatch"),
        ("expected_validation_speakers", 6, "validation speaker count mismatch"),
        ("expected_selected_speakers", 4, "selected development speaker count mismatch"),
        ("expected_remaining_speakers", 3, "held-out candidate speaker count mismatch"),
    ],
)
def test_count_mismatches_fail(
    tmp_path: Path, key: str, value: int, message: str
) -> None:
    state = _fixture_state(tmp_path)
    expectations = dict(EXPECTED)
    expectations[key] = value
    with pytest.raises(SystemExit, match=message):
        _validate_state(state, **expectations)


def test_training_validation_overlap_fails(tmp_path: Path) -> None:
    state = _fixture_state(
        tmp_path,
        training=DEFAULT_TRAINING + ("v004",),
    )
    with pytest.raises(SystemExit, match="overlap"):
        _validate_state(state, **EXPECTED)


def test_selected_speaker_outside_validation_pool_fails(tmp_path: Path) -> None:
    state = _fixture_state(
        tmp_path,
        selected=DEFAULT_SELECTED + ("t001",),
    )
    expectations = dict(EXPECTED)
    expectations["expected_selected_speakers"] = 4
    with pytest.raises(SystemExit, match="not a subset of the validation pool"):
        _validate_state(state, **expectations)


def test_diagnostic_speaker_outside_training_fails(tmp_path: Path) -> None:
    state = _fixture_state(tmp_path, diagnostics=("v004",))
    with pytest.raises(SystemExit, match="not reserved for training"):
        _validate_state(state, **EXPECTED)


def test_write_pool_refuses_existing_files_without_force(tmp_path: Path) -> None:
    payload = {"pool_version": POOL_VERSION}
    output = tmp_path / "pool.json"
    checksum = tmp_path / "pool.sha256"

    checksum.write_text("existing checksum\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        _write_pool(payload, output=output, checksum_path=checksum, force=False)
    assert not output.exists()
    assert checksum.read_text(encoding="utf-8") == "existing checksum\n"

    output.write_text("existing pool\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        _write_pool(payload, output=output, checksum_path=checksum, force=False)

    digest = _write_pool(payload, output=output, checksum_path=checksum, force=True)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert checksum.read_text(encoding="utf-8") == f"{digest}  pool.json\n"


def test_select_pool_is_deterministic_and_skips_invalid_metadata() -> None:
    rows = _fixture_rows()
    first, dropped = _select_pool(rows, ["v005", "v004"])
    reversed_rows, dropped_reversed = _select_pool(list(reversed(rows)), ["v005", "v004"])
    assert dropped == dropped_reversed == []
    assert first == reversed_rows
    by_speaker = {entry["speaker_id"]: entry for entry in first}
    assert by_speaker["v005"]["selected_clip_id"] == "v005_seg0001"
    assert by_speaker["v005"]["invalid_metadata_clip_ids"] == ["v005_seg0000"]
    assert by_speaker["v004"]["selected_clip_id"] == "v004_seg0000"


def test_speaker_without_eligible_clips_is_dropped() -> None:
    rows = [_row("d001_seg0000", "d001", 1.5)]
    entries, dropped = _select_pool(rows, ["d001"])
    assert entries == []
    assert dropped == ["d001"]

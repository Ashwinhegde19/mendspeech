"""Tests for deterministic MUCS validation cohort selection."""

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

from scripts.create_mucs_validation_split import (
    DIAGNOSTIC_SPEAKERS_DEFAULT,
    _bucket,
    _load_inventory,
    _sha256,
    _speaker_buckets,
    _validate_inventory,
)
from src.hinglish.data import MucsHinglishRecord


def _write_inventory(path: Path, rows: list[dict[str, str]], shards: list[int]) -> None:
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
        writer.writerows(
            {**row, "shard": str(shards[index % len(shards)])}
            for index, row in enumerate(rows)
        )


def _rows(speakers: list[str], per_speaker: int = 2) -> list[dict[str, str]]:
    rows = []
    for speaker_index, speaker in enumerate(speakers):
        remainder = speaker_index % 5
        if remainder == 0:
            english_ratio, hindi_ratio = "0.8", "0.2"
        elif remainder in (1, 2):
            english_ratio, hindi_ratio = "0.5", "0.5"
        else:
            english_ratio, hindi_ratio = "0.2", "0.8"
        for part in range(per_speaker):
            rows.append(
                {
                    "segment_id": f"{speaker}_seg{part:04d}",
                    "speaker_id": speaker,
                    "transcript": f"speaker {speaker} sample {part}",
                    "ratio_english_words": english_ratio,
                    "ratio_hindi_words": hindi_ratio,
                    "count_english_words": "2" if part % 2 == 0 else "8",
                    "count_dev_words": "10",
                    "segment_part": f"{part:04d}",
                }
            )
    return rows


def _record(
    clip_id: str,
    speaker_id: str,
    *,
    english_ratio: float,
    transcript: str = "sample text",
    metadata_valid: bool = True,
    english_word_count: int = 2,
    hindi_word_count: int = 8,
) -> MucsHinglishRecord:
    return MucsHinglishRecord(
        clip_id=clip_id,
        speaker_id=speaker_id,
        split="train",
        transcript=transcript,
        ratio_english_words=english_ratio,
        ratio_hindi_words=1.0 - english_ratio,
        english_words="",
        count_english_words=english_word_count,
        count_hindi_words=hindi_word_count,
        segment_part="0000",
        metadata_valid=metadata_valid,
        invalid_fields=() if metadata_valid else ("ratio_english_words",),
    )


def test_bucket_boundaries() -> None:
    assert _bucket(0.0) == "hindi_dominant"
    assert _bucket(0.4) == "hindi_dominant"
    assert _bucket(0.5) == "mixed_code"
    assert _bucket(0.6) == "english_dominant"
    assert _bucket(1.0) == "english_dominant"


def test_speaker_buckets_ignores_invalid_metadata() -> None:
    """Invalid ratio rows must not shift a speaker's average bucket."""
    records = [
        _record("a_0000", "s1", english_ratio=0.8),
        _record("a_0001", "s1", english_ratio=0.0, metadata_valid=False),
        _record("b_0000", "s2", english_ratio=0.5),
    ]
    buckets = _speaker_buckets(records)
    assert buckets == {"s1": "english_dominant", "s2": "mixed_code"}


def test_speaker_buckets_uses_per_speaker_average() -> None:
    """The stratification key is the speaker's average English ratio."""
    records = [
        _record("a_0000", "s1", english_ratio=0.3),
        _record("a_0001", "s1", english_ratio=0.9),
    ]
    # Average = 0.6, so the speaker lands in english_dominant.
    assert _speaker_buckets(records) == {"s1": "english_dominant"}


def test_validate_inventory_rejects_duplicate_segments(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.csv"
    duplicate_row = {
        "segment_id": "dup",
        "speaker_id": "s1",
        "transcript": "x",
        "ratio_english_words": "0.5",
        "ratio_hindi_words": "0.5",
        "count_english_words": "1",
        "count_dev_words": "1",
        "segment_part": "0000",
    }
    _write_inventory(inventory_path, [duplicate_row] * 2, [0])
    rows = _load_inventory(inventory_path)
    with pytest.raises(SystemExit, match="duplicate segment ids"):
        _validate_inventory(
            rows, expected_rows=None, expected_checksum=None, path=inventory_path
        )


def test_validate_inventory_rejects_wrong_row_count(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.csv"
    rows = _rows([f"s{index:04d}" for index in range(5)])
    _write_inventory(inventory_path, rows, [0, 1, 2])
    loaded = _load_inventory(inventory_path)
    with pytest.raises(SystemExit, match="row mismatch"):
        _validate_inventory(
            loaded,
            expected_rows=len(rows) + 1,
            expected_checksum=None,
            path=inventory_path,
        )


def test_validate_inventory_rejects_checksum_mismatch(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.csv"
    rows = _rows(["s0001", "s0002"])
    _write_inventory(inventory_path, rows, [0])
    loaded = _load_inventory(inventory_path)
    with pytest.raises(SystemExit, match="checksum mismatch"):
        _validate_inventory(
            loaded,
            expected_rows=len(rows),
            expected_checksum="0" * 64,
            path=inventory_path,
        )


def test_validate_inventory_rejects_shard_gap(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.csv"
    rows = _rows(["s0001", "s0002", "s0003"])
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "segment_id",
                "speaker_id",
                "transcript",
                "ratio_english_words",
                "ratio_hindi_words",
                "count_english_words",
                "count_dev_words",
                "segment_part",
                "shard",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {**row, "shard": "2" if index % 2 == 0 else "0"}
            for index, row in enumerate(rows)
        )
    loaded = _load_inventory(inventory_path)
    with pytest.raises(SystemExit, match="shard boundaries are not contiguous"):
        _validate_inventory(
            loaded,
            expected_rows=len(rows),
            expected_checksum=None,
            path=inventory_path,
        )


def test_validate_inventory_reports_invalid_ratio_rows(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.csv"
    rows = _rows(["s0001", "s0002"])
    rows[0]["ratio_english_words"] = "1.5"
    rows[0]["ratio_hindi_words"] = "-0.5"
    _write_inventory(inventory_path, rows, [0])
    loaded = _load_inventory(inventory_path)
    stats = _validate_inventory(
        loaded, expected_rows=None, expected_checksum=None, path=inventory_path
    )
    assert stats["invalid_ratio_rows"] == [rows[0]["segment_id"]]


def test_diagnostic_speakers_default_are_known() -> None:
    assert DIAGNOSTIC_SPEAKERS_DEFAULT == (
        "100051",
        "100356",
        "102503",
        "104662",
        "106319",
    )


def test_sha256_digest(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert _sha256(path) == hashlib.sha256(b"").hexdigest()


def test_end_to_end_selection_produces_thirty_frozen_clips(
    tmp_path: Path,
) -> None:
    inventory_path = tmp_path / "inventory.csv"
    output_dir = tmp_path / "out"
    speakers = [f"s{index:04d}" for index in range(300)]
    rows = _rows(speakers)
    _write_inventory(inventory_path, rows, [0, 1, 2])

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "create_mucs_validation_split.py"),
        "--inventory", str(inventory_path),
        "--inventory-sha256", _sha256(inventory_path),
        "--expected-rows", str(len(rows)),
        "--validation-speakers", "30",
        "--validation-ratio", "0.15",
        "--seed", "42",
        "--exclude-speakers", "s0000",
        "--output-dir", str(output_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    manifest = list(
        csv.DictReader(
            (output_dir / "frozen_validation_manifest.csv").open(encoding="utf-8")
        )
    )
    assert len(manifest) == 30
    assert len({row["speaker_id"] for row in manifest}) == 30
    assert all(row["assigned_role"] == "validation" for row in manifest)
    assert all(row["source_partition"] == "train" for row in manifest)
    assert all(row["split_seed"] == "42" for row in manifest)
    assert all(
        row["selection_version"] == "mendspeech.mucs_validation.v1" for row in manifest
    )
    assert all(row["reference_transcript"].strip() for row in manifest)
    assert all(row["dataset_revision"] for row in manifest)

    selected_speakers = {row["speaker_id"] for row in manifest}
    # The excluded diagnostic speaker must not enter the validation cohort.
    assert "s0000" not in selected_speakers

    speaker_split = json.loads(
        (output_dir / "speaker_split.json").read_text(encoding="utf-8")
    )
    assert "s0000" in speaker_split["training_speakers"]
    assert set(speaker_split["validation_speakers"]).isdisjoint(
        set(speaker_split["training_speakers"])
    )

    audit = json.loads(
        (output_dir / "leakage_audit.json").read_text(encoding="utf-8")
    )
    assert audit["speaker_overlap"] == []
    assert audit["duplicate_clips_within_split"] == []
    assert audit["duplicate_clips_across_split"] == []
    assert audit["diagnostic_speakers_in_validation"] == []
    assert "recording_separation_limitation" in audit

    recorded_checksum = (
        output_dir / "frozen_validation_manifest.sha256"
    ).read_text().split()[0]
    assert hashlib.sha256(
        (output_dir / "frozen_validation_manifest.csv").read_bytes()
    ).hexdigest() == recorded_checksum


def test_end_to_end_selection_is_reproducible(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.csv"
    speakers = [f"s{index:04d}" for index in range(60)]
    rows = _rows(speakers)
    _write_inventory(inventory_path, rows, [0])

    base_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "create_mucs_validation_split.py"),
        "--inventory", str(inventory_path),
        "--inventory-sha256", _sha256(inventory_path),
        "--expected-rows", str(len(rows)),
        "--validation-speakers", "9",
        "--validation-ratio", "0.15",
        "--seed", "42",
        "--exclude-speakers", "",
        "--output-dir",
    ]
    first = tmp_path / "first"
    second = tmp_path / "second"
    for destination in (first, second):
        completed = subprocess.run(
            base_command + [str(destination)], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr

    first_checksum = (
        first / "frozen_validation_manifest.sha256"
    ).read_text().split()[0]
    second_checksum = (
        second / "frozen_validation_manifest.sha256"
    ).read_text().split()[0]
    assert first_checksum == second_checksum


def test_selection_fails_gracefully_when_pool_is_too_small(
    tmp_path: Path,
) -> None:
    """Asking for more validation speakers than the pool is a clean error."""
    import subprocess

    inventory_path = tmp_path / "inventory.csv"
    output_dir = tmp_path / "out"
    speakers = [f"s{index:04d}" for index in range(60)]
    rows = _rows(speakers)
    _write_inventory(inventory_path, rows, [0])

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "create_mucs_validation_split.py"),
        "--inventory", str(inventory_path),
        "--inventory-sha256", _sha256(inventory_path),
        "--expected-rows", str(len(rows)),
        "--validation-speakers", "60",
        "--validation-ratio", "0.15",
        "--seed", "42",
        "--exclude-speakers", "",
        "--output-dir", str(output_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 1
    assert "speakers for stratified selection" in completed.stderr
    assert not (output_dir / "frozen_validation_manifest.csv").exists()

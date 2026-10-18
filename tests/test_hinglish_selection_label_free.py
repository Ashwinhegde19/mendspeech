"""Offline tests for the §6 selection-correction contract.

Covers:

1. ``--require-full-predictions`` rejecting a missing prediction file.
2. Failed-inference rows never silently counting as usable ASR coverage.
3. ``max_per_speaker`` defaulting to one clip per speaker.
4. The distinct-speaker assertion on finalized arms after duration matching.
5. Explicit ``oracle`` versus ``label_free`` acquisition modes.
6. Label-free ranking prohibiting ground-truth WER/S/D/I features, reference
   word counts, and transcript-derived language ratios.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hinglish.selection import (
    ACQUISITION_LABEL_FREE,
    ACQUISITION_ORACLE,
    DEFAULT_MAX_PER_SPEAKER,
    AsrFailureFeatures,
    ArmSelection,
    CandidateRecord,
    RankedCandidate,
    SelectionConfig,
    assert_finalized_arm_speakers,
    build_paired_selection,
    score_candidate,
)

CLI = REPO_ROOT / "scripts" / "select_hinglish_training_candidates.py"
SENTINEL = "SECRET_REFERENCE_SENTINEL"


def _candidate(
    clip_id: str,
    speaker_id: str,
    *,
    ratio_english: float = 0.5,
    count_english: int = 8,
    count_hindi: int = 12,
    duration: float | None = 6.0,
    shard: str = "0",
    asr: AsrFailureFeatures | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        clip_id=clip_id,
        speaker_id=speaker_id,
        reference_available=True,
        ratio_english_words=ratio_english,
        ratio_hindi_words=1.0 - ratio_english,
        count_english_words=count_english,
        count_hindi_words=count_hindi,
        segment_part="0000",
        shard=shard,
        duration_sec=duration,
        metadata_valid=True,
        invalid_fields=(),
        asr=asr,
    )


def _asr(wer: float = 0.5) -> AsrFailureFeatures:
    return AsrFailureFeatures(
        reference_words=10,
        substitutions=5,
        deletions=0,
        insertions=0,
        wer=wer,
        hallucination_flag=False,
    )


# ---------------------------------------------------------------------------
# 3. max_per_speaker default
# ---------------------------------------------------------------------------


def test_default_max_per_speaker_is_one() -> None:
    assert DEFAULT_MAX_PER_SPEAKER == 1
    assert SelectionConfig().max_per_speaker == 1


# ---------------------------------------------------------------------------
# 5/6. acquisition modes
# ---------------------------------------------------------------------------


def test_unknown_acquisition_mode_rejected() -> None:
    with pytest.raises(ValueError):
        SelectionConfig(acquisition_mode="hybrid")
    assert SelectionConfig().acquisition_mode == ACQUISITION_ORACLE


def test_label_free_config_requires_duration_and_zero_word_gate() -> None:
    with pytest.raises(ValueError):
        SelectionConfig(acquisition_mode=ACQUISITION_LABEL_FREE)
    with pytest.raises(ValueError):
        SelectionConfig(
            acquisition_mode=ACQUISITION_LABEL_FREE,
            require_duration=True,
            min_total_words=2,
        )
    config = SelectionConfig(
        acquisition_mode=ACQUISITION_LABEL_FREE,
        require_duration=True,
        min_total_words=0,
    )
    assert config.acquisition_mode == ACQUISITION_LABEL_FREE


def test_oracle_scoring_uses_reference_features() -> None:
    config = SelectionConfig()
    same_duration_high_ratio = _candidate("a", "s1", ratio_english=0.5, count_english=20, count_hindi=20)
    same_duration_low_ratio = _candidate("b", "s2", ratio_english=0.02, count_english=0, count_hindi=1)
    high = score_candidate(same_duration_high_ratio, config)
    low = score_candidate(same_duration_low_ratio, config)
    assert high.components["code_mix"] == 1.0
    assert high.score > low.score
    assert high.components["quality"] > low.components["quality"]


def test_label_free_scoring_uses_duration_only() -> None:
    config = SelectionConfig(
        acquisition_mode=ACQUISITION_LABEL_FREE,
        require_duration=True,
        min_total_words=0,
    )
    transcript_rich = _candidate("a", "s1", ratio_english=0.5, count_english=40, count_hindi=40)
    transcript_poor = _candidate("b", "s2", ratio_english=0.02, count_english=0, count_hindi=1)
    rich = score_candidate(transcript_rich, config)
    poor = score_candidate(transcript_poor, config)
    assert rich.components["code_mix"] is None
    assert rich.components["quality"] is None
    assert rich.score == poor.score == rich.components["duration"]


def test_label_free_scoring_prohibits_asr_features() -> None:
    config = SelectionConfig(
        acquisition_mode=ACQUISITION_LABEL_FREE,
        require_duration=True,
        min_total_words=0,
    )
    with pytest.raises(ValueError):
        score_candidate(_candidate("a", "s1", asr=_asr()), config)


# ---------------------------------------------------------------------------
# 4. distinct-speaker assertion on finalized arms
# ---------------------------------------------------------------------------


def _ranked(clip_id: str, speaker_id: str, rank: int) -> RankedCandidate:
    return RankedCandidate(
        rank=rank,
        candidate=_candidate(clip_id, speaker_id),
        score=0.5,
        components={"duration": 0.5},
        asr_status="unavailable",
        asr_wer=None,
        difficulty_bucket="unavailable",
        code_mix_bucket="in_band",
    )


def test_assert_finalized_arm_speakers_enforces_one_clip_per_speaker() -> None:
    one = SelectionConfig(max_per_speaker=1)
    two = SelectionConfig(max_per_speaker=2)
    arm = ArmSelection(
        arm="intelligent",
        ranked=(
            _ranked("a", "s1", 1),
            _ranked("b", "s1", 2),
        ),
        total_duration_sec=12.0,
        duration_known_count=2,
        duration_unknown_count=0,
    )
    with pytest.raises(ValueError):
        assert_finalized_arm_speakers(arm, one)
    assert_finalized_arm_speakers(arm, two)  # duplicates allowed at cap 2
    distinct = ArmSelection(
        arm="intelligent",
        ranked=(_ranked("a", "s1", 1), _ranked("b", "s2", 2)),
        total_duration_sec=12.0,
        duration_known_count=2,
        duration_unknown_count=0,
    )
    assert_finalized_arm_speakers(distinct, one)


def test_paired_selection_distinct_speakers_after_budget_matching() -> None:
    candidates: list[CandidateRecord] = []
    speakers = [f"spk{index:02d}" for index in range(10)]
    for index, speaker in enumerate(speakers):
        for part in range(2):
            candidates.append(
                _candidate(
                    f"{speaker}_seg{part:04d}",
                    speaker,
                    duration=2.0 + index * 0.4 + part * 0.9,
                    shard=str((index + part) % 3),
                )
            )
    config = SelectionConfig(
        target_count=8,
        max_per_speaker=1,
        duration_tolerance=0.05,
    )
    result = build_paired_selection(candidates, training_speakers=speakers, config=config)
    for arm in (result.intelligent, result.random):
        clip_ids = [entry.candidate for entry in arm.ranked]
        assert len(clip_ids) == 8
        assert len({entry.candidate.speaker_id for entry in arm.ranked}) == 8


# ---------------------------------------------------------------------------
# 1/2. require-full-predictions and failed-row coverage (CLI)
# ---------------------------------------------------------------------------


def _write_inventory(path: Path, speakers: list[str], per_speaker: int = 2) -> None:
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
        for index, speaker in enumerate(speakers):
            for part in range(per_speaker):
                writer.writerow(
                    {
                        "segment_id": f"{speaker}_seg{part:04d}",
                        "speaker_id": speaker,
                        "transcript": f"{SENTINEL} words {index} {part}",
                        "ratio_english_words": 0.5,
                        "ratio_hindi_words": 0.5,
                        "count_english_words": 5,
                        "count_dev_words": 5,
                        "segment_part": f"{part:04d}",
                        "shard": str((index + part) % 2),
                    }
                )


def _write_split(path: Path, training: list[str], validation: list[str]) -> None:
    payload = {
        "seed": 42,
        "training_speakers": training,
        "validation_speakers": validation,
        "diagnostic_speakers_reserved_for_training": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_predictions(
    path: Path, rows: list[tuple[str, str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip_id", "prediction", "status"], lineterminator="\n"
        )
        writer.writeheader()
        for clip_id, prediction, status in rows:
            writer.writerow(
                {"clip_id": clip_id, "prediction": prediction, "status": status}
            )


def _write_candidate_pool(path: Path, clip_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id"], lineterminator="\n")
        writer.writeheader()
        for clip_id in clip_ids:
            writer.writerow({"clip_id": clip_id})


def _base_command(
    tmp_path: Path, speakers: list[str], *, dry_run: bool = True
) -> tuple[list[str], Path]:
    inventory = tmp_path / "inventory.csv"
    split = tmp_path / "split.json"
    _write_inventory(inventory, speakers)
    _write_split(split, speakers[:-1], speakers[-1:])
    command = [
        sys.executable,
        str(CLI),
        "--inventory", str(inventory),
        "--speaker-split", str(split),
        "--arm", "intelligent",
        "--count", "2",
        "--output-dir", str(tmp_path / "out"),
    ]
    if dry_run:
        command.append("--dry-run")
    return command, split


def test_require_full_predictions_rejects_missing_file(tmp_path: Path) -> None:
    speakers = [f"spk{index:02d}" for index in range(4)]
    command, _ = _base_command(tmp_path, speakers)
    completed = subprocess.run(
        command
        + [
            "--candidate-pool", str(tmp_path / "pool.csv"),
            "--predictions", str(tmp_path / "absent.csv"),
            "--require-full-predictions",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "not found" in completed.stderr


def test_require_full_predictions_requires_predictions_argument(tmp_path: Path) -> None:
    speakers = [f"spk{index:02d}" for index in range(4)]
    command, _ = _base_command(tmp_path, speakers)
    pool_rows = [f"spk{index:02d}_seg0000" for index in range(3)]
    _write_candidate_pool(tmp_path / "pool.csv", pool_rows)
    completed = subprocess.run(
        command
        + ["--candidate-pool", str(tmp_path / "pool.csv"), "--require-full-predictions"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "requires --predictions" in completed.stderr


def test_require_full_predictions_rejects_failed_rows_as_not_covered(
    tmp_path: Path,
) -> None:
    speakers = [f"spk{index:02d}" for index in range(4)]
    command, _ = _base_command(tmp_path, speakers)
    pool_rows = [f"spk{index:02d}_seg0000" for index in range(3)]
    _write_candidate_pool(tmp_path / "pool.csv", pool_rows)
    _write_predictions(
        tmp_path / "predictions.csv",
        [(pool_rows[0], "okay words", "ok"), (pool_rows[1], "", "failed")],
    )
    completed = subprocess.run(
        command
        + [
            "--candidate-pool", str(tmp_path / "pool.csv"),
            "--predictions", str(tmp_path / "predictions.csv"),
            "--require-full-predictions",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "failed inference rows" in completed.stderr


def test_failed_row_is_reported_not_counted_as_coverage(tmp_path: Path) -> None:
    speakers = [f"spk{index:02d}" for index in range(4)]
    command, _ = _base_command(tmp_path, speakers, dry_run=False)
    pool_rows = [f"spk{index:02d}_seg0000" for index in range(3)]
    _write_candidate_pool(tmp_path / "pool.csv", pool_rows)
    _write_predictions(
        tmp_path / "predictions.csv",
        [(pool_rows[0], "okay words", "ok"), (pool_rows[1], "", "failed")],
    )
    output_dir = tmp_path / "out"
    completed = subprocess.run(
        command
        + [
            "--candidate-pool", str(tmp_path / "pool.csv"),
            "--predictions", str(tmp_path / "predictions.csv"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output_dir / "selection_report.json").read_text(encoding="utf-8"))
    inputs = report["inputs"]
    assert inputs["prediction_failed_clip_count"] == 1
    assert inputs["prediction_failed_clip_ids"] == [pool_rows[1]]
    assert inputs["prediction_usable_count"] == 1
    assert inputs["prediction_coverage_fraction"] == 0.3333


def test_cli_label_free_requires_duration_and_zero_word_gate(tmp_path: Path) -> None:
    speakers = [f"spk{index:02d}" for index in range(4)]
    command, _ = _base_command(tmp_path, speakers)
    completed = subprocess.run(
        command + ["--acquisition-mode", "label_free"], capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "--require-duration" in completed.stderr

    completed = subprocess.run(
        command + ["--acquisition-mode", "label_free", "--require-duration"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--min-total-words 0" in completed.stderr

    clip_ids = [f"spk{index:02d}_seg0000" for index in range(3)]
    _write_candidate_pool(tmp_path / "pool.csv", clip_ids)
    durations = tmp_path / "durations.csv"
    with durations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip_id", "duration_sec"], lineterminator="\n"
        )
        writer.writeheader()
        for index, clip_id in enumerate(clip_ids):
            writer.writerow({"clip_id": clip_id, "duration_sec": 5.0 + index})
    output_dir = tmp_path / "out"
    command_write, _ = _base_command(tmp_path, speakers, dry_run=False)
    completed = subprocess.run(
        command_write
        + [
            "--acquisition-mode", "label_free",
            "--require-duration",
            "--min-total-words", "0",
            "--duration-manifest", str(durations),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output_dir / "selection_report.json").read_text(encoding="utf-8"))
    assert report["arms"]["intelligent"]["distinct_speakers"] == 2
    assert report["acquisition_mode"] == "label_free"
    assert report["pool"]["asr_available_count"] == 0
    assert report["pool"]["eligible"] == 3


def test_cli_label_free_rejects_predictions(tmp_path: Path) -> None:
    speakers = [f"spk{index:02d}" for index in range(4)]
    command, _ = _base_command(tmp_path, speakers)
    pool_rows = [f"spk{index:02d}_seg0000" for index in range(3)]
    _write_candidate_pool(tmp_path / "pool.csv", pool_rows)
    _write_predictions(
        tmp_path / "predictions.csv", [(pool_rows[0], "words", "ok")]
    )
    completed = subprocess.run(
        command
        + [
            "--acquisition-mode", "label_free",
            "--require-duration",
            "--min-total-words", "0",
            "--candidate-pool", str(tmp_path / "pool.csv"),
            "--predictions", str(tmp_path / "predictions.csv"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "prohibits" in completed.stderr

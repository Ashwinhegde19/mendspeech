"""Tests for deterministic Hinglish candidate selection.

All fixtures are synthetic. No private reference transcripts or measured
predictions are used, and no network access is required.
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
    ARM_INTELLIGENT,
    ARM_RANDOM,
    SCORING_WEIGHTS,
    AsrFailureFeatures,
    CandidateRecord,
    SelectionConfig,
    _assert_safe_rows,
    build_paired_selection,
    build_selection_report,
    code_mix_component,
    dedupe_candidates,
    evaluate_eligibility,
    failure_component,
    match_duration_budget,
    prepare_candidate_pool,
    score_candidate,
    select_intelligent_arm,
    select_random_arm,
    selection_manifest_rows,
    write_selection_manifest,
)

SPEAKERS = tuple(f"spk{index:03d}" for index in range(60))


def _candidate(
    clip_id: str,
    speaker_id: str,
    *,
    ratio_english: float = 0.5,
    count_english: int = 8,
    count_hindi: int = 12,
    duration: float | None = 6.0,
    shard: str = "0",
    metadata_valid: bool = True,
    reference_available: bool = True,
    asr: AsrFailureFeatures | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        clip_id=clip_id,
        speaker_id=speaker_id,
        reference_available=reference_available,
        ratio_english_words=ratio_english,
        ratio_hindi_words=1.0 - ratio_english,
        count_english_words=count_english,
        count_hindi_words=count_hindi,
        segment_part="0000",
        shard=shard,
        duration_sec=duration,
        metadata_valid=metadata_valid,
        invalid_fields=() if metadata_valid else ("ratio_english_words",),
        asr=asr,
    )


def _asr(
    reference_words: int = 10,
    substitutions: int = 2,
    deletions: int = 0,
    insertions: int = 0,
    wer: float = 0.2,
    hallucination: bool = False,
) -> AsrFailureFeatures:
    return AsrFailureFeatures(
        reference_words=reference_words,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        wer=wer,
        hallucination_flag=hallucination,
    )


def _pool(count: int = 120, *, per_speaker: int = 2) -> list[CandidateRecord]:
    candidates: list[CandidateRecord] = []
    for speaker_index, speaker in enumerate(SPEAKERS):
        for part in range(per_speaker):
            candidates.append(
                _candidate(
                    f"{speaker}_seg{part:04d}",
                    speaker,
                    ratio_english=0.3 + 0.005 * (speaker_index % 40),
                    shard=str((speaker_index + part) % 4),
                )
            )
            if len(candidates) >= count:
                return candidates
    return candidates


# ---------------------------------------------------------------------------
# Configuration and eligibility
# ---------------------------------------------------------------------------


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        SelectionConfig(target_count=0)
    with pytest.raises(ValueError):
        SelectionConfig(max_per_speaker=0)
    with pytest.raises(ValueError):
        SelectionConfig(min_duration_sec=5.0, max_duration_sec=2.0)
    with pytest.raises(ValueError):
        SelectionConfig(duration_tolerance=1.5)
    with pytest.raises(ValueError):
        SelectionConfig(preferred_duration_sec=25.0)


def test_eligibility_rejects_speaker_outside_training_pool() -> None:
    decision = evaluate_eligibility(
        _candidate("a", "outsider"), training_speakers={"spk001"}
    )
    assert not decision.eligible
    assert "speaker_outside_training_pool" in decision.reasons


def test_eligibility_rejects_forbidden_speakers_and_clips() -> None:
    decision = evaluate_eligibility(
        _candidate("a", "spk001"),
        training_speakers={"spk001"},
        forbidden_speakers={"spk001"},
        forbidden_clip_ids={"a"},
    )
    assert not decision.eligible
    assert "speaker_forbidden" in decision.reasons
    assert "clip_forbidden" in decision.reasons


def test_eligibility_rejects_invalid_states() -> None:
    cases = [
        (_candidate("a", "s", metadata_valid=False), "metadata_invalid"),
        (_candidate("a", "s", reference_available=False), "reference_unavailable"),
        (_candidate("a", "s", ratio_english=1.4), "ratio_out_of_range"),
        (_candidate("a", "s", duration=0.2), "duration_out_of_bounds"),
        (_candidate("a", "s", duration=60.0), "duration_out_of_bounds"),
        (_candidate("a", "s", count_english=0, count_hindi=1), "insufficient_word_count"),
        (_candidate("a", "s", duration=None), "duration_missing"),
    ]
    for candidate, expected in cases:
        config = SelectionConfig(require_duration=True)
        decision = evaluate_eligibility(
            candidate, training_speakers={"s"}, config=config
        )
        assert not decision.eligible, expected
        assert expected in decision.reasons, (expected, decision.reasons)


def test_eligibility_rejects_empty_reference_after_normalization() -> None:
    decision = evaluate_eligibility(
        _candidate("a", "s", asr=_asr(reference_words=0)),
        training_speakers={"s"},
    )
    assert not decision.eligible
    assert "empty_reference_after_normalization" in decision.reasons


def test_dedupe_keeps_first_occurrence() -> None:
    result = dedupe_candidates(
        [_candidate("a", "s1"), _candidate("a", "s2"), _candidate("b", "s3")]
    )
    assert [candidate.clip_id for candidate in result.kept] == ["a", "b"]
    assert result.duplicate_clip_ids == ("a",)


def test_prepare_pool_requires_training_speakers() -> None:
    with pytest.raises(ValueError, match="training_speakers"):
        prepare_candidate_pool([_candidate("a", "s")], training_speakers=[])


def test_prepare_pool_records_rejections_and_availability() -> None:
    pool = prepare_candidate_pool(
        [
            _candidate("a", "spk001"),
            _candidate("b", "outsider"),
            _candidate("c", "spk001", asr=_asr()),
        ],
        training_speakers={"spk001"},
    )
    assert {candidate.clip_id for candidate in pool.eligible} == {"a", "c"}
    assert pool.rejected == (("b", ("speaker_outside_training_pool",)),)
    assert pool.asr_available_count == 1
    assert pool.asr_unavailable_count == 1
    assert pool.duration_unknown_count == 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_failure_component_saturates_and_floors_hallucinations() -> None:
    assert failure_component(None) is None
    assert failure_component(_asr(reference_words=0)) is None
    assert failure_component(_asr(substitutions=0, deletions=0, insertions=0)) == 0.0
    extreme = failure_component(
        _asr(reference_words=10, substitutions=50, deletions=50, insertions=100)
    )
    assert extreme == pytest.approx(1.0)
    flagged = failure_component(_asr(substitutions=0, wer=0.0, hallucination=True))
    assert flagged == pytest.approx(0.75)


def test_code_mix_component_band_and_falloff() -> None:
    assert code_mix_component(0.5) == 1.0
    assert code_mix_component(0.4) == 1.0
    assert code_mix_component(0.6) == 1.0
    assert code_mix_component(0.3) == pytest.approx(0.75)
    assert code_mix_component(0.0) == 0.0
    assert code_mix_component(1.0) == 0.0


def test_score_marks_asr_unavailable_and_renormalizes() -> None:
    candidate = _candidate("a", "s", ratio_english=0.5, duration=8.0)
    score = score_candidate(candidate)
    assert score.asr_status == "unavailable"
    assert score.components["failure"] is None
    expected = (
        SCORING_WEIGHTS["code_mix"] * 1.0
        + SCORING_WEIGHTS["duration"] * 1.0
        + SCORING_WEIGHTS["quality"] * min(20 / 30, 1.0)
    ) / (SCORING_WEIGHTS["code_mix"] + SCORING_WEIGHTS["duration"] + SCORING_WEIGHTS["quality"])
    assert score.score == pytest.approx(expected)


def test_score_marks_duration_unavailable_when_unknown() -> None:
    score = score_candidate(_candidate("a", "s", duration=None))
    assert score.components["duration"] is None
    expected = (0.3 * 1.0 + 0.1 * min(20 / 30, 1.0)) / 0.4
    assert score.score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Arm selection determinism and constraints
# ---------------------------------------------------------------------------


def test_random_arm_is_deterministic_across_runs() -> None:
    pool = _pool()
    first = select_random_arm(pool, SelectionConfig(target_count=20))
    second = select_random_arm(pool, SelectionConfig(target_count=20))
    assert [entry.candidate.clip_id for entry in first.ranked] == [
        entry.candidate.clip_id for entry in second.ranked
    ]
    assert first.sample_order == second.sample_order


def test_random_arm_changes_with_seed() -> None:
    pool = _pool()
    first = select_random_arm(pool, SelectionConfig(target_count=20, seed=1))
    second = select_random_arm(pool, SelectionConfig(target_count=20, seed=2))
    assert first.sample_order != second.sample_order


def test_intelligent_arm_is_deterministic_across_runs() -> None:
    pool = _pool()
    first = select_intelligent_arm(pool, SelectionConfig(target_count=20))
    second = select_intelligent_arm(pool, SelectionConfig(target_count=20))
    assert first.ranked == second.ranked


def test_per_speaker_cap_enforced_in_both_arms() -> None:
    pool = _pool(count=60, per_speaker=3)
    config = SelectionConfig(target_count=30, max_per_speaker=2)
    for arm in (select_random_arm(pool, config), select_intelligent_arm(pool, config)):
        counts: dict[str, int] = {}
        for entry in arm.ranked:
            counts[entry.candidate.speaker_id] = counts.get(entry.candidate.speaker_id, 0) + 1
        assert counts and max(counts.values()) <= 2
        assert len(counts) >= 15


def test_one_clip_per_speaker_preferred() -> None:
    pool = [
        _candidate(f"{speaker}_seg{part:04d}", speaker)
        for speaker in ("s1", "s2", "s3")
        for part in range(4)
    ]
    config = SelectionConfig(target_count=3, max_per_speaker=2)
    for arm in (select_random_arm(pool, config), select_intelligent_arm(pool, config)):
        speakers = [entry.candidate.speaker_id for entry in arm.ranked]
        assert sorted(speakers) == ["s1", "s2", "s3"]


def test_selection_raises_when_pool_cannot_satisfy_target() -> None:
    pool = [_candidate(f"c{index}", f"s{index}") for index in range(3)]
    config = SelectionConfig(target_count=5)
    for selector in (select_random_arm, select_intelligent_arm):
        with pytest.raises(ValueError, match="target_count"):
            selector(pool, config)


def test_max_per_speaker_one_caps_target() -> None:
    pool = [_candidate(f"c{index}", "s1") for index in range(5)]
    with pytest.raises(ValueError, match="target_count"):
        select_intelligent_arm(pool, SelectionConfig(target_count=2, max_per_speaker=1))


def test_intelligent_prefers_in_band_code_mix() -> None:
    in_band = _candidate("in_band", "sA", ratio_english=0.5)
    below = _candidate("below", "sB", ratio_english=0.1)
    above = _candidate("above", "sC", ratio_english=0.9)
    arm = select_intelligent_arm(
        [below, above, in_band],
        SelectionConfig(target_count=3, max_per_speaker=1),
    )
    assert arm.ranked[0].candidate.clip_id == "in_band"


def test_intelligent_prefers_hallucination_heavy_clip() -> None:
    clean = _candidate("clean", "sA", asr=_asr(substitutions=1, wer=0.1))
    broken = _candidate(
        "broken",
        "sB",
        asr=_asr(substitutions=0, insertions=40, wer=4.0, hallucination=True),
    )
    arm = select_intelligent_arm(
        [clean, broken], SelectionConfig(target_count=2, max_per_speaker=1)
    )
    assert arm.ranked[0].candidate.clip_id == "broken"
    assert arm.ranked[0].asr_status == "available"


def test_intelligent_shard_interleave_spreads_rank_prefix() -> None:
    pool = [
        _candidate(f"spk{index:03d}_seg0000", f"spk{index:03d}", shard=str(index % 4))
        for index in range(40)
    ]
    arm = select_intelligent_arm(
        pool, SelectionConfig(target_count=10, max_per_speaker=1)
    )
    shards = {entry.candidate.shard for entry in arm.ranked[:10]}
    assert len(shards) >= 3


def test_unavailable_asr_never_scores_zero() -> None:
    without = _candidate("a", "s1")
    with_feature = _candidate("b", "s2", asr=_asr(substitutions=0, deletions=0, insertions=0, wer=0.0))
    score_without = score_candidate(without)
    score_with = score_candidate(with_feature)
    assert score_without.components["failure"] is None
    assert score_with.components["failure"] == 0.0
    # A perfect-but-unavailable candidate is not penalised below a measured
    # perfect candidate on accounting grounds, but the marks differ.
    assert score_without.asr_status == "unavailable"
    assert score_with.asr_status == "available"
    assert score_without.score > 0.0


# ---------------------------------------------------------------------------
# Duration-budget matching
# ---------------------------------------------------------------------------


def test_matched_budget_within_tolerance() -> None:
    pool: list[CandidateRecord] = []
    for speaker_index in range(60):
        speaker = f"spk{speaker_index:03d}"
        long_duration = 10.0 + (speaker_index % 3) * 0.1
        pool.append(
            _candidate(
                f"{speaker}_long", speaker, duration=long_duration, shard=str(speaker_index % 4)
            )
        )
        pool.append(
            _candidate(
                f"{speaker}_short", speaker, duration=2.0, shard=str(speaker_index % 4)
            )
        )
    config = SelectionConfig(target_count=30, max_per_speaker=2, duration_tolerance=0.02)
    paired = build_paired_selection(pool, training_speakers=set(SPEAKERS), config=config)
    assert paired.intelligent.total_duration_sec is not None
    assert paired.random.budget_match is not None
    match = paired.random.budget_match
    assert match.applied
    assert match.within_tolerance, match
    assert abs(match.residual_fraction) <= config.duration_tolerance
    assert len(paired.random.ranked) == 30
    assert (
        paired.random.total_duration_sec - paired.intelligent.total_duration_sec
    ) == pytest.approx(match.residual_sec, abs=1e-9)


def test_matched_budget_tolerates_duration_variety() -> None:
    pool = [
        _candidate(
            f"spk{index:03d}_seg0000",
            f"spk{index:03d}",
            duration=2.0 + (index % 17) * 0.75,
            shard=str(index % 3),
        )
        for index in range(120)
    ]
    config = SelectionConfig(target_count=40, max_per_speaker=1, duration_tolerance=0.02)
    paired = build_paired_selection(pool, training_speakers=set(SPEAKERS), config=config)
    assert paired.random.budget_match is not None
    assert paired.random.budget_match.within_tolerance
    assert paired.random.budget_match.swaps_applied >= 0


def test_match_duration_budget_reports_residual_when_impossible() -> None:
    selection = [_candidate(f"a{index}", f"s{index}", duration=10.0) for index in range(10)]
    replacements = [_candidate(f"b{index}", f"r{index}", duration=10.0) for index in range(10)]
    adjusted, result = match_duration_budget(
        selection, replacements, target_total_duration_sec=150.0, tolerance=0.02
    )
    assert adjusted == tuple(selection)
    assert result.applied
    assert not result.within_tolerance
    assert result.note == "tolerance_not_reached_after_swaps"
    assert result.residual_sec == pytest.approx(-50.0)


def test_match_duration_budget_skips_without_durations() -> None:
    selection = [_candidate("a", "s1", duration=None)]
    adjusted, result = match_duration_budget(
        selection, [], target_total_duration_sec=10.0
    )
    assert result.applied is False
    assert result.note == "skipped_unknown_durations"
    assert not result.within_tolerance


def test_match_duration_budget_is_deterministic() -> None:
    selection = [_candidate(f"a{index}", f"s{index}", duration=5.0) for index in range(10)]
    replacements = [
        _candidate(f"b{index}", f"r{index}", duration=5.0 + index * 0.5) for index in range(20)
    ]
    first, first_result = match_duration_budget(
        selection, replacements, target_total_duration_sec=80.0
    )
    second, second_result = match_duration_budget(
        selection, replacements, target_total_duration_sec=80.0
    )
    assert first == second
    assert first_result == second_result


def test_match_duration_budget_keeps_count_and_speaker_cap() -> None:
    # Both speakers sit at the cap of two. A replacement from s2 may only swap
    # a selected s2 clip; s1 slots cannot take s2 replacements without
    # exceeding the cap.
    selection = [
        _candidate("a1", "s1", duration=5.0),
        _candidate("a2", "s1", duration=5.0),
        _candidate("a3", "s2", duration=5.0),
        _candidate("a4", "s2", duration=5.0),
    ]
    replacements = [
        _candidate(f"b{index}", "s2", duration=12.0) for index in range(4)
    ]
    adjusted, result = match_duration_budget(
        selection,
        replacements,
        target_total_duration_sec=48.0,
        max_per_speaker=2,
        tolerance=0.01,
    )
    assert len(adjusted) == 4
    counts: dict[str, int] = {}
    for candidate in adjusted:
        counts[candidate.speaker_id] = counts.get(candidate.speaker_id, 0) + 1
    assert counts == {"s1": 2, "s2": 2}
    assert result.swaps_applied == 2
    assert not result.within_tolerance
    assert result.residual_sec == pytest.approx(-14.0)


# ---------------------------------------------------------------------------
# Reporting and safe output
# ---------------------------------------------------------------------------


def test_report_contains_bias_warning_and_cap_usage() -> None:
    pool = _pool()
    config = SelectionConfig(target_count=20, max_per_speaker=2)
    intelligent = select_intelligent_arm(pool, config)
    random_arm = select_random_arm(pool, config)
    candidate_pool = prepare_candidate_pool(pool, training_speakers=set(SPEAKERS), config=config)
    report = build_selection_report(
        candidate_pool, {ARM_RANDOM: random_arm, ARM_INTELLIGENT: intelligent}, config
    )
    assert "bias_warning" in report
    assert "Intelligent selection" in report["bias_warning"]
    assert len(report["mitigations"]) >= 4
    for arm_name in (ARM_RANDOM, ARM_INTELLIGENT):
        summary = report["arms"][arm_name]
        assert summary["size"] == 20
        assert summary["distinct_speakers"] >= 10
        assert summary["clips_per_speaker_max"] <= 2
        assert set(summary["difficulty_bucket_counts"]) == {"unavailable"}
        assert sum(summary["shard_counts"].values()) == 20
    assert report["pool"]["eligible"] == len(pool)


def test_manifest_rows_are_safe() -> None:
    pool = _pool()
    arm = select_intelligent_arm(pool, SelectionConfig(target_count=5))
    rows = selection_manifest_rows(arm)
    assert all("transcript" not in key for row in rows for key in row)
    assert all(row["arm"] == "intelligent" for row in rows)
    assert [row["rank"] for row in rows] == [1, 2, 3, 4, 5]


def test_assert_safe_rows_rejects_unsafe_columns() -> None:
    with pytest.raises(ValueError, match="unsafe manifest column"):
        _assert_safe_rows([{"clip_id": "a", "reference_transcript": "x"}])
    with pytest.raises(ValueError, match="unknown manifest column"):
        _assert_safe_rows([{"clip_id": "a", "mystery": 1}])


def test_write_selection_manifest_header(tmp_path: Path) -> None:
    arm = select_intelligent_arm(_pool(), SelectionConfig(target_count=3))
    path = tmp_path / "manifest.csv"
    write_selection_manifest(path, arm)
    with path.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "transcript" not in ",".join(header)
    assert "clip_id" in header
    assert "arm" in header


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------

CLI = REPO_ROOT / "scripts" / "select_hinglish_training_candidates.py"
SENTINEL = "SECRET_REFERENCE_SENTINEL"


def _write_inventory(path: Path, speakers: list[str], per_speaker: int = 3) -> None:
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
            if index % 2 == 0:
                ratio, count_english, count_hindi = 0.5, 5, 5
            else:
                ratio, count_english, count_hindi = 0.1, 1, 9
            for part in range(per_speaker):
                writer.writerow(
                    {
                        "segment_id": f"{speaker}_seg{part:04d}",
                        "speaker_id": speaker,
                        "transcript": f"{SENTINEL} words {index} {part}",
                        "ratio_english_words": ratio,
                        "ratio_hindi_words": 1.0 - ratio,
                        "count_english_words": count_english,
                        "count_dev_words": count_hindi,
                        "segment_part": f"{part:04d}",
                        "shard": str((index + part) % 3),
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


def _write_durations(path: Path, clip_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip_id", "duration_sec"], lineterminator="\n"
        )
        writer.writeheader()
        for index, clip_id in enumerate(clip_ids):
            writer.writerow({"clip_id": clip_id, "duration_sec": 2.0 + (index % 11) * 0.7})


def test_cli_end_to_end_writes_safe_manifests(tmp_path: Path) -> None:
    speakers = [f"spk{index:03d}" for index in range(12)]
    inventory = tmp_path / "inventory.csv"
    split = tmp_path / "split.json"
    _write_inventory(inventory, speakers)
    _write_split(split, speakers[:10], speakers[10:])
    clip_ids = [f"{speaker}_seg{part:04d}" for speaker in speakers for part in range(3)]
    durations = tmp_path / "durations.csv"
    _write_durations(durations, clip_ids)
    output_dir = tmp_path / "out"

    command = [
        sys.executable,
        str(CLI),
        "--inventory", str(inventory),
        "--speaker-split", str(split),
        "--arm", "both",
        "--count", "8",
        "--seed", "7",
        "--max-per-speaker", "2",
        "--duration-manifest", str(durations),
        "--output-dir", str(output_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    for arm in ("random", "intelligent"):
        manifest = output_dir / f"selection_manifest_{arm}.csv"
        assert manifest.exists()
        content = manifest.read_text(encoding="utf-8")
        assert SENTINEL not in content
        with manifest.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 8
        assert {row["speaker_id"] for row in rows} <= set(speakers[:10])
        assert all(row["arm"] == arm for row in rows)

    report = json.loads((output_dir / "selection_report.json").read_text(encoding="utf-8"))
    assert report["transcript_written"] is False
    assert set(report["arms"]) == {"random", "intelligent"}
    assert report["arms"]["random"]["budget_match"] is not None
    assert report["arms"]["random"]["budget_match"]["applied"] is True
    assert report["selection_guardrails"]["validation_speakers_forbidden"] is True
    assert "bias_warning" in report
    assert SENTINEL not in (output_dir / "selection_report.json").read_text(encoding="utf-8")


def test_cli_is_reproducible(tmp_path: Path) -> None:
    speakers = [f"spk{index:03d}" for index in range(12)]
    inventory = tmp_path / "inventory.csv"
    split = tmp_path / "split.json"
    _write_inventory(inventory, speakers)
    _write_split(split, speakers[:10], speakers[10:])
    base = [
        sys.executable,
        str(CLI),
        "--inventory", str(inventory),
        "--speaker-split", str(split),
        "--arm", "both",
        "--count", "6",
        "--max-per-speaker", "2",
    ]
    for destination_name in ("first", "second"):
        completed = subprocess.run(
            base + ["--output-dir", str(tmp_path / destination_name)],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
    for arm in ("random", "intelligent"):
        first = (tmp_path / "first" / f"selection_manifest_{arm}.csv").read_bytes()
        second = (tmp_path / "second" / f"selection_manifest_{arm}.csv").read_bytes()
        assert first == second


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    speakers = [f"spk{index:03d}" for index in range(6)]
    inventory = tmp_path / "inventory.csv"
    split = tmp_path / "split.json"
    _write_inventory(inventory, speakers)
    _write_split(split, speakers[:5], speakers[5:])
    output_dir = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--inventory", str(inventory),
            "--speaker-split", str(split),
            "--arm", "intelligent",
            "--count", "4",
            "--dry-run",
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not output_dir.exists()


def test_cli_rejects_overlapping_split(tmp_path: Path) -> None:
    speakers = [f"spk{index:03d}" for index in range(6)]
    inventory = tmp_path / "inventory.csv"
    split = tmp_path / "split.json"
    _write_inventory(inventory, speakers)
    _write_split(split, speakers[:5], speakers[:3])
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--inventory", str(inventory),
            "--speaker-split", str(split),
            "--arm", "random",
            "--count", "2",
            "--output-dir", str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "overlaps" in completed.stderr


def test_cli_rejects_duplicate_and_unknown_prediction_ids(tmp_path: Path) -> None:
    speakers = [f"spk{index:03d}" for index in range(6)]
    inventory = tmp_path / "inventory.csv"
    split = tmp_path / "split.json"
    _write_inventory(inventory, speakers)
    _write_split(split, speakers[:5], speakers[5:])

    duplicate = tmp_path / "dup.csv"
    duplicate.write_text(
        "clip_id,prediction\nspk000_seg0000,a\nspk000_seg0000,b\n", encoding="utf-8"
    )
    completed = subprocess.run(
        [
            sys.executable, str(CLI),
            "--inventory", str(inventory),
            "--speaker-split", str(split),
            "--arm", "intelligent",
            "--count", "2",
            "--predictions", str(duplicate),
            "--output-dir", str(tmp_path / "out_dup"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "duplicate" in completed.stderr

    unknown = tmp_path / "unknown.csv"
    unknown.write_text("clip_id,prediction\nnope_seg0000,a\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable, str(CLI),
            "--inventory", str(inventory),
            "--speaker-split", str(split),
            "--arm", "intelligent",
            "--count", "2",
            "--predictions", str(unknown),
            "--output-dir", str(tmp_path / "out_unknown"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "unknown prediction clip IDs" in completed.stderr


def test_cli_predictions_mark_features_and_score_failures_higher(tmp_path: Path) -> None:
    speakers = [f"spk{index:03d}" for index in range(6)]
    inventory = tmp_path / "inventory.csv"
    split = tmp_path / "split.json"
    _write_inventory(inventory, speakers)
    _write_split(split, speakers[:5], speakers[5:])
    # Two clips with identical composition metadata (English ratio 0.5) get
    # predictions: one heavily failing and one perfect. Every other clip has
    # no prediction, so its failure component must be marked unavailable
    # rather than scored as zero.
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip_id", "prediction"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow({"clip_id": "spk000_seg0000", "prediction": "totally wrong output"})
        writer.writerow({"clip_id": "spk002_seg0000", "prediction": SENTINEL + " words 2 0"})
    output_dir = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable, str(CLI),
            "--inventory", str(inventory),
            "--speaker-split", str(split),
            "--arm", "intelligent",
            "--count", "15",
            "--max-per-speaker", "3",
            "--predictions", str(predictions),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    with (output_dir / "selection_manifest_intelligent.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_clip = {row["clip_id"]: row for row in rows}
    assert by_clip["spk000_seg0000"]["asr_status"] == "available"
    assert by_clip["spk002_seg0000"]["asr_status"] == "available"
    # The failing clip scores higher than the perfect clip with the same
    # composition metadata.
    assert float(by_clip["spk000_seg0000"]["score"]) > float(
        by_clip["spk002_seg0000"]["score"]
    )
    # Unpredicted eligible clips keep the failure component unavailable.
    assert any(row["asr_status"] == "unavailable" for row in rows)
    # The prediction text never leaves the scoring path.
    for path in output_dir.iterdir():
        assert SENTINEL not in path.read_text(encoding="utf-8")

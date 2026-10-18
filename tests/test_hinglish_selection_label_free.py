"""Offline tests for the §6 selection-correction contract.

Covers:

1. ``--require-full-predictions`` rejecting a missing prediction file.
2. Failed-inference rows never silently counting as usable ASR coverage.
3. ``max_per_speaker`` defaulting to one clip per speaker.
4. The distinct-speaker assertion on finalized arms after duration matching.
5. Explicit ``oracle`` versus ``label_free`` acquisition modes.
6. Label-free ranking prohibiting ground-truth WER/S/D/I features, reference
   word counts, and transcript-derived language ratios.
7. The frozen ``hinglish.labelfree.v1`` scoring formula: hand-computed
   component values, weight renormalization over available components,
   failed-scoring-record rejection (``scoring_failed``), repetition floors
   and the natural-stop clamp, the J1-derived length-anomaly band, oracle
   golden regression floats, the mandatory ``AsrOutputFeatures`` record,
   and the manifest writer's None handling.
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
    AsrOutputFeatures,
    ArmSelection,
    CandidateRecord,
    LABEL_FREE_COMPONENTS,
    LABEL_FREE_SCORING_WEIGHTS,
    LF_CAP_HIT_NO_EOS_FLOOR,
    LF_LENGTH_ANOMALY_CAP_HIT_FLOOR,
    LF_NATURAL_EOS_REPETITION_CEILING,
    LF_REPETITION_FLAG_FLOOR,
    LF_TOKEN_RATE_BAND_HIGH,
    LF_TOKEN_RATE_BAND_LOW,
    LF_TOKEN_RATE_BAND_MEAN,
    LF_TOKEN_RATE_SATURATION_HIGH,
    LF_TOKEN_RATE_SATURATION_LOW,
    LF_QUALITY_PEAK_FLOOR_RISK,
    RankedCandidate,
    SAFE_MANIFEST_COLUMNS,
    SCORING_WEIGHTS,
    SelectionConfig,
    assert_finalized_arm_speakers,
    build_paired_selection,
    build_selection_report,
    lf_audio_quality_component,
    lf_confidence_component,
    lf_length_anomaly_component,
    lf_repetition_component,
    prepare_candidate_pool,
    score_candidate,
    selection_manifest_rows,
)

CLI = REPO_ROOT / "scripts" / "select_hinglish_training_candidates.py"
SENTINEL = "SECRET_REFERENCE_SENTINEL"


def _candidate(
    clip_id: str,
    speaker_id: str,
    *,
    ratio_english: "float | None" = 0.5,
    count_english: "int | None" = 8,
    count_hindi: "int | None" = 12,
    duration: float | None = 6.0,
    shard: str = "0",
    asr: AsrFailureFeatures | None = None,
    asr_output: AsrOutputFeatures | None = None,
    audio_peak_dbfs: float | None = None,
    audio_clipping_fraction: float | None = None,
    audio_mean_rms_dbfs: float | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        clip_id=clip_id,
        speaker_id=speaker_id,
        reference_available=True,
        ratio_english_words=ratio_english,
        ratio_hindi_words=1.0 - ratio_english if ratio_english is not None else None,
        count_english_words=count_english,
        count_hindi_words=count_hindi,
        segment_part="0000",
        shard=shard,
        duration_sec=duration,
        metadata_valid=True,
        invalid_fields=(),
        asr=asr,
        asr_output=asr_output,
        audio_peak_dbfs=audio_peak_dbfs,
        audio_clipping_fraction=audio_clipping_fraction,
        audio_mean_rms_dbfs=audio_mean_rms_dbfs,
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


def _asr_output(
    *,
    status: str = "ok",
    generated_token_count: int = 86,
    eos_emitted_effective: bool = True,
    hit_length_limit_effective: bool = False,
    repeated_token_fraction: float | None = 0.0,
    max_consecutive_repeated_tokens: int | None = 1,
    repetition_flag: bool | None = False,
    mean_logprob: float | None = -0.1,
    min_logprob: float | None = -0.8,
    audio_duration_sec: float | None = 7.275,
    inference_runtime_sec: float | None = 1.0,
    entropy_mean: float | None = 2.5,
) -> AsrOutputFeatures:
    """A healthy, naturally-stopped scoring record as a base for tweaking."""
    return AsrOutputFeatures(
        status=status,
        audio_duration_sec=audio_duration_sec,
        inference_runtime_sec=inference_runtime_sec,
        generated_token_count=generated_token_count,
        eos_emitted_effective=eos_emitted_effective,
        hit_length_limit_effective=hit_length_limit_effective,
        max_consecutive_repeated_tokens=max_consecutive_repeated_tokens,
        repeated_token_fraction=repeated_token_fraction,
        repetition_flag=repetition_flag,
        mean_logprob=mean_logprob,
        min_logprob=min_logprob,
        entropy_mean=entropy_mean,
    )


def _label_free_config(**overrides: object) -> SelectionConfig:
    defaults: dict[str, object] = {
        "acquisition_mode": ACQUISITION_LABEL_FREE,
        "require_duration": True,
        "min_total_words": 0,
    }
    defaults.update(overrides)
    return SelectionConfig(**defaults)  # type: ignore[arg-type]


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


def test_label_free_scoring_ignores_transcript_derived_signals() -> None:
    config = _label_free_config()
    transcript_rich = _candidate(
        "a", "s1", ratio_english=0.5, count_english=40, count_hindi=40,
        asr_output=_asr_output(),
    )
    transcript_poor = _candidate(
        "b", "s2", ratio_english=0.02, count_english=0, count_hindi=1,
        asr_output=_asr_output(),
    )
    rich = score_candidate(transcript_rich, config)
    poor = score_candidate(transcript_poor, config)
    # The component set in label_free mode is exactly the five frozen
    # formula components; transcript-derived code_mix/quality are not part
    # of it (not even as unavailable slots).
    assert set(rich.components) == set(LABEL_FREE_COMPONENTS)
    # The two records differ only in transcript-derived fields; every
    # component value and the overall score must be identical.
    rich_clean = {name: value for name, value in rich.components.items()}
    poor_clean = {name: value for name, value in poor.components.items()}
    assert rich_clean == poor_clean
    assert rich.score == poor.score
    # Renormalization check: score is the weighted mean over available
    # components only (unavailable -> None, never zero-filled).
    weighted = sum(
        LABEL_FREE_SCORING_WEIGHTS[name] * float(value)
        for name, value in rich.components.items()
        if value is not None
    )
    weight_total = sum(
        LABEL_FREE_SCORING_WEIGHTS[name]
        for name, value in rich.components.items()
        if value is not None
    )
    assert rich.score == pytest.approx(weighted / weight_total)


def test_label_free_scoring_prohibits_asr_features() -> None:
    config = _label_free_config()
    with pytest.raises(ValueError, match="prohibits ground-truth"):
        score_candidate(_candidate("a", "s1", asr=_asr()), config)


def test_oracle_golden_scores_are_byte_identical() -> None:
    """Oracle scoring must equal the pre-change behaviour exactly.

    Golden values were captured from the pre-change module for fixed
    records: float identity is asserted (not approx), so any drift in the
    oracle path or weight table breaks this test.
    """
    golden = CandidateRecord(
        clip_id="golden",
        speaker_id="spk",
        reference_available=True,
        ratio_english_words=0.5,
        ratio_hindi_words=0.5,
        count_english_words=20,
        count_hindi_words=20,
        segment_part="0000",
        shard="0",
        duration_sec=6.0,
        metadata_valid=True,
        invalid_fields=(),
        asr=AsrFailureFeatures(
            reference_words=10,
            substitutions=5,
            deletions=0,
            insertions=0,
            wer=0.5,
            hallucination_flag=False,
        ),
    )
    score = score_candidate(golden, SelectionConfig())
    assert dict(score.components) == {
        "failure": 0.175,
        "code_mix": 1.0,
        "duration": 0.8333333333333334,
        "quality": 1.0,
    }
    assert score.score == 0.5708333333333333
    assert score.asr_status == "available"
    no_asr = CandidateRecord(
        clip_id="golden2",
        speaker_id="spk2",
        reference_available=True,
        ratio_english_words=0.5,
        ratio_hindi_words=0.5,
        count_english_words=20,
        count_hindi_words=20,
        duration_sec=8.0,
    )
    score_without = score_candidate(no_asr, SelectionConfig())
    assert dict(score_without.components) == {
        "failure": None,
        "code_mix": 1.0,
        "duration": 1.0,
        "quality": 1.0,
    }
    assert score_without.score == 1.0
    # Oracle mode ignores a carried scoring record entirely.
    carried = dict(golden.__dict__)
    carried["asr_output"] = _asr_output()
    with_carried = score_candidate(CandidateRecord(**carried), SelectionConfig())
    assert with_carried.score == 0.5708333333333333
    assert dict(with_carried.components) == {
        "failure": 0.175,
        "code_mix": 1.0,
        "duration": 0.8333333333333334,
        "quality": 1.0,
    }


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


def test_cli_label_free_config_gates_and_premature_run_fails(
    tmp_path: Path,
) -> None:
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

    # Config gates pass. A premature label_free run (before the J2 scoring
    # run exists) must fail loudly instead of silently ranking on duration
    # alone ("selection CLI runs only post-J2"). Under the integrated
    # contract the CLI requires the frozen pool manifest and the completed
    # scoring results; missing inputs hard-stop, and a scoring file missing
    # coverage hard-stops listing the gap.
    clip_ids = [f"spk{index:02d}_seg0000" for index in range(3)]
    pool_header = "clip_id,speaker_id,audio_duration_sec,shard\n"
    pool_body = "".join(
        f"{clip_id},{clip_id[:6]},6.5,0\n" for clip_id in clip_ids
    )
    (tmp_path / "pool_manifest.csv").write_text(
        pool_header + pool_body, encoding="utf-8"
    )
    command_write, _ = _base_command(tmp_path, speakers, dry_run=False)
    completed = subprocess.run(
        command_write
        + ["--acquisition-mode", "label_free", "--require-duration",
           "--min-total-words", "0"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    # Structural gates fire in fixed order: pool manifest first, then scoring.
    assert "--pool-manifest" in completed.stderr

    empty_scoring = {"run": "premature", "config": "C1", "records": []}
    (tmp_path / "empty_scoring.json").write_text(
        json.dumps(empty_scoring), encoding="utf-8"
    )
    completed = subprocess.run(
        command_write
        + [
            "--acquisition-mode", "label_free",
            "--require-duration",
            "--min-total-words", "0",
            "--pool-manifest", str(tmp_path / "pool_manifest.csv"),
            "--scoring", str(tmp_path / "empty_scoring.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "non-empty records list" in completed.stderr

    partial = {
        "run": "premature",
        "config": "C1",
        "records": [
            {
                "clip_id": clip_ids[0],
                "status": "ok",
                "generated_token_count": 40,
                "eos_emitted_effective": True,
                "hit_length_limit_effective": False,
                "max_consecutive_repeated_tokens": 1,
                "repeated_token_fraction": 0.02,
                "repetition_flag": False,
                "mean_logprob": -0.1,
                "min_logprob": -0.3,
                "entropy_mean": 0.8,
            }
        ],
    }
    (tmp_path / "partial_scoring.json").write_text(
        json.dumps(partial), encoding="utf-8"
    )
    completed = subprocess.run(
        command_write
        + [
            "--acquisition-mode", "label_free",
            "--require-duration",
            "--min-total-words", "0",
            "--pool-manifest", str(tmp_path / "pool_manifest.csv"),
            "--scoring", str(tmp_path / "partial_scoring.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "missing" in completed.stderr


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


# ---------------------------------------------------------------------------
# 7. Frozen formula ``hinglish.labelfree.v1``
# ---------------------------------------------------------------------------


def test_label_free_formula_hand_computed() -> None:
    """Hand-computed component values for one fully specified record.

    All inputs are chosen to hit band midpoints and simple arithmetic; the
    expected values re-derive the frozen math independently of the module
    implementation helpers.
    """
    config = _label_free_config()
    candidate = _candidate(
        "lf1",
        "s1",
        duration=10.0,
        asr_output=AsrOutputFeatures(
            status="ok",
            audio_duration_sec=10.0,
            generated_token_count=140,
            eos_emitted_effective=True,
            hit_length_limit_effective=False,
            repeated_token_fraction=0.2,
            max_consecutive_repeated_tokens=6,
            repetition_flag=False,
            mean_logprob=-0.175,
            min_logprob=-0.45,
        ),
        audio_peak_dbfs=-3.0,
        audio_clipping_fraction=0.01,
        audio_mean_rms_dbfs=48.0,
    )
    score = score_candidate(candidate, config)
    components = score.components

    # duration_fit: duration 10.0 s inside [1, 20] decays linearly from
    # the 8.0 s preferred point: 1.0 - |10 - 8| / 12 = 0.8333...
    assert components["duration_fit"] == pytest.approx(1.0 - 2.0 / 12.0)

    # repetition_risk: base = 0.6*min(0.2/0.4,1) + 0.4*min(6/24,1)
    #   = 0.6*0.5 + 0.4*0.25 = 0.4; natural stop (EOS true, cap false)
    #   clamps to at most 0.55, so the value stays 0.4.
    assert components["repetition_risk"] == pytest.approx(0.4)

    # length_anomaly: rate = 140/10.0 = 14.0 tokens/sec > band high edge.
    anomaly = (14.0 - LF_TOKEN_RATE_BAND_HIGH) / (
        LF_TOKEN_RATE_SATURATION_HIGH - LF_TOKEN_RATE_BAND_HIGH
    )
    assert components["length_anomaly"] == pytest.approx(anomaly)
    assert 0.0 < components["length_anomaly"] < 1.0

    # confidence_risk: min risk (0.10 - (-0.45))/1.10 = 0.5 and
    # mean risk (0.05 - (-0.175))/0.45 = 0.5, combined 0.5/0.5.
    assert components["confidence_risk"] == pytest.approx(0.5)

    # audio_quality_risk: clipping 0.01/0.05 = 0.2 (peak -3.0 < -0.5, no
    # floor) and rms (48 - 38)/20 = 0.5, combined at equal weights.
    assert components["audio_quality_risk"] == pytest.approx((0.2 + 0.5) / 2.0)

    # Overall: all five components available, frozen weights sum to 1.0.
    assert sum(LABEL_FREE_SCORING_WEIGHTS.values()) == pytest.approx(1.0)
    duration_fit = 1.0 - 2.0 / 12.0
    expected_overall = (
        LABEL_FREE_SCORING_WEIGHTS["duration_fit"] * duration_fit
        + LABEL_FREE_SCORING_WEIGHTS["repetition_risk"] * 0.4
        + LABEL_FREE_SCORING_WEIGHTS["length_anomaly"] * anomaly
        + LABEL_FREE_SCORING_WEIGHTS["confidence_risk"] * 0.5
        + LABEL_FREE_SCORING_WEIGHTS["audio_quality_risk"] * 0.35
    )
    assert score.score == pytest.approx(expected_overall)


def test_label_free_renormalizes_over_available_components() -> None:
    """Missing confidence and quality inputs -> None, weights renormalize."""
    config = _label_free_config()
    candidate = _candidate(
        "lf2",
        "s1",
        duration=10.0,
        asr_output=AsrOutputFeatures(
            status="ok",
            generated_token_count=140,
            eos_emitted_effective=True,
            hit_length_limit_effective=False,
            repeated_token_fraction=0.2,
            max_consecutive_repeated_tokens=6,
            repetition_flag=False,
            mean_logprob=None,
            min_logprob=None,
        ),
    )
    score = score_candidate(candidate, config)
    assert score.components["confidence_risk"] is None
    assert score.components["audio_quality_risk"] is None
    available = {
        name: value
        for name, value in score.components.items()
        if value is not None
    }
    assert set(available) == {"duration_fit", "repetition_risk", "length_anomaly"}
    anomaly = (14.0 - LF_TOKEN_RATE_BAND_HIGH) / (
        LF_TOKEN_RATE_SATURATION_HIGH - LF_TOKEN_RATE_BAND_HIGH
    )
    duration_fit = 1.0 - 2.0 / 12.0
    weighted = (
        LABEL_FREE_SCORING_WEIGHTS["duration_fit"] * duration_fit
        + LABEL_FREE_SCORING_WEIGHTS["repetition_risk"] * 0.4
        + LABEL_FREE_SCORING_WEIGHTS["length_anomaly"] * anomaly
    )
    renormalized = weighted / (
        LABEL_FREE_SCORING_WEIGHTS["duration_fit"]
        + LABEL_FREE_SCORING_WEIGHTS["repetition_risk"]
        + LABEL_FREE_SCORING_WEIGHTS["length_anomaly"]
    )
    assert score.score == pytest.approx(renormalized)
    # The missing components are never zero-filled: with zero-filling the
    # score could only stay unchanged when the renormalization happens to
    # hide the difference; assert the renormalized value is strictly the
    # weighted mean over available components only.
    assert score.score == pytest.approx(weighted / 0.80)


def test_label_free_rejects_failed_scoring_record() -> None:
    """A ``failed`` record rejects eligibility; it is never scored/zeroed."""
    failed = AsrOutputFeatures(
        status="failed",
        audio_duration_sec=6.0,
        generated_token_count=None,
        eos_emitted_effective=None,
        hit_length_limit_effective=None,
        max_consecutive_repeated_tokens=None,
        repeated_token_fraction=None,
        repetition_flag=None,
        mean_logprob=None,
        min_logprob=None,
    )
    candidate = _candidate("lf_fail", "s1", duration=6.0, asr_output=failed)
    config = _label_free_config()
    decision = prepare_candidate_pool(
        [candidate],
        training_speakers=["s1"],
        config=config,
    )
    assert decision.eligible == ()
    assert decision.rejected == (("lf_fail", ("scoring_failed",)),)
    # Never selectable: scoring raises instead of zero-filling.
    with pytest.raises(
        ValueError, match="label_free scoring requires AsrOutputFeatures"
    ):
        score_candidate(candidate, config)
    # The status-gate is label_free-scoped: oracle mode does not add the
    # reason for a failed model record.
    oracle_decision = prepare_candidate_pool(
        [candidate], training_speakers=["s1"], config=SelectionConfig()
    )
    assert oracle_decision.eligible == (candidate,)
    assert oracle_decision.rejected == ()


def test_label_free_missing_record_raises_exactly() -> None:
    config = _label_free_config()
    with pytest.raises(
        ValueError,
        match="label_free scoring requires AsrOutputFeatures \\(status ok\\) "
        "for candidate lf_none",
    ):
        score_candidate(
            _candidate("lf_none", "s1", duration=6.0), config
        )


def test_label_free_ground_truth_prohibition_covers_both_records() -> None:
    config = _label_free_config()
    both = _candidate(
        "lf_both", "s1", duration=6.0, asr=_asr(), asr_output=_asr_output()
    )
    with pytest.raises(ValueError, match="prohibits ground-truth"):
        score_candidate(both, config)


def test_repetition_component_floor_and_natural_stop_clamp() -> None:
    # Cap-hit without EOS: floored at 0.85 regardless of a low base
    # (J1 evidence: cap-hit without EOS is looping until truncation).
    cap_hit = _asr_output(
        repeated_token_fraction=0.05,
        max_consecutive_repeated_tokens=2,
        repetition_flag=False,
        eos_emitted_effective=False,
        hit_length_limit_effective=True,
    )
    base = 0.6 * min(0.05 / 0.4, 1.0) + 0.4 * min(2.0 / 24.0, 1.0)
    assert lf_repetition_component(cap_hit) == pytest.approx(
        LF_CAP_HIT_NO_EOS_FLOOR
    )
    assert LF_CAP_HIT_NO_EOS_FLOOR > base

    # repetition_flag floor at 0.75 when no precedence rule overrides it
    # (here: stop mode unknown, no cap hit).
    flagged = _asr_output(
        repeated_token_fraction=0.1,
        max_consecutive_repeated_tokens=4,
        repetition_flag=True,
        eos_emitted_effective=None,
        hit_length_limit_effective=False,
    )
    assert lf_repetition_component(flagged) == pytest.approx(
        LF_REPETITION_FLAG_FLOOR
    )

    # Frozen precedence: the natural-stop clamp is evaluated LAST and is
    # therefore unconditional over the flag floor — a flagged record that
    # stopped naturally is clamped to 0.55 (spec: natural stop reduces
    # suspicion). This is a deliberate consequence of the frozen rule
    # order, not a bug.
    flagged_natural = _asr_output(
        repeated_token_fraction=0.1,
        max_consecutive_repeated_tokens=4,
        repetition_flag=True,
        eos_emitted_effective=True,
        hit_length_limit_effective=False,
    )
    assert lf_repetition_component(flagged_natural) == pytest.approx(
        LF_NATURAL_EOS_REPETITION_CEILING
    )

    # Cap floor dominates the flag floor (evaluated after it).
    flagged_cap_hit = _asr_output(
        repeated_token_fraction=0.0,
        max_consecutive_repeated_tokens=0,
        repetition_flag=True,
        eos_emitted_effective=False,
        hit_length_limit_effective=True,
    )
    assert lf_repetition_component(flagged_cap_hit) == pytest.approx(
        LF_CAP_HIT_NO_EOS_FLOOR
    )

    # Natural stop (EOS, no cap): extreme base clamps to at most 0.55.
    natural_extreme = _asr_output(
        repeated_token_fraction=1.0,
        max_consecutive_repeated_tokens=30,
        repetition_flag=False,
        eos_emitted_effective=True,
        hit_length_limit_effective=False,
    )
    assert lf_repetition_component(natural_extreme) == pytest.approx(
        LF_NATURAL_EOS_REPETITION_CEILING
    )
    # ...and a modest base stays below the clamp threshold untouched:
    # the healthy record's base is 0.4 * (1/24) and the EOS clamp only
    # limits values above 0.55.
    natural_modest = _asr_output()  # healthy defaults
    assert lf_repetition_component(natural_modest) == pytest.approx(
        0.4 * (1.0 / 24.0)
    )

    # Contradictory instrumentation (EOS and cap both effective) applies
    # neither rule; the base stands.
    contradictory = _asr_output(
        repeated_token_fraction=1.0,
        max_consecutive_repeated_tokens=30,
        repetition_flag=False,
        eos_emitted_effective=True,
        hit_length_limit_effective=True,
    )
    assert lf_repetition_component(contradictory) == pytest.approx(1.0)

    # Missing repetition statistics -> None, never zero.
    assert lf_repetition_component(
        _asr_output(repeated_token_fraction=None, max_consecutive_repeated_tokens=1)
    ) is None
    assert lf_repetition_component(
        _asr_output(repeated_token_fraction=0.0, max_consecutive_repeated_tokens=None)
    ) is None
    assert lf_confidence_component(
        _asr_output(mean_logprob=None)
    ) is None


def test_length_anomaly_band_edges_and_cap_override() -> None:
    # Anchor the frozen saturation-constant relations that the
    # pre-registration JSON documents: saturations are 50% of/beyond the
    # BAND EDGES (not of any single control clip rate).
    assert LF_TOKEN_RATE_SATURATION_LOW == pytest.approx(0.5 * LF_TOKEN_RATE_BAND_LOW)
    assert LF_TOKEN_RATE_SATURATION_HIGH == pytest.approx(1.5 * LF_TOKEN_RATE_BAND_HIGH)
    assert LF_TOKEN_RATE_BAND_LOW == pytest.approx(16.0 / 1.55)
    assert LF_TOKEN_RATE_BAND_HIGH == pytest.approx(76.0 / 5.55)
    assert LF_TOKEN_RATE_BAND_MEAN == pytest.approx(
        (86.0 / 7.275 + 76.0 / 5.55 + 16.0 / 1.55) / 3.0
    )

    # Interior rate: the healthy band is exactly 0.
    inside = lf_length_anomaly_component(_asr_output(generated_token_count=24), 2.0)
    assert inside == 0.0

    # Below the band: linear ramp from 0 at the low edge.
    below = lf_length_anomaly_component(
        _asr_output(generated_token_count=103), 10.0
    )
    expected_below = (LF_TOKEN_RATE_BAND_LOW - 10.3) / (
        LF_TOKEN_RATE_BAND_LOW - LF_TOKEN_RATE_SATURATION_LOW
    )
    assert below == pytest.approx(expected_below)

    # Above the band: linear ramp from 0 at the high edge.
    above = lf_length_anomaly_component(
        _asr_output(generated_token_count=138), 10.0
    )
    expected_above = (13.8 - LF_TOKEN_RATE_BAND_HIGH) / (
        LF_TOKEN_RATE_SATURATION_HIGH - LF_TOKEN_RATE_BAND_HIGH
    )
    assert above == pytest.approx(expected_above)

    # Saturations: exactly 50% below the band low edge -> 1.0;
    # far above the high edge -> clamped 1.0.
    at_low_saturation = lf_length_anomaly_component(
        _asr_output(generated_token_count=1),
        1.0 / LF_TOKEN_RATE_SATURATION_LOW,
    )
    assert at_low_saturation == pytest.approx(1.0)
    far_above = lf_length_anomaly_component(
        _asr_output(generated_token_count=100), 1.0
    )
    assert far_above == 1.0

    # Cap-hit without EOS overrides to at least 0.80 even when the rate
    # is inside the healthy band.
    cap_override = lf_length_anomaly_component(
        _asr_output(
            generated_token_count=24,
            eos_emitted_effective=False,
            hit_length_limit_effective=True,
        ),
        2.0,
    )
    assert cap_override == pytest.approx(LF_LENGTH_ANOMALY_CAP_HIT_FLOOR)
    # ...and never lowers a larger anomaly.
    dominated = lf_length_anomaly_component(
        _asr_output(
            generated_token_count=100,
            eos_emitted_effective=False,
            hit_length_limit_effective=True,
        ),
        1.0,
    )
    assert dominated == 1.0

    # Unknown inputs: component None (never zero).
    assert lf_length_anomaly_component(_asr_output(generated_token_count=None), 2.0) is None
    assert lf_length_anomaly_component(_asr_output(generated_token_count=24), None) is None
    with pytest.raises(ValueError, match="duration_sec must be positive"):
        lf_length_anomaly_component(_asr_output(generated_token_count=24), 0.0)


def test_audio_quality_component_partial_inputs_renormalize() -> None:
    # All inputs absent -> None.
    assert lf_audio_quality_component(
        audio_peak_dbfs=None, audio_clipping_fraction=None, audio_mean_rms_dbfs=None
    ) is None
    # Clipping only.
    assert lf_audio_quality_component(
        audio_peak_dbfs=None, audio_clipping_fraction=0.02, audio_mean_rms_dbfs=None
    ) == pytest.approx(0.02 / 0.05)
    # RMS only.
    assert lf_audio_quality_component(
        audio_peak_dbfs=None, audio_clipping_fraction=None, audio_mean_rms_dbfs=58.0
    ) == 1.0
    # Peak floor stands alone without a clipping measurement.
    assert lf_audio_quality_component(
        audio_peak_dbfs=-0.2, audio_clipping_fraction=None, audio_mean_rms_dbfs=None
    ) == pytest.approx(LF_QUALITY_PEAK_FLOOR_RISK)
    # ...and combines with an RMS term at equal weights: peak floor 0.70
    # with a quiet RMS risk of 0.0 -> (0.70 + 0.0) / 2 = 0.35.
    assert lf_audio_quality_component(
        audio_peak_dbfs=-0.2, audio_clipping_fraction=None, audio_mean_rms_dbfs=38.0
    ) == pytest.approx(0.35)
    # Peak floor never lowers a bigger base: clipping 0.10 saturates the
    # base at 1.0 and the quiet RMS is 0.0 -> 0.5.
    assert lf_audio_quality_component(
        audio_peak_dbfs=-0.2, audio_clipping_fraction=0.10, audio_mean_rms_dbfs=38.0
    ) == pytest.approx((1.0 + 0.0) / 2.0)
    # Quiet, unconspicuous audio -> 0.0 risk (computed, not unavailable).
    fully_quiet = lf_audio_quality_component(
        audio_peak_dbfs=-6.0, audio_clipping_fraction=0.0, audio_mean_rms_dbfs=38.0
    )
    assert fully_quiet == 0.0


def test_build_selection_report_label_free_keys(tmp_path: Path) -> None:
    config = _label_free_config(target_count=2)
    ok = _candidate("lf_ok", "s1", duration=8.0, asr_output=_asr_output())
    ok2 = _candidate("lf_ok2", "s2", duration=8.0, asr_output=_asr_output())
    pool = prepare_candidate_pool([ok, ok2], training_speakers=["s1", "s2"], config=config)
    arm = build_paired_selection([ok, ok2], training_speakers=["s1", "s2"], config=config)
    report = build_selection_report(pool, {
        "intelligent": arm.intelligent,
        "random": arm.random,
    }, config)
    assert report["formula_version"] == "hinglish.labelfree.v1"
    availability = report["label_free_component_availability"]
    assert set(availability) == set(LABEL_FREE_COMPONENTS)
    assert availability["duration_fit"]["available_count"] == 2
    assert availability["duration_fit"]["eligible_count"] == 2
    assert availability["confidence_risk"]["available_count"] == 2
    assert availability["audio_quality_risk"]["available_count"] == 0
    assert availability["length_anomaly"]["available_count"] == 2

    # The rejected reason includes scoring_failed, and missing components
    # are reported as unavailable counts over the same eligible pool.
    failed = _candidate("lf_bad", "s3", duration=8.0, asr_output=_asr_output(status="failed"))
    no_logprob_cand = _candidate(
        "lf_ok3", "s4", duration=8.0, asr_output=_asr_output(mean_logprob=None, min_logprob=None)
    )
    pool2 = prepare_candidate_pool(
        [ok, ok2, no_logprob_cand, failed],
        training_speakers=["s1", "s2", "s3", "s4"],
        config=config,
    )
    assert pool2.rejected == (("lf_bad", ("scoring_failed",)),)
    paired2 = build_paired_selection(
        [ok, ok2, no_logprob_cand],
        training_speakers=["s1", "s2", "s4"],
        config=config,
    )
    report2 = build_selection_report(pool2, {
        "intelligent": paired2.intelligent,
        "random": paired2.random,
    }, config)
    assert report2["pool"]["rejection_reason_counts"]["scoring_failed"] == 1
    availability2 = report2["label_free_component_availability"]
    assert availability2["confidence_risk"]["available_count"] == 2
    assert availability2["confidence_risk"]["eligible_count"] == 3

    # Oracle reports carry None for both new keys.
    oracle_config = SelectionConfig(target_count=1)
    oracle_pool = prepare_candidate_pool(
        [_candidate("o1", "s1")], training_speakers=["s1"], config=oracle_config
    )
    oracle_arm = build_paired_selection(
        [_candidate("o1", "s1")], training_speakers=["s1"], config=oracle_config
    )
    oracle_report = build_selection_report(
        oracle_pool,
        {"intelligent": oracle_arm.intelligent, "random": oracle_arm.random},
        oracle_config,
    )
    assert oracle_report["formula_version"] is None
    assert oracle_report["label_free_component_availability"] is None


def test_manifest_rows_write_empty_strings_for_none_values() -> None:
    candidate = _candidate(
        "lf_manifest",
        "s1",
        ratio_english=None,
        count_english=None,
        count_hindi=None,
        duration=None,
        asr_output=_asr_output(),
    )
    score = score_candidate(candidate, _label_free_config())
    arm = ArmSelection(
        arm="intelligent",
        ranked=(
            RankedCandidate(
                rank=1,
                candidate=candidate,
                score=score.score,
                components=score.components,
                asr_status=score.asr_status,
                asr_wer=None,
                difficulty_bucket=score.difficulty_bucket,
                code_mix_bucket=score.code_mix_bucket,
            ),
        ),
        total_duration_sec=None,
        duration_known_count=0,
        duration_unknown_count=1,
    )
    rows = selection_manifest_rows(arm)
    assert len(rows) == 1
    row = rows[0]
    assert row["score"] == round(score.score, 6)
    # Columns that carry None in label-free outputs write "" and never a
    # NaN or None string.
    for column in (
        "ratio_english_words",
        "ratio_hindi_words",
        "count_english_words",
        "count_hindi_words",
        "duration_sec",
        "asr_wer",
    ):
        assert row[column] == "", (column, row[column])
    # Column set unchanged.
    assert tuple(row) == SAFE_MANIFEST_COLUMNS

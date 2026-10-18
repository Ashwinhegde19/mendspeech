"""Regression tests for Hinglish evaluation guardrails."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_eval(
    tmp_path: Path,
    predictions_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]] | None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    predictions_path = tmp_path / "predictions.csv"
    _write_csv(predictions_path, predictions_rows)
    command = [
        sys.executable,
        "scripts/evaluate_hinglish_predictions.py",
        "--predictions",
        str(predictions_path),
    ]
    if manifest_rows is not None:
        manifest_path = tmp_path / "manifest.csv"
        _write_csv(manifest_path, manifest_rows)
        command.extend(["--manifest", str(manifest_path)])
    if extra_args:
        command.extend(extra_args)
    return subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        capture_output=True,
    )


def test_evaluate_requires_manifest_flag() -> None:
    result = _run_eval(
        Path("/tmp"),
        [{"clip_id": "clip_1", "prediction": "hello"}],
        None,
    )
    assert result.returncode != 0
    assert "actions" in result.stderr or "required" in result.stderr.lower()


def test_valid_one_to_one_evaluation_succeeds(tmp_path: Path) -> None:
    predictions = [{"clip_id": "clip_1", "prediction": "hello"}]
    manifest = [
        {
            "clip_id": "clip_1",
            "transcript": "hello",
            "speaker_id": "spk_1",
            "ratio_english_words": 0.4,
        }
    ]
    result = _run_eval(tmp_path, predictions, manifest, ["--output", str(tmp_path / "metrics.json")])
    assert result.returncode == 0, result.stderr
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["utterances"] == 1
    assert metrics["data_quality"]["invalid_ratio_records"] == 0


def test_unknown_prediction_clip_is_rejected(tmp_path: Path) -> None:
    result = _run_eval(
        tmp_path,
        [{"clip_id": "missing", "prediction": "hello"}],
        [{"clip_id": "clip_1", "transcript": "hello", "speaker_id": "spk_1"}],
    )
    assert result.returncode != 0
    assert "unknown prediction clip IDs" in result.stderr


def test_manifest_clip_without_prediction_is_rejected(tmp_path: Path) -> None:
    result = _run_eval(
        tmp_path,
        [{"clip_id": "clip_1", "prediction": "hello"}],
        [
            {"clip_id": "clip_1", "transcript": "hello", "speaker_id": "spk_1"},
            {"clip_id": "clip_2", "transcript": "hello", "speaker_id": "spk_2"},
        ],
    )
    assert result.returncode != 0
    assert "without predictions" in result.stderr


def test_duplicate_prediction_clip_is_rejected(tmp_path: Path) -> None:
    result = _run_eval(
        tmp_path,
        [
            {"clip_id": "clip_1", "prediction": "hello"},
            {"clip_id": "clip_1", "prediction": "world"},
        ],
        [{"clip_id": "clip_1", "transcript": "hello", "speaker_id": "spk_1"}],
    )
    assert result.returncode != 0
    assert "duplicate clip IDs in predictions" in result.stderr


def test_duplicate_manifest_clip_is_rejected(tmp_path: Path) -> None:
    # Duplicate manifest must exist as a separate file, so construct one manually.
    manifest_path = tmp_path / "duplicate_manifest.csv"
    pd.DataFrame(
        [
            {"clip_id": "clip_1", "transcript": "hello", "speaker_id": "spk_1"},
            {"clip_id": "clip_1", "transcript": "hello", "speaker_id": "spk_1"},
        ]
    ).to_csv(manifest_path, index=False)
    predictions_path = tmp_path / "duplicate_predictions.csv"
    pd.DataFrame([{"clip_id": "clip_1", "prediction": "hello"}]).to_csv(
        predictions_path, index=False
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_hinglish_predictions.py",
            "--predictions",
            str(predictions_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "metrics.json"),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "duplicate clip IDs in manifest" in result.stderr


def test_empty_reference_is_rejected(tmp_path: Path) -> None:
    result = _run_eval(
        tmp_path,
        [{"clip_id": "clip_1", "prediction": "hello"}],
        [{"clip_id": "clip_1", "transcript": " ", "speaker_id": "spk_1"}],
    )
    assert result.returncode != 0
    assert "empty reference" in result.stderr


def test_failed_prediction_is_reported_separately(tmp_path: Path) -> None:
    result = _run_eval(
        tmp_path,
        [{"clip_id": "clip_1", "prediction": "", "status": "FAILED", "error": "timeout"}],
        [{"clip_id": "clip_1", "transcript": "hello", "speaker_id": "spk_1"}],
        ["--output", str(tmp_path / "metrics.json")],
    )
    assert result.returncode == 0, result.stderr
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["utterances"] == 0
    assert metrics["failed_prediction_count"] == 1
    assert metrics["failed_predictions"][0]["clip_id"] == "clip_1"


def test_score_prediction_rejects_empty_reference() -> None:
    from src.hinglish.evaluation import HinglishPrediction, score_prediction

    try:
        score_prediction(
            HinglishPrediction(
                clip_id="clip_1",
                speaker_id="spk_1",
                reference=" ",
                prediction="hello",
            )
        )
    except ValueError as exc:
        assert "reference transcript" in str(exc)
    else:
        raise AssertionError("expected empty reference to be rejected")

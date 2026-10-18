"""Tests for the Hinglish baseline metrics verification CLI.

All fixtures are synthetic (invented ASCII words only) and written to
``tmp_path``; the tests never read the private run archive and never use the
network. One test runs the real CLI in a subprocess to cover the entrypoint;
the remaining tests call ``main()`` in-process so the suite stays fast (the
repository's ``src.metrics`` package imports torch, which is expensive to
repeat per subprocess).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_hinglish_baseline_metrics.py"

PREDICTION_ROWS: list[dict[str, Any]] = [
    {"clip_id": "clip_a", "prediction": "alpha beta"},
    {"clip_id": "clip_b", "prediction": "gamma"},
    {"clip_id": "clip_c", "prediction": "one two three four"},
    {"clip_id": "clip_d", "prediction": "k x m n"},
]

MANIFEST_ROWS: list[dict[str, Any]] = [
    {
        "clip_id": "clip_a",
        "speaker_id": "spk_a",
        "reference_transcript": "alpha beta",
        "ratio_english_words": 0.5,
        "stratification_bucket": "hindi_dominant",
        "record_language_bucket": "hindi_dominant",
    },
    {
        "clip_id": "clip_b",
        "speaker_id": "spk_b",
        "reference_transcript": "gamma delta",
        "ratio_english_words": 0.1,
        "stratification_bucket": "hindi_dominant",
        "record_language_bucket": "mixed_code",
    },
    {
        "clip_id": "clip_c",
        "speaker_id": "spk_c",
        "reference_transcript": "one two three",
        "ratio_english_words": 0.9,
        "stratification_bucket": "mixed_code",
        "record_language_bucket": "mixed_code",
    },
    {
        "clip_id": "clip_d",
        "speaker_id": "spk_d",
        "reference_transcript": "k l m n",
        "ratio_english_words": 0.1,
        "stratification_bucket": "hindi_dominant",
        "record_language_bucket": "mixed_code",
    },
]


@pytest.fixture(scope="module")
def verifier() -> Any:
    """Import the verification CLI once for in-process main() calls."""
    spec = importlib.util.spec_from_file_location(
        "verify_hinglish_baseline_metrics_cli", VERIFY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _committed_per_clip_rows() -> list[dict[str, Any]]:
    # Rates are rounded to four decimals, mimicking the committed artifact.
    return [
        {
            "clip_id": "clip_a",
            "speaker_id": "spk_a",
            "wer": 0.0,
            "cer": 0.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "reference_words": 2,
            "hypothesis_words": 2,
            "ratio_bin": "code-switched",
            "ratio_english_words": 0.5,
            "stratification_bucket": "hindi_dominant",
            "hallucination_flagged": False,
        },
        {
            "clip_id": "clip_b",
            "speaker_id": "spk_b",
            "wer": 0.5,
            "cer": 0.5455,
            "substitutions": 0,
            "deletions": 1,
            "insertions": 0,
            "reference_words": 2,
            "hypothesis_words": 1,
            "ratio_bin": "hindi-dominant",
            "ratio_english_words": 0.1,
            "stratification_bucket": "hindi_dominant",
            "hallucination_flagged": False,
        },
        {
            "clip_id": "clip_c",
            "speaker_id": "spk_c",
            "wer": 0.3333,
            "cer": 0.3846,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 1,
            "reference_words": 3,
            "hypothesis_words": 4,
            "ratio_bin": "english-dominant",
            "ratio_english_words": 0.9,
            "stratification_bucket": "mixed_code",
            "hallucination_flagged": False,
        },
        {
            "clip_id": "clip_d",
            "speaker_id": "spk_d",
            "wer": 0.25,
            "cer": 0.1429,
            "substitutions": 1,
            "deletions": 0,
            "insertions": 0,
            "reference_words": 4,
            "hypothesis_words": 4,
            "ratio_bin": "hindi-dominant",
            "ratio_english_words": 0.1,
            "stratification_bucket": "hindi_dominant",
            "hallucination_flagged": False,
        },
    ]


def _committed_metrics(manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "test_fixture.v1",
        "aggregate": {
            "utterances": 4,
            "reference_words": 11,
            "reference_characters": 41,
            "total_edits": 3,
            "substitutions": 1,
            "deletions": 1,
            "insertions": 1,
            "corpus_wer": 0.2727,
            "corpus_cer": 0.2927,
            "mean_utterance_wer": 0.2708,
            "mean_utterance_cer": 0.2682,
            "normalization_version": "mendspeech.hinglish.v1",
        },
        "composition_bins": {
            "wer_code-switched": 0.0,
            "wer_hindi-dominant": 0.375,
            "wer_english-dominant": 0.3333333333333333,
        },
        "hallucination_diagnostics": {
            "likely_hallucination_clips": 0,
            "likely_hallucination_clip_ids": [],
            "insertions_from_flagged_clips": 0,
            "insertions_from_clean_clips": 1,
            "max_consecutive_repeat": 1,
            "max_repeated_phrase": 0,
            "corpus_wer_excluding_flagged": 0.2727,
            "corpus_wer_flagged_only": 0.0,
            "flagged_by_bin": {},
            "clean_clip_count_by_bin": {
                "code-switched": 1,
                "english-dominant": 1,
                "hindi-dominant": 2,
            },
        },
        "dataset": {
            "manifest_sha256": manifest_sha256,
            "manifest_clips": 4,
            "manifest_speakers": 4,
            "stratification_bucket_distribution": {
                "hindi_dominant": 3,
                "mixed_code": 1,
            },
            "record_level_bucket_distribution": {
                "hindi_dominant": 1,
                "mixed_code": 3,
            },
        },
        "per_clip": _committed_per_clip_rows(),
    }


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    """Write synthetic predictions, manifest and committed metrics."""
    predictions_path = tmp_path / "predictions.csv"
    manifest_path = tmp_path / "manifest.csv"
    metrics_path = tmp_path / "metrics.json"
    _write_csv(predictions_path, PREDICTION_ROWS)
    _write_csv(manifest_path, MANIFEST_ROWS)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metrics_path.write_text(
        json.dumps(_committed_metrics(digest), indent=2), encoding="utf-8"
    )
    return {
        "predictions": predictions_path,
        "manifest": manifest_path,
        "metrics": metrics_path,
    }


def _verifier_argv(paths: dict[str, Path], extra_args: list[str]) -> list[str]:
    return [
        str(VERIFY_SCRIPT),
        "--predictions",
        str(paths["predictions"]),
        "--manifest",
        str(paths["manifest"]),
        "--metrics",
        str(paths["metrics"]),
        *extra_args,
    ]


def _run_in_process(
    verifier: Any,
    paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str] | None = None,
) -> tuple[int, dict[str, Any], str]:
    monkeypatch.setattr(sys, "argv", _verifier_argv(paths, extra_args or []))
    try:
        code = verifier.main()
    except SystemExit as exc:  # pragma: no cover - only argparse errors
        code = int(exc.code) if exc.code is not None else 0
    captured = capsys.readouterr()
    report = json.loads(captured.out) if captured.out.strip() else {}
    return code, report, captured.err


def _run_subprocess(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *_verifier_argv(paths, [])],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def test_cli_entrypoint_passes_on_matching_fixture(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    result = _run_subprocess(paths)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["overall_match"] is True
    for name in (
        "aggregate",
        "per_clip",
        "composition_bins",
        "hallucination_diagnostics",
        "dataset",
    ):
        assert report["checks"][name]["match"] is True, name
    aggregate = report["recomputed"]["aggregate"]
    assert aggregate["wer"] == pytest.approx(3 / 11)
    assert aggregate["cer"] == pytest.approx(12 / 41)
    assert aggregate["substitutions"] == 1
    assert aggregate["deletions"] == 1
    assert aggregate["insertions"] == 1


def test_perturbed_aggregate_wer_fails(
    tmp_path: Path,
    verifier: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _build_fixture(tmp_path)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    metrics["aggregate"]["corpus_wer"] = 1.5
    paths["metrics"].write_text(json.dumps(metrics), encoding="utf-8")
    code, report, stderr = _run_in_process(verifier, paths, monkeypatch, capsys)
    assert code == 1
    assert report["overall_match"] is False
    assert report["checks"]["aggregate"]["match"] is False
    field = report["checks"]["aggregate"]["fields"]["corpus_wer"]
    assert field["match"] is False
    assert field["committed"] == 1.5
    assert "FAILED" in stderr


def test_perturbed_per_clip_wer_fails(
    tmp_path: Path,
    verifier: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _build_fixture(tmp_path)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    for row in metrics["per_clip"]:
        if row["clip_id"] == "clip_b":
            row["wer"] = 0.9
    paths["metrics"].write_text(json.dumps(metrics), encoding="utf-8")
    code, report, _ = _run_in_process(verifier, paths, monkeypatch, capsys)
    assert code == 1
    assert report["checks"]["aggregate"]["match"] is True
    per_clip = report["checks"]["per_clip"]
    assert per_clip["match"] is False
    assert per_clip["mismatches"][0]["clip_id"] == "clip_b"
    assert per_clip["mismatches"][0]["field"] == "wer"


def test_duplicate_prediction_clip_is_rejected(
    tmp_path: Path,
    verifier: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _build_fixture(tmp_path)
    duplicate_rows = list(PREDICTION_ROWS) + [
        {"clip_id": "clip_a", "prediction": "alpha beta"}
    ]
    _write_csv(paths["predictions"], duplicate_rows)
    code, _, stderr = _run_in_process(verifier, paths, monkeypatch, capsys)
    assert code != 0
    assert "duplicate clip IDs in predictions" in stderr


def test_unknown_prediction_clip_is_rejected(
    tmp_path: Path,
    verifier: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _build_fixture(tmp_path)
    unknown_rows = list(PREDICTION_ROWS) + [
        {"clip_id": "clip_zzz", "prediction": "alpha"}
    ]
    _write_csv(paths["predictions"], unknown_rows)
    code, _, stderr = _run_in_process(verifier, paths, monkeypatch, capsys)
    assert code != 0
    assert "unknown prediction clip IDs" in stderr
    assert "clip_zzz" in stderr


def test_bucket_tables_report_macro_and_weighted_rates(
    tmp_path: Path,
    verifier: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _build_fixture(tmp_path)
    code, report, _ = _run_in_process(verifier, paths, monkeypatch, capsys)
    assert code == 0
    tables = report["bucket_tables"]
    assert set(tables) == {
        "english_ratio_bin",
        "stratification_bucket",
        "record_language_bucket",
    }
    evaluation_rows = {
        row["bucket"]: row for row in tables["english_ratio_bin"]["rows"]
    }
    assert set(evaluation_rows) == {
        "code-switched",
        "hindi-dominant",
        "english-dominant",
    }
    hindi_dominant = evaluation_rows["hindi-dominant"]
    assert hindi_dominant["clips"] == 2
    assert hindi_dominant["macro_wer"] == pytest.approx(0.375)
    assert hindi_dominant["corpus_wer"] == pytest.approx(1 / 3)
    assert hindi_dominant["macro_wer"] != pytest.approx(hindi_dominant["corpus_wer"])
    assert hindi_dominant["macro_cer"] == pytest.approx((6 / 11 + 1 / 7) / 2)
    assert hindi_dominant["corpus_cer"] == pytest.approx(7 / 18)
    strat_rows = {
        row["bucket"]: row for row in tables["stratification_bucket"]["rows"]
    }
    assert strat_rows["hindi_dominant"]["clips"] == 3
    assert strat_rows["mixed_code"]["clips"] == 1
    assert sum(row["clips"] for row in tables["english_ratio_bin"]["rows"]) == 4


def test_raw_output_with_log_prefix_is_verified(
    tmp_path: Path,
    verifier: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _build_fixture(tmp_path)
    raw_rows = []
    for row in PREDICTION_ROWS:
        raw_rows.append(
            {
                "clip_id": row["clip_id"],
                "transcript": row["prediction"],
                "model_id": "openai/whisper-small",
                "task": "transcribe",
                "language": "hi",
                "precision": "float16",
                "runtime_sec": 0.5,
                "audio_duration_sec": 1.5,
            }
        )
    raw_path = tmp_path / "raw_output.json"
    raw_path.write_text(
        "Launching synthetic run...\n" + json.dumps(raw_rows), encoding="utf-8"
    )
    code, report, _ = _run_in_process(
        verifier, paths, monkeypatch, capsys, ["--raw-output", str(raw_path)]
    )
    assert code == 0
    assert report["checks"]["raw_output"]["match"] is True
    assert report["checks"]["raw_output"]["transcript_match"] is True


def test_raw_output_transcript_mismatch_fails(
    tmp_path: Path,
    verifier: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _build_fixture(tmp_path)
    raw_rows = [
        {
            "clip_id": row["clip_id"],
            "transcript": row["prediction"],
            "model_id": "openai/whisper-small",
            "task": "transcribe",
            "language": "hi",
            "precision": "float16",
            "runtime_sec": 0.5,
            "audio_duration_sec": 1.5,
        }
        for row in PREDICTION_ROWS
    ]
    raw_rows[1]["transcript"] = "different"
    raw_path = tmp_path / "raw_output.json"
    raw_path.write_text(
        "Launching synthetic run...\n" + json.dumps(raw_rows), encoding="utf-8"
    )
    code, report, _ = _run_in_process(
        verifier, paths, monkeypatch, capsys, ["--raw-output", str(raw_path)]
    )
    assert code == 1
    raw_check = report["checks"]["raw_output"]
    assert raw_check["match"] is False
    assert raw_check["mismatched_transcript_clip_ids"] == ["clip_b"]


def test_diagnostics_file_is_cross_checked(
    tmp_path: Path,
    verifier: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _build_fixture(tmp_path)
    diagnostics = {
        "records": [
            {
                "clip_id": row["clip_id"],
                "likely_repetitive_hallucination": False,
            }
            for row in PREDICTION_ROWS
        ],
        "summary": {
            "utterances": 4,
            "total_reference_words": 11,
            "total_edits": 3,
            "corpus_wer": 3 / 11,
            "total_insertions": 1,
            "likely_hallucination_clips": 0,
            "likely_hallucination_ids": [],
            "max_consecutive_repeat": 1,
            "max_repeated_phrase": 0,
        },
    }
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    code, report, _ = _run_in_process(
        verifier, paths, monkeypatch, capsys, ["--diagnostics", str(diagnostics_path)]
    )
    assert code == 0
    assert report["checks"]["diagnostics_file"]["match"] is True

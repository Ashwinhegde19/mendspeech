"""Baseline-immutability guards for the frozen Hinglish artifacts.

These locks make a silent overwrite of the frozen baseline artifacts
detectable in CI. The artifacts under ``results/hinglish/`` back the Phase
7.1 decoder probe (J1); if any byte of them changes, the guards fail and
someone must consciously update the embedded SHA-256 constants after
explaining the legitimate change.

Embedded constants were computed at test-authoring time (2026-10-18,
branch ``feat/hinglish-error-selection-v0``) from the byte content of the
committed files:

* ``validation_baseline_metrics.json``:
  ``79784de66c05bc808ff1e232140f4563320e02fc03551c0da907a6bdbe3452a0``
* ``failure_intelligence.json``:
  ``6787e4de929cf738175a2adacfc39bb580281dfad7a7bb50210d0307e34bc08f``
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_METRICS_PATH = REPO_ROOT / "results" / "hinglish" / "validation_baseline_metrics.json"
FAILURE_INTELLIGENCE_PATH = REPO_ROOT / "results" / "hinglish" / "failure_intelligence.json"

# Byte-level locks (see module docstring). Update only with a deliberate,
# explained change to the committed artifact.
BASELINE_METRICS_SHA256 = "79784de66c05bc808ff1e232140f4563320e02fc03551c0da907a6bdbe3452a0"
FAILURE_INTELLIGENCE_SHA256 = "6787e4de929cf738175a2adacfc39bb580281dfad7a7bb50210d0307e34bc08f"

EXPECTED_SCHEMA_VERSION = "mendspeech.hinglish_validation_baseline.v2"
EXPECTED_CORPUS_WER = 2.9676
EXPECTED_CORPUS_CER = 2.8391
EXPECTED_SUBSTITUTIONS = 210
EXPECTED_DELETIONS = 19
EXPECTED_INSERTIONS = 777
EXPECTED_PER_CLIP_COUNT = 30
WERT_TOLERANCE = 0.01

INTELLIGENCE_PER_CLIP_KEYS = (
    "clip_id",
    "failure_class",
    "insertions",
    "has_word_loop",
    "has_char_loop",
    "heuristic_flag",
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _collect_keys(value: Any) -> List[str]:
    """Collect every dictionary key inside a nested structure.

    Args:
        value: Any JSON-ish value.

    Returns:
        List of keys found anywhere in the structure.
    """
    found: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.append(key)
            found.extend(_collect_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_keys(item))
    return found


@pytest.fixture(scope="module")
def baseline_metrics() -> Dict[str, Any]:
    """Parsed frozen baseline metrics (the artifact itself is the fixture)."""
    assert BASELINE_METRICS_PATH.is_file(), (
        f"the frozen baseline artifact must exist at {BASELINE_METRICS_PATH}; "
        "its disappearance is a broken guard, not a skip"
    )
    return json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def intelligence() -> Dict[str, Any]:
    """Parsed failure intelligence payload."""
    assert FAILURE_INTELLIGENCE_PATH.is_file(), (
        f"the committed failure-intelligence artifact must exist at "
        f"{FAILURE_INTELLIGENCE_PATH}"
    )
    return json.loads(FAILURE_INTELLIGENCE_PATH.read_text(encoding="utf-8"))


def test_baseline_metrics_bytes_are_locked() -> None:
    """The committed baseline JSON is byte-for-byte identical to the lock."""
    assert _sha256(BASELINE_METRICS_PATH) == BASELINE_METRICS_SHA256


def test_baseline_metrics_schema_and_aggregate(baseline_metrics: Dict[str, Any]) -> None:
    """Schema version, corpus aggregate, and normalizer identity are frozen."""
    assert baseline_metrics["schema_version"] == EXPECTED_SCHEMA_VERSION
    aggregate = baseline_metrics["aggregate"]
    assert aggregate["corpus_wer"] == pytest.approx(EXPECTED_CORPUS_WER, abs=WERT_TOLERANCE)
    assert aggregate["corpus_cer"] == pytest.approx(EXPECTED_CORPUS_CER, abs=WERT_TOLERANCE)
    assert aggregate["utterances"] == EXPECTED_PER_CLIP_COUNT
    assert aggregate["substitutions"] == EXPECTED_SUBSTITUTIONS
    assert aggregate["deletions"] == EXPECTED_DELETIONS
    assert aggregate["insertions"] == EXPECTED_INSERTIONS
    assert aggregate["normalization_version"] == "mendspeech.hinglish.v1"


def test_baseline_metrics_config_is_the_frozen_run(
    baseline_metrics: Dict[str, Any],
) -> None:
    """The pinned inference configuration matches the J1 baseline config."""
    config = baseline_metrics["inference_config"]
    assert config["model_id"] == "openai/whisper-small"
    assert config["model_revision"] == "973afd24965f72e36ca33b3055d56a652f456b4d"
    assert config["language"] == "hi"
    assert config["task"] == "transcribe"
    assert config["precision"] == "float16"


def test_baseline_metrics_per_clip_records(baseline_metrics: Dict[str, Any]) -> None:
    """Exactly 30 unique per-clip records that sum to the frozen totals."""
    per_clip = baseline_metrics["per_clip"]
    assert len(per_clip) == EXPECTED_PER_CLIP_COUNT
    clip_ids = [record["clip_id"] for record in per_clip]
    assert len(set(clip_ids)) == EXPECTED_PER_CLIP_COUNT
    for key, expected_total in (
        ("substitutions", EXPECTED_SUBSTITUTIONS),
        ("deletions", EXPECTED_DELETIONS),
        ("insertions", EXPECTED_INSERTIONS),
    ):
        assert sum(int(record[key]) for record in per_clip) == expected_total


def test_baseline_metrics_privacy_note_and_no_transcript_keys(
    baseline_metrics: Dict[str, Any],
) -> None:
    """A privacy note exists and the artifact carries no transcript keys."""
    note = baseline_metrics["privacy_note"]
    assert isinstance(note, str) and note.strip()
    assert "Reference transcripts" in note
    assert "private run archive" in note
    keys = _collect_keys(baseline_metrics)
    for forbidden in (
        "reference_transcript",
        "hypothesis_transcript",
        "transcript",
        "prediction_text",
    ):
        assert forbidden not in keys, f"artifact unexpectedly contains key {forbidden!r}"


def test_failure_intelligence_bytes_are_locked() -> None:
    """The committed intelligence JSON is byte-for-byte identical to the lock."""
    assert _sha256(FAILURE_INTELLIGENCE_PATH) == FAILURE_INTELLIGENCE_SHA256


def test_failure_intelligence_consistency_marker(intelligence: Dict[str, Any]) -> None:
    """The intelligence self-consistency marker is present and passing."""
    consistency = intelligence["consistency"]
    assert consistency["checked"] is True
    assert consistency["match"] is True
    assert consistency["clip_count"] == EXPECTED_PER_CLIP_COUNT
    per_clip = intelligence["per_clip"]
    assert len(per_clip) == EXPECTED_PER_CLIP_COUNT
    for record in per_clip:
        for key in INTELLIGENCE_PER_CLIP_KEYS:
            assert key in record
        assert isinstance(record["insertions"], int)
        assert isinstance(record["has_word_loop"], bool)
        assert isinstance(record["has_char_loop"], bool)

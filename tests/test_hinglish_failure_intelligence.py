"""Tests for Hinglish failure intelligence classification and reporting.

All fixtures are synthetic. The tests assert that report outputs carry clip
IDs, counts, and evidence text only — never transcript content.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.hinglish.failure_intelligence import (
    CLASS_EARLY_REPETITION,
    CLASS_GENERATION_CAP_REPETITION,
    CLASS_GENERATION_CAP_UNEXPLAINED,
    CLASS_NONE,
    CLASS_RECOGNITION,
    CLASS_SCRIPT_MISMATCH,
    CONFIDENCE_ESTIMATE,
    CONFIDENCE_HEURISTIC,
    CONFIDENCE_STRUCTURAL,
    DOMINANCE_REPETITIVE_GENERATION,
    ClipErrorProfile,
    GenerationFailureSignals,
    build_failure_intelligence_payload,
    classify_failure,
    render_markdown_report,
    script_mismatch_statistics,
    summarize_failure_attribution,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI = REPO_ROOT / "scripts" / "report_hinglish_failure_intelligence.py"

MARKER = "zzmarkerword"


def _profile(
    clip_id: str = "clip_a",
    *,
    reference: str = "alpha beta gamma",
    hypothesis: str = "alpha beta gamma",
    substitutions: int = 0,
    deletions: int = 0,
    insertions: int = 0,
    wer: float = 0.0,
    cap_hit: bool = False,
    has_word_loop: bool = False,
    has_char_loop: bool = False,
    flagged: bool = False,
    loop_coverage_words: int = 0,
) -> ClipErrorProfile:
    return ClipErrorProfile(
        clip_id=clip_id,
        speaker_id="speaker_a",
        reference_words=len(reference.split()),
        hypothesis_words=len(hypothesis.split()),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        wer=wer,
        english_ratio_bin="hindi-dominant",
        script=script_mismatch_statistics(reference, hypothesis),
        generation=GenerationFailureSignals(
            cap_hit=cap_hit,
            has_word_loop=has_word_loop,
            has_char_loop=has_char_loop,
            has_repetition_evidence=has_word_loop or has_char_loop,
            heuristic_flag=flagged,
            loop_coverage_words=loop_coverage_words,
            output_words=len(hypothesis.split()),
        ),
    )


# ---------------------------------------------------------------------------
# Script mismatch statistics
# ---------------------------------------------------------------------------


def test_script_mismatch_counts_latin_to_devanagari_replacement() -> None:
    stats = script_mismatch_statistics("login करें", "लॉगिन करें")
    assert stats.replaced_script_mismatch == 1
    assert stats.deleted_latin_tokens == 0
    assert stats.script_mismatch_edits == 1


def test_script_mismatch_ignores_same_script_substitution() -> None:
    stats = script_mismatch_statistics("alpha beta", "alpha delta")
    assert stats.replaced_script_mismatch == 0
    assert stats.replaced_same_script == 1
    assert stats.script_mismatch_edits == 0


def test_deleted_latin_tokens_counted() -> None:
    stats = script_mismatch_statistics("password भूल गया", "भूल गया")
    assert stats.deleted_latin_tokens == 1
    assert stats.script_mismatch_edits == 1


def test_pure_insertion_is_not_a_script_mismatch() -> None:
    stats = script_mismatch_statistics("alpha", "alpha alpha alpha")
    assert stats.inserted_extra_tokens == 2
    assert stats.script_mismatch_edits == 0


def test_mixed_replace_block_splits_remainder_into_deletions() -> None:
    stats = script_mismatch_statistics("one two three", "एक दो")
    assert stats.replaced_script_mismatch == 2
    assert stats.deleted_latin_tokens == 1
    assert stats.script_mismatch_edits == 3


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_cap_hit_with_word_loop_is_structural() -> None:
    classification = classify_failure(
        _profile(cap_hit=True, has_word_loop=True, insertions=40, wer=7.0)
    )
    assert classification.failure_class == CLASS_GENERATION_CAP_REPETITION
    assert classification.confidence == CONFIDENCE_STRUCTURAL


def test_cap_hit_with_char_loop_is_structural() -> None:
    classification = classify_failure(
        _profile(cap_hit=True, has_char_loop=True, insertions=1, wer=0.4)
    )
    assert classification.failure_class == CLASS_GENERATION_CAP_REPETITION
    assert classification.confidence == CONFIDENCE_STRUCTURAL


def test_cap_hit_without_loop_evidence_is_undetermined_content() -> None:
    classification = classify_failure(_profile(cap_hit=True))
    assert classification.failure_class == CLASS_GENERATION_CAP_UNEXPLAINED
    assert classification.confidence == CONFIDENCE_HEURISTIC


def test_early_repetition_below_ceiling() -> None:
    classification = classify_failure(
        _profile(has_word_loop=True, flagged=True, insertions=5, wer=1.2)
    )
    assert classification.failure_class == CLASS_EARLY_REPETITION
    assert classification.confidence == CONFIDENCE_HEURISTIC


def test_script_mismatch_dominant_classification() -> None:
    profile = _profile(
        reference="meeting tomorrow login करें",
        hypothesis="मीटिंग कल लॉगिन करें",
        substitutions=2,
        wer=0.8,
    )
    classification = classify_failure(profile)
    assert profile.script.script_mismatch_edits == 3
    assert classification.failure_class == CLASS_SCRIPT_MISMATCH
    assert classification.confidence == CONFIDENCE_ESTIMATE


def test_single_script_edit_is_not_script_dominant() -> None:
    classification = classify_failure(
        _profile(
            reference="report भेज दो भईया",
            hypothesis="रिपोर्ट भेज दो भैया",
            substitutions=2,
            wer=0.5,
        )
    )
    assert classification.failure_class == CLASS_RECOGNITION
    assert classification.confidence == CONFIDENCE_HEURISTIC


def test_no_failure_classification() -> None:
    classification = classify_failure(_profile())
    assert classification.failure_class == CLASS_NONE


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def _attribution_fixture():
    profiles = [
        _profile("clip_cap", cap_hit=True, has_word_loop=True, insertions=60, wer=6.0),
        _profile(
            "clip_script",
            reference="login करें",
            hypothesis="लॉगिन करें",
            substitutions=1,
            wer=0.5,
        ),
        _profile(
            "clip_recog",
            reference="alpha beta gamma delta",
            hypothesis="alpha beta",
            deletions=2,
            wer=0.5,
        ),
        _profile("clip_clean"),
    ]
    classifications = {
        profile.clip_id: classify_failure(profile) for profile in profiles
    }
    return profiles, classifications


def test_attribution_repetitive_dominance_and_exclusions() -> None:
    profiles, classifications = _attribution_fixture()
    attribution = summarize_failure_attribution(profiles, classifications)
    assert attribution.cap_hit_clip_count == 1
    assert attribution.insertions_from_generation_side == 60
    assert attribution.insertions_generation_side_share == 1.0
    assert attribution.dominant_error_type == DOMINANCE_REPETITIVE_GENERATION
    # cap clip: 60 of 63 total edits; exclusion changes the corpus rate
    assert attribution.corpus_wer == pytest.approx(63 / 12)
    assert attribution.corpus_wer_excluding_cap_hits == pytest.approx(3 / 9)
    assert (
        attribution.corpus_wer_excluding_generation_side
        == pytest.approx(3 / 9)
    )


def test_attribution_requires_all_classifications() -> None:
    profiles, classifications = _attribution_fixture()
    classifications.pop("clip_clean")
    with pytest.raises(KeyError):
        summarize_failure_attribution(profiles, classifications)


# ---------------------------------------------------------------------------
# Payload and markdown
# ---------------------------------------------------------------------------


def _payload_fixture():
    profiles, classifications = _attribution_fixture()
    return build_failure_intelligence_payload(
        profiles,
        classifications,
        {"predictions_sha256": "0" * 64, "manifest_sha256": "1" * 64},
    )


def test_payload_is_deterministic() -> None:
    assert _payload_fixture() == _payload_fixture()


def test_payload_and_markdown_never_contain_transcript_text() -> None:
    payload = _payload_fixture()
    serialized = json.dumps(payload, ensure_ascii=False)
    markdown = render_markdown_report(payload)
    assert MARKER not in serialized
    assert MARKER not in markdown
    # fixture reference tokens must not leak either
    for needle in ("alpha", "beta", "gamma", "login"):
        assert needle not in serialized


def test_markdown_covers_all_seven_questions() -> None:
    markdown = render_markdown_report(_payload_fixture())
    for heading in (
        "## 1. Which utterances failed?",
        "## 2. What type of error dominates?",
        "## 3. Failure classes",
        "## 4. Evidence and confidence per clip",
        "## 5. What is still uncertain?",
        "## 6. Which training examples might be valuable?",
        "## 7. How will we evaluate whether selection helped?",
    ):
        assert heading in markdown
    assert "mendspeech.hinglish.v1" in markdown


def test_ranked_training_candidates_exclude_generation_side_clips() -> None:
    payload = _payload_fixture()
    ranked = payload["questions"]["training_value_hypotheses"]["ranked_candidates"]
    clip_ids = {item["clip_id"] for item in ranked}
    assert "clip_script" in clip_ids
    assert "clip_recog" in clip_ids
    assert "clip_cap" not in clip_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_cli_fixtures(tmp_path: Path, metrics: Path | None = None):
    manifest_path = tmp_path / "manifest.csv"
    predictions_path = tmp_path / "predictions.csv"
    inspection_path = tmp_path / "inspection.json"
    manifest_path.write_text(
        "clip_id,speaker_id,reference_transcript,ratio_english_words\n"
        "clip_clean,spk_a,alpha beta gamma delta epsilon,0.3\n"
        "clip_cap,spk_b,alpha beta gamma delta,0.3\n"
        f"clip_marker,spk_a,{MARKER} token token token token,0.3\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "clip_id,prediction\n"
        "clip_clean,alpha beta gamma delta epsilon\n"
        "clip_cap," + " ".join(["beta"] * 10) + "\n"
        f"clip_marker,{MARKER} token token\n",
        encoding="utf-8",
    )
    inspection_path.write_text(
        json.dumps(
            {
                "flagged_clip_ids": ["clip_cap"],
                "runtime_ceiling": {"members": ["clip_cap"]},
                "per_clip": [
                    {
                        "clip_id": "clip_clean",
                        "flagged": False,
                        "has_word_loop": False,
                        "has_char_loop": False,
                        "has_repetition_evidence": False,
                        "loop_region_coverage_words": 0,
                        "output_words": 5,
                    },
                    {
                        "clip_id": "clip_cap",
                        "flagged": True,
                        "has_word_loop": True,
                        "has_char_loop": False,
                        "has_repetition_evidence": True,
                        "loop_region_coverage_words": 8,
                        "output_words": 10,
                    },
                    {
                        "clip_id": "clip_marker",
                        "flagged": False,
                        "has_word_loop": False,
                        "has_char_loop": False,
                        "has_repetition_evidence": False,
                        "loop_region_coverage_words": 0,
                        "output_words": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, predictions_path, inspection_path


def test_cli_writes_reports_and_keeps_them_safe(tmp_path: Path) -> None:
    manifest_path, predictions_path, inspection_path = _write_cli_fixtures(tmp_path)
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--predictions",
            str(predictions_path),
            "--manifest",
            str(manifest_path),
            "--inspection-json",
            str(inspection_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["questions"]["which_utterances_failed"]["clip_count"] == 3
    text = output_json.read_text(encoding="utf-8") + output_md.read_text(
        encoding="utf-8"
    )
    assert MARKER not in text
    assert "alpha beta gamma" not in text
    assert payload["consistency"]["checked"] is False


def test_cli_refuses_inconsistent_inputs(tmp_path: Path) -> None:
    manifest_path, predictions_path, inspection_path = _write_cli_fixtures(tmp_path)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "per_clip": [
                    {"clip_id": "clip_clean", "wer": 9.9, "substitutions": 5,
                     "deletions": 0, "insertions": 0},
                ]
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--predictions",
            str(predictions_path),
            "--manifest",
            str(manifest_path),
            "--inspection-json",
            str(inspection_path),
            "--metrics",
            str(metrics_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 2
    assert not output_json.exists()
    assert "disagree" in completed.stderr

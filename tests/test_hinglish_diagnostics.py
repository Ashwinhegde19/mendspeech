from src.hinglish.diagnostics import (
    analyze_repetition,
    likely_repetitive_hallucination,
    script_word_counts,
    summarize_diagnostics,
    diagnose_prediction,
)
from src.hinglish.evaluation import HinglishPrediction


def test_repetition_finds_exact_repeated_word() -> None:
    finding = analyze_repetition(["a", "b", "a", "b"])
    assert finding.max_consecutive_repeat == 1
    assert finding.longest_repeated_phrase == 2
    assert finding.repeated_phrase_count == 1


def test_likely_hallucination_requires_insertions() -> None:
    assert likely_repetitive_hallucination(
        reference_words=5,
        prediction_words=30,
        insertion_count=20,
        max_consecutive_repeat=1,
        longest_repeated_phrase=1,
    )


def test_script_counts_separate_devanagari_and_latin() -> None:
    counts = script_word_counts("hello नमस्ते test")
    assert counts["latin_script_words"] == 2
    assert counts["devanagari_words"] == 1


def test_diagnosis_uses_existing_score() -> None:
    diagnostic = diagnose_prediction(
        HinglishPrediction(
            clip_id="clip_1",
            speaker_id="spk_1",
            reference="hello नमस्ते",
            prediction="hello hallucination hallucination world",
        )
    )
    assert diagnostic["clip_id"] == "clip_1"
    assert diagnostic["insertions"] > 0
    assert "likely_repetitive_hallucination" in diagnostic


def test_summary_reports_hallucination_ids() -> None:
    diagnostics = [
        {
            "clip_id": "clip_1",
            "reference_words": 2,
            "total_edits": 10,
            "insertions": 8,
            "max_consecutive_repeat": 4,
            "longest_repeated_phrase": 1,
            "likely_repetitive_hallucination": True,
        }
    ]
    summary = summarize_diagnostics(diagnostics)
    assert summary["likely_hallucination_clips"] == 1
    assert summary["likely_hallucination_ids"] == ["clip_1"]

def test_ordinary_repeated_phrase_without_insertions_is_not_hallucination() -> None:
    assert not likely_repetitive_hallucination(
        reference_words=4,
        prediction_words=4,
        insertion_count=0,
        max_consecutive_repeat=2,
        longest_repeated_phrase=2,
    )



def test_diagnostic_script_requires_arguments() -> None:
    import subprocess
    import sys
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "diagnose_hinglish_baseline.py"
    )
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "required" in (result.stderr + result.stdout).lower()

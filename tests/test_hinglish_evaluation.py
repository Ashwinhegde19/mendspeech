"""Unit tests for Hinglish result scoring."""

from src.hinglish.evaluation import (
    HinglishPrediction,
    score_prediction,
    summarize_predictions,
)


def test_prediction_scoring_reports_wer_and_code_switch_bin() -> None:
    score = score_prediction(
        HinglishPrediction(
            clip_id="clip_1",
            speaker_id="spk_1",
            reference="hello नमस्ते",
            prediction="hello",
            ratio_english_words=0.8,
        )
    )
    assert 0 <= score["wer"] <= 1
    assert score["english_ratio_bin"] == "english-dominant"


def test_prediction_summary_aggregates_by_code_switch_bin() -> None:
    summary = summarize_predictions(
        [
            HinglishPrediction("a", "s1", "a b", "a b", 0.2),
            HinglishPrediction("b", "s2", "c d", "x d", 0.8),
        ]
    )
    assert summary["utterances"] == 2
    assert "bins" in summary

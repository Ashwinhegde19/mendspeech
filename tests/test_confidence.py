"""Tests for confidence definitions and label alignment (Day 11)."""

import pytest

from src.asr.confidence import (
    DELETED,
    INSERTED,
    MATCHED,
    VALID,
    AlignmentResult,
    ConfidenceProvenance,
    TokenScore,
    WordScore,
    align_tokens_to_reference,
    confidence_summary,
    mean_frame_confidence,
    mean_token_confidence,
    min_token_confidence,
)

PROVENANCE = ConfidenceProvenance(
    model="WAV2VEC2_ASR_BASE_960H",
    head="ctc",
    tokenizer="chars",
    decoder="greedy_ctc",
    precision="float32",
)


def token(text: str, probability: float, index: int = 0) -> TokenScore:
    """Build a TokenScore for a test."""
    return TokenScore(
        token=text,
        index=index,
        probability=probability,
        frame_index=index,
        timestamp_sec=round(index * 0.02, 4),
        blank_before=True,
    )


class TestFrameVersusTokenConfidence:
    """Concept 1: frame mean and token mean are different quantities."""

    def test_frame_mean_includes_every_frame(self):
        assert mean_frame_confidence([0.9, 0.8, 0.7]) == pytest.approx(0.8)

    def test_token_mean_uses_emitted_tokens_only(self):
        scores = [token("A", 0.5, 0), token("B", 0.7, 1)]
        assert mean_token_confidence(scores) == pytest.approx(0.6)

    def test_blanks_lower_the_frame_mean_below_token_mean(self):
        """The measured Day 8 clip: 150 blank frames out of 292."""
        frames = [0.99] * 142 + [0.90] * 150
        scores = [token("A", 0.99, 0)]
        assert mean_frame_confidence(frames) < mean_token_confidence(scores)

    def test_the_two_means_differ_on_blank_heavy_audio(self):
        frames = [0.99] * 100 + [0.90] * 100
        scores = [token("A", 0.99, 0), token("B", 0.99, 1)]
        assert mean_frame_confidence(frames) != pytest.approx(
            mean_token_confidence(scores)
        )

    def test_empty_inputs_return_zero(self):
        assert mean_frame_confidence([]) == 0.0
        assert mean_token_confidence([]) == 0.0

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            mean_frame_confidence([0.5, 1.4])

    def test_min_token_confidence_finds_the_weakest_word(self):
        scores = [token("A", 0.99, 0), token("B", 0.40, 1), token("C", 0.98, 2)]
        assert min_token_confidence(scores) == pytest.approx(0.40)

    def test_mean_hides_a_single_bad_word_that_min_exposes(self):
        """The case that motivates reporting both."""
        scores = [token("A", 0.99, i) for i in range(9)] + [token("BAD", 0.10, 9)]
        assert mean_token_confidence(scores) > 0.85
        assert min_token_confidence(scores) == pytest.approx(0.10)

    def test_min_returns_none_when_nothing_emitted(self):
        assert min_token_confidence([]) is None


class TestConfidenceSummary:
    """Both means reported together, with provenance attached."""

    def test_reports_gap_and_provenance(self):
        frames = [0.99] * 100 + [0.90] * 100
        scores = [token("A", 0.99, 0)]
        summary = confidence_summary(frames, scores, PROVENANCE)
        assert summary["num_frames"] == 200
        assert summary["num_tokens"] == 1
        assert summary["provenance"]["model"] == "WAV2VEC2_ASR_BASE_960H"
        assert summary["provenance"]["head"] == "ctc"

    def test_gap_is_frame_mean_minus_token_mean(self):
        summary = confidence_summary([0.5, 0.5], [token("A", 1.0, 0)], PROVENANCE)
        assert summary["frame_token_gap"] == pytest.approx(-0.5)

    def test_empty_emission_gives_none_minimum(self):
        summary = confidence_summary([0.9, 0.9], [], PROVENANCE)
        assert summary["min_token_confidence"] is None
        assert summary["mean_token_confidence"] == 0.0


class TestProvenance:
    """A score is meaningless without knowing what produced it."""

    def test_to_dict_has_every_field(self):
        assert set(PROVENANCE.to_dict()) == {
            "model", "head", "tokenizer", "decoder", "precision",
        }

    def test_provenance_is_immutable(self):
        with pytest.raises(Exception):
            PROVENANCE.model = "other"  # type: ignore[misc]


class TestAlignTokensToReference:
    """Label alignment and the valid/missing states."""

    def test_perfect_alignment_marks_everything_matched(self):
        scores = [token("I", 0.9, 0), token("HAVE", 0.9, 1), token("TIME", 0.9, 2)]
        result = align_tokens_to_reference(scores, ["I", "HAVE", "TIME"])
        assert result.num_matched_words == 3
        assert result.num_deleted_words == 0
        assert all(w.state == MATCHED for w in result.word_scores)

    def test_substitution_is_matched_not_missing(self):
        """The reference word was attempted, just realized wrongly."""
        result = align_tokens_to_reference([token("HAD", 0.55, 0)], ["HAVE"])
        assert result.word_scores[0].state == MATCHED
        assert result.word_scores[0].token_probability == pytest.approx(0.55)

    def test_deleted_word_carries_none_not_zero(self):
        """A missed word is absent, not a zero-confidence word."""
        result = align_tokens_to_reference(
            [token("I", 0.9, 0)], ["I", "HAVE", "TIME"]
        )
        deleted = [w for w in result.word_scores if w.state == DELETED]
        assert len(deleted) == 2
        for word in deleted:
            assert word.token_probability is None
            assert word.token_count == 0

    def test_inserted_token_marked(self):
        result = align_tokens_to_reference(
            [token("I", 0.9, 0), token("EXTRA", 0.6, 1)], ["I"]
        )
        assert result.num_inserted >= 1
        assert any(s.state == INSERTED for s in result.token_scores)

    def test_word_separator_is_not_a_word(self):
        scores = [
            token("I", 0.9, 0), token("|", 0.9, 1), token("HAVE", 0.9, 2),
        ]
        result = align_tokens_to_reference(scores, ["I", "HAVE"])
        assert result.num_matched_words == 2
        assert result.num_inserted == 0
        assert all(s.token != "|" for s in result.token_scores)

    def test_comparison_is_case_insensitive(self):
        assert align_tokens_to_reference([token("her", 0.9, 0)], ["HER"]).num_matched_words == 1

    def test_empty_hypothesis_deletes_every_word(self):
        result = align_tokens_to_reference([], ["A", "B"])
        assert result.num_deleted_words == 2
        assert result.num_matched_words == 0
        assert isinstance(result, AlignmentResult)

    def test_empty_reference_leaves_tokens_inserted(self):
        result = align_tokens_to_reference([token("A", 0.9, 0)], [])
        assert result.num_inserted == 1
        assert result.num_deleted_words == 0

    def test_both_empty(self):
        assert align_tokens_to_reference([], []).to_dict() == {
            "num_valid_tokens": 0,
            "num_inserted": 0,
            "num_matched_words": 0,
            "num_deleted_words": 0,
        }

    def test_repeated_word_aligns_without_double_counting(self):
        result = align_tokens_to_reference(
            [token("NO", 0.9, 0), token("NO", 0.8, 1)], ["NO", "NO"]
        )
        assert result.num_matched_words == 2
        assert [w.token_count for w in result.word_scores] == [1, 1]

    def test_inputs_are_not_mutated(self):
        scores = [token("A", 0.9, 0)]
        reference = ["B"]
        align_tokens_to_reference(scores, reference)
        assert scores[0].state == VALID
        assert reference == ["B"]

    def test_word_score_to_dict(self):
        result = align_tokens_to_reference([token("A", 0.9, 0)], ["A"])
        word = result.word_scores[0]
        assert isinstance(word, WordScore)
        assert set(word.to_dict()) == {
            "word", "index", "state", "token_probability", "token_count",
        }


class TestDifferentSystemsAreDistinguishable:
        other = ConfidenceProvenance(
            model="OTHER", head="ctc", tokenizer="chars",
            decoder="greedy_ctc", precision="float32",
        )
        assert PROVENANCE != other

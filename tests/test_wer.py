"""Tests for word and character error rates (Day 10, concept 1)."""

import math

import pytest

from src.metrics.wer import (
    EMPTY_REFERENCE_RATE,
    ErrorRate,
    character_error_rate,
    word_error_rate,
)


class TestWordErrorRate:
    """WER = Levenshtein distance over words / reference word count."""

    def test_identical_text_scores_zero(self):
        rate = word_error_rate("the quick brown fox", "the quick brown fox")
        assert isinstance(rate, ErrorRate)
        assert rate.metric == "wer"
        assert rate.error_rate == 0.0
        assert rate.total_edits == 0
        assert rate.ref_len == 4
        assert rate.hyp_len == 4

    def test_single_substitution(self):
        rate = word_error_rate("the cat sat", "the bat sat")
        assert rate.total_edits == 1
        assert rate.ref_len == 3
        assert rate.error_rate == pytest.approx(1 / 3)

    def test_deleted_word(self):
        rate = word_error_rate("a b c d", "a d")
        assert rate.total_edits == 2
        assert rate.error_rate == pytest.approx(0.5)

    def test_inserted_word(self):
        rate = word_error_rate("a d", "a b c d")
        assert rate.total_edits == 2
        assert rate.error_rate == pytest.approx(1.0)

    def test_empty_hypothesis_is_total_error(self):
        rate = word_error_rate("a b c", "")
        assert rate.total_edits == 3
        assert rate.hyp_len == 0
        assert rate.error_rate == pytest.approx(1.0)

    def test_empty_both_strings_score_zero(self):
        assert word_error_rate("", "").error_rate == 0.0

    def test_empty_reference_with_content_is_sentinel(self):
        rate = word_error_rate("", "a b")
        assert rate.error_rate == EMPTY_REFERENCE_RATE
        assert math.isinf(rate.error_rate)

    def test_leading_deletion_not_charged_per_token(self):
        """A true alignment charges one edit, not one per following token.

        Positional comparison would report all three words as differing.
        """
        rate = word_error_rate("the summer months", "summer months")
        assert rate.total_edits == 1
        assert rate.error_rate == pytest.approx(1 / 3)
        positional = sum(
            1 for a, b in zip("the summer months".split(), "summer months".split()) if a != b
        )
        assert positional == 2
        assert rate.total_edits < positional

    def test_case_sensitive_until_normalized(self):
        """Documents that the module does not lowercase its inputs."""
        assert word_error_rate("The cat", "the cat").error_rate > 0.0

    def test_rejects_non_string(self):
        with pytest.raises(ValueError, match="must both be str"):
            word_error_rate(["the", "cat"], "the cat")


class TestCharacterErrorRate:
    """CER = Levenshtein distance over characters / reference length."""

    def test_identical_text_scores_zero(self):
        rate = character_error_rate("abcd", "abcd")
        assert rate.metric == "cer"
        assert rate.ref_len == 4
        assert rate.error_rate == 0.0

    def test_single_substitution(self):
        rate = character_error_rate("abcd", "abxd")
        assert rate.total_edits == 1
        assert rate.error_rate == pytest.approx(0.25)

    def test_empty_both_strings_score_zero(self):
        assert character_error_rate("", "").error_rate == 0.0

    def test_empty_reference_with_content_is_sentinel(self):
        assert math.isinf(character_error_rate("", "abc").error_rate)

    def test_spaces_are_scored_as_characters(self):
        rate = character_error_rate("a b", "ab")
        assert rate.total_edits == 1
        assert rate.ref_len == 3


class TestRateBehaviour:
    """Properties the two scorers must share."""

    def test_cer_lower_than_wer_for_whole_word_swaps(self):
        """A swapped word costs one WER unit but several CER units."""
        ref = "THE SUMMER MONTHS PLIED UP"
        hyp = "THE SUMMER MONTHS QUIED UP"
        assert character_error_rate(ref, hyp).error_rate < word_error_rate(ref, hyp).error_rate

    def test_rates_are_fractions_not_percentages(self):
        rate = word_error_rate("a b c d", "x b c d")
        assert 0.0 < rate.error_rate <= 1.0

    def test_to_dict_is_flat_and_csv_ready(self):
        row = word_error_rate("a b", "a c").to_dict()
        assert set(row) == {
            "metric",
            "error_rate",
            "total_edits",
            "ref_len",
            "hyp_len",
        }
        assert all(isinstance(v, (int, float, str)) for v in row.values())

    def test_repeated_calls_are_deterministic(self):
        ref, hyp = "a b x d e", "a c d e z"
        assert word_error_rate(ref, hyp) == word_error_rate(ref, hyp)
        assert character_error_rate(ref, hyp) == character_error_rate(ref, hyp)

    def test_inputs_are_not_mutated(self):
        ref, hyp = "a b c", "a x c"
        word_error_rate(ref, hyp)
        character_error_rate(ref, hyp)
        assert (ref, hyp) == ("a b c", "a x c")

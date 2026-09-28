"""Tests for word and character error rates (Day 10, concept 1)."""

import math

import pytest

from src.metrics.wer import (
    EMPTY_REFERENCE_RATE,
    EditCounts,
    ErrorRate,
    character_error_rate,
    levenshtein_counts,
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
            "substitutions",
            "deletions",
            "insertions",
        }
        assert all(isinstance(v, (int, float, str)) for v in row.values())

    def test_repeated_calls_are_deterministic(self):
        ref, hyp = "a b x d e", "a c d e z"
        assert word_error_rate(ref, hyp) == word_error_rate(ref, hyp)
        assert character_error_rate(ref, hyp) == character_error_rate(ref, hyp)



class TestEditCounts:
    """Substitution, deletion and insertion counts from one alignment."""

    def test_identical_sequences_have_no_edits(self):
        counts = levenshtein_counts(["the", "cat"], ["the", "cat"])
        assert counts.substitutions == 0
        assert counts.deletions == 0
        assert counts.insertions == 0
        assert counts.total_edits == 0

    def test_substitution_is_not_counted_as_delete_plus_insert(self):
        """One wrong word is one substitution, not two operations."""
        counts = levenshtein_counts(["the", "cat"], ["the", "bat"])
        assert counts.substitutions == 1
        assert counts.deletions == 0
        assert counts.insertions == 0
        assert counts.total_edits == 1

    def test_deletion_is_a_missing_reference_token(self):
        counts = levenshtein_counts(["a", "b", "c"], ["a", "c"])
        assert counts.deletions == 1
        assert counts.substitutions == 0
        assert counts.insertions == 0

    def test_insertion_is_an_extra_hypothesis_token(self):
        counts = levenshtein_counts(["a", "c"], ["a", "b", "c"])
        assert counts.insertions == 1
        assert counts.substitutions == 0
        assert counts.deletions == 0

    def test_two_substitutions_kept_aligned(self):
        """Hand-checked: a b c d e vs a x c d z.

        Aligns as keep a, sub b->x, keep c, keep d, sub e->z.
        """
        counts = levenshtein_counts(
            ["a", "b", "c", "d", "e"], ["a", "x", "c", "d", "z"]
        )
        assert (counts.substitutions, counts.deletions, counts.insertions) == (2, 0, 0)
        assert counts.total_edits == 2

    def test_mixed_operations_from_measured_baseline_clip(self):
        """Hand-checked against results/day08_baseline_transcripts.csv.

        clean_02 under additive_noise/medium/seed 42 gives a distance of 5
        over a 15-word reference. The tail contains a genuine tie:
        ref BETWEEN against hyp BETEEN can be read either as one
        substitution (BETWEEN->BETEEN) or as a deletion of BETWEEN plus an
        insertion of BETEEN. Both cost one edit, so both alignments are
        optimal and the total is 5 either way.

        The counts below follow the documented tie-break (diagonal, then up,
        then left), which prefers the substitution at the tail. The
        distribution between S/D/I is therefore alignment-dependent, while
        the total is not.
        """
        ref = (
            "THE SUMMER MONTHS PLIED UP AND DOWN THE LOCH AND "
            "INCIDENTALLY CARRIED ON COMMUNICATION BETWEEN"
        ).split()
        hyp = (
            "SUMMER MONTHS QUIED UP AND DOWN THE LA AND "
            "INCIDENTALLY CARRIED OUT COMMUNICATION BETEEN"
        ).split()
        counts = levenshtein_counts(ref, hyp)
        assert counts.ref_len == 15
        assert counts.hyp_len == 14
        assert counts.total_edits == 5
        # Tie-break prefers BETWEEN->BETEEN as a substitution.
        assert (counts.substitutions, counts.deletions, counts.insertions) == (4, 1, 0)
        # Deletion of the leading THE is unambiguous.
        assert counts.deletions == 1

    def test_counts_always_sum_to_total_edits(self):
        cases = [
            (["a"], ["b"]),
            (["a", "b"], []),
            ([], ["a", "b"]),
            (["a", "b", "c"], ["x", "c", "y", "z"]),
        ]
        for ref, hyp in cases:
            counts = levenshtein_counts(ref, hyp)
            assert (
                counts.substitutions + counts.deletions + counts.insertions
                == counts.total_edits
            )

    def test_total_edits_equals_optimal_distance(self):
        """The breakdown must not change the distance concept 1 reported."""
        ref = "the quick brown fox jumps".split()
        hyp = "the slow brown ox jumped far".split()
        counts = levenshtein_counts(ref, hyp)
        n, m = len(ref), len(hyp)
        dist = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dist[i][0] = i
        for j in range(m + 1):
            dist[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dist[i][j] = min(
                    dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                    dist[i - 1][j] + 1,
                    dist[i][j - 1] + 1,
                )
        assert counts.total_edits == dist[n][m]

    def test_empty_sequences(self):
        both_empty = levenshtein_counts([], [])
        assert both_empty.total_edits == 0
        assert levenshtein_counts([], ["a", "b"]).insertions == 2
        assert levenshtein_counts(["a", "b"], []).deletions == 2

    def test_inputs_are_not_mutated(self):
        ref, hyp = ["a", "b"], ["a", "c"]
        levenshtein_counts(ref, hyp)
        assert ref == ["a", "b"]
        assert hyp == ["a", "c"]

    def test_deterministic_for_tied_alignments(self):
        ref = ["a", "b", "c", "d"]
        hyp = ["b", "c", "d", "e"]
        assert levenshtein_counts(ref, hyp) == levenshtein_counts(ref, hyp)

    def test_to_dict_includes_total_edits(self):
        counts = levenshtein_counts(["a", "b", "c"], ["a", "x"])
        row = counts.to_dict()
        assert set(row) == {
            "substitutions",
            "deletions",
            "insertions",
            "ref_len",
            "hyp_len",
            "total_edits",
        }
        assert row["total_edits"] == counts.total_edits

    def test_edit_counts_constructs_directly(self):
        counts = EditCounts(
            substitutions=1, deletions=2, insertions=3, ref_len=10, hyp_len=11
        )
        assert counts.total_edits == 6


class TestRateCarriesCounts:
    """The rate record must expose the operations behind the ratio."""

    def test_word_rate_exposes_word_counts(self):
        rate = word_error_rate("a b c", "a x c")
        assert rate.substitutions == 1
        assert rate.deletions == 0
        assert rate.insertions == 0
        assert rate.total_edits == (
            rate.substitutions + rate.deletions + rate.insertions
        )

    def test_character_rate_exposes_character_counts(self):
        rate = character_error_rate("abcd", "abxd")
        assert rate.substitutions == 1
        assert rate.ref_len == 4

    def test_rate_denominator_unchanged_by_breakdown(self):
        """Concept 2 must not perturb the concept 1 rate."""
        rate = word_error_rate("a b c d", "a d")
        assert rate.deletions == 2
        assert rate.total_edits == 2
        assert rate.error_rate == pytest.approx(0.5)

    def test_error_rate_dataclass_constructs_directly(self):
        rate = ErrorRate(
            metric="wer",
            error_rate=0.5,
            total_edits=2,
            ref_len=4,
            hyp_len=4,
            substitutions=0,
            deletions=2,
            insertions=0,
        )
        assert rate.to_dict()["deletions"] == 2


class TestInputSafety:
    """Scoring must not alter the caller's data."""

    def test_inputs_are_not_mutated(self):
        ref, hyp = "a b c", "a x c"
        word_error_rate(ref, hyp)
        character_error_rate(ref, hyp)
        assert (ref, hyp) == ("a b c", "a x c")

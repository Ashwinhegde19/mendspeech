"""Unit tests for offline decoder-probe instrumentation (Phase 7.1).

All tests are synthetic, deterministic, offline, and fast: they exercise
:mod:`src.hinglish.decoder_diagnostics` on plain integer sequences and fakes
for ``model.generate`` and its output object. No transformers, torch, or
network involvement.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re

import pytest

from src.hinglish.decoder_diagnostics import (
    PROBE_VERSION,
    CONFIG_DEFINITIONS,
    GenerationCapture,
    GenerationProbeResult,
    analyze_generation,
    baseline_generate_kwargs,
    config_generate_kwargs,
    token_repetition_statistics,
)

# Stand-in token ids (offline; never fetched): 50258... prefix ids, 50257 EOS.
PREFIX_IDS = [50258, 50358, 50360]
EOS_IDS = [50257]

NOT_APPLICABLE_TAIL = {
    "period": None,
    "copies": 0,
    "coverage_tokens": 0,
    "ratio": 0.0,
}


# ---------------------------------------------------------------------------
# generate-kwargs contract
# ---------------------------------------------------------------------------


def test_probe_version_is_pinned() -> None:
    assert PROBE_VERSION == "mendspeech.hinglish_decoder_probe.v1"


def test_baseline_generate_kwargs_is_exact_and_fresh() -> None:
    first = baseline_generate_kwargs()
    second = baseline_generate_kwargs()
    assert first == {"task": "transcribe", "language": "hi"}
    assert first == second
    assert first is not second
    first["task"] = "translated"
    assert baseline_generate_kwargs() == {"task": "transcribe", "language": "hi"}


def test_config_generate_kwargs_c0_c1_c2() -> None:
    assert config_generate_kwargs("C0") == {
        "task": "transcribe",
        "language": "hi",
    }
    assert config_generate_kwargs("C1") == {
        "task": "transcribe",
        "language": "hi",
        "max_new_tokens": 160,
    }
    assert config_generate_kwargs("C2") == {
        "task": "transcribe",
        "language": "hi",
        "no_repeat_ngram_size": 4,
    }


def test_config_generate_kwargs_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError):
        config_generate_kwargs("C3")


def test_config_definitions_document_each_probe_config() -> None:
    assert set(CONFIG_DEFINITIONS) == {"C0", "C1", "C2"}
    assert "baseline" in CONFIG_DEFINITIONS["C0"]
    assert "160" in CONFIG_DEFINITIONS["C1"]
    assert "4" in CONFIG_DEFINITIONS["C2"]


# ---------------------------------------------------------------------------
# token_repetition_statistics exact cases
# ---------------------------------------------------------------------------


def test_repetition_statistics_no_repeat() -> None:
    stats = token_repetition_statistics([11, 35, 42, 7], decoder_prefix_tokens=0)
    assert stats["generated_token_count"] == 4
    assert stats["max_consecutive_token_repeats"] == 1
    assert stats["longest_repeated_token_ngram_length"] == 0
    assert stats["longest_repeated_token_ngram_copies"] == 0
    assert stats["periodic_tail"] == NOT_APPLICABLE_TAIL


def test_repetition_statistics_repeated_single_token_mid_sequence() -> None:
    stats = token_repetition_statistics([1, 5, 5, 5, 5, 9], decoder_prefix_tokens=0)
    assert stats["generated_token_count"] == 6
    assert stats["max_consecutive_token_repeats"] == 4
    # Overlapping occurrences count: (5, 5, 5) starts at indices 1 and 2.
    assert stats["longest_repeated_token_ngram_length"] == 3
    assert stats["longest_repeated_token_ngram_copies"] == 2
    assert stats["longest_repeated_token_ngram_copies"] < stats["max_consecutive_token_repeats"]
    assert stats["periodic_tail"] == NOT_APPLICABLE_TAIL, (
        "the identical run ends mid-sequence, so no end-anchored tail loop"
    )


def test_repetition_statistics_tail_of_identical_tokens_is_period_one() -> None:
    stats = token_repetition_statistics([1, 9, 5, 5, 5, 5], decoder_prefix_tokens=0)
    assert stats["periodic_tail"] == {
        "period": 1,
        "copies": 4,
        "coverage_tokens": 4,
        "ratio": 4 / 6,
    }


def test_repetition_statistics_repeated_trigram() -> None:
    stats = token_repetition_statistics(
        [1, 2, 3, 10, 11, 12, 2, 3, 10, 11, 12, 4], decoder_prefix_tokens=0
    )
    assert stats["generated_token_count"] == 12
    assert stats["max_consecutive_token_repeats"] == 1
    assert stats["longest_repeated_token_ngram_length"] == 5
    assert stats["longest_repeated_token_ngram_copies"] == 2
    assert stats["periodic_tail"] == NOT_APPLICABLE_TAIL


def test_repetition_statistics_periodic_tail_full_coverage() -> None:
    stats = token_repetition_statistics(
        PREFIX_IDS + [7, 8, 9, 7, 8, 9], decoder_prefix_tokens=3
    )
    assert stats["generated_token_count"] == 6
    assert stats["periodic_tail"] == {
        "period": 3,
        "copies": 2,
        "coverage_tokens": 6,
        "ratio": 1.0,
    }
    assert stats["longest_repeated_token_ngram_length"] == 3
    assert stats["longest_repeated_token_ngram_copies"] == 2


def test_repetition_statistics_periodic_tail_partial_coverage() -> None:
    stats = token_repetition_statistics(
        PREFIX_IDS + [1, 2] + [7, 8, 9, 7, 8, 9], decoder_prefix_tokens=3
    )
    assert stats["periodic_tail"] == {
        "period": 3,
        "copies": 2,
        "coverage_tokens": 6,
        "ratio": 6 / 8,
    }


def test_repetition_statistics_sixteen_gram_cap() -> None:
    gram = list(range(20, 36))  # 16 distinct token ids
    noisy = [1, 2] + gram + [3, 4] + gram + [5]
    stats = token_repetition_statistics(noisy, decoder_prefix_tokens=0)
    assert stats["longest_repeated_token_ngram_length"] == 16
    assert stats["longest_repeated_token_ngram_copies"] == 2


def test_repetition_statistics_tie_break_prefers_earliest_start() -> None:
    # Two 2-grams each occur twice: [1, 2] (starts 0 and 4) and [5, 6]
    # (starts 6 and 8). Same length and same copies: the earliest start wins.
    stats = token_repetition_statistics(
        [1, 2, 3, 4, 1, 2, 5, 6, 5, 6], decoder_prefix_tokens=0
    )
    assert stats["longest_repeated_token_ngram_length"] == 2
    assert stats["longest_repeated_token_ngram_copies"] == 2


def test_repetition_statistics_prefers_more_copies_at_same_length() -> None:
    # At the largest repeating n (2 here): [8, 9] occurs three times,
    # [1, 2] twice, so the higher copy count wins the tie-break.
    sequence = [1, 2, 3, 4, 1, 2, 5, 8, 9, 8, 9, 0, 8, 9]
    stats = token_repetition_statistics(sequence, decoder_prefix_tokens=0)
    assert stats["longest_repeated_token_ngram_length"] == 2
    assert stats["longest_repeated_token_ngram_copies"] == 3  # [8, 9]


def test_repetition_statistics_overlapping_occurrences_counted() -> None:
    # "aaaa": the 3-gram (a, a, a) occurs at starts 0 and 1 (overlapping),
    # so the largest qualifying n is 3 with 2 occurrences.
    stats = token_repetition_statistics([1, 1, 1, 1], decoder_prefix_tokens=0)
    assert stats["max_consecutive_token_repeats"] == 4
    assert stats["longest_repeated_token_ngram_length"] == 3
    assert stats["longest_repeated_token_ngram_copies"] == 2
    assert stats["periodic_tail"] == {
        "period": 1,
        "copies": 4,
        "coverage_tokens": 4,
        "ratio": 1.0,
    }


def test_repetition_statistics_excludes_decoder_prefix_tokens() -> None:
    stats = token_repetition_statistics(
        PREFIX_IDS + [10, 10, 10, 10], decoder_prefix_tokens=3
    )
    assert stats["generated_token_count"] == 4
    assert stats["max_consecutive_token_repeats"] == 4
    assert stats["periodic_tail"] == {
        "period": 1,
        "copies": 4,
        "coverage_tokens": 4,
        "ratio": 1.0,
    }


def test_repetition_statistics_empty_generated_portion() -> None:
    prefix_only = list(PREFIX_IDS)
    stats = token_repetition_statistics(prefix_only, decoder_prefix_tokens=3)
    assert stats == {
        "generated_token_count": 0,
        "max_consecutive_token_repeats": 0,
        "longest_repeated_token_ngram_length": 0,
        "longest_repeated_token_ngram_copies": 0,
        "periodic_tail": NOT_APPLICABLE_TAIL,
    }
    whole_prefix = token_repetition_statistics(prefix_only, decoder_prefix_tokens=0)
    assert whole_prefix["generated_token_count"] == 3
    assert whole_prefix["max_consecutive_token_repeats"] == 1


def test_repetition_statistics_works_on_tuples_empty_and_single() -> None:
    stats = token_repetition_statistics((9, 9, 9), decoder_prefix_tokens=0)
    assert stats["generated_token_count"] == 3
    assert stats["max_consecutive_token_repeats"] == 3
    assert token_repetition_statistics([], decoder_prefix_tokens=0) == {
        "generated_token_count": 0,
        "max_consecutive_token_repeats": 0,
        "longest_repeated_token_ngram_length": 0,
        "longest_repeated_token_ngram_copies": 0,
        "periodic_tail": NOT_APPLICABLE_TAIL,
    }
    single = token_repetition_statistics([7], decoder_prefix_tokens=0)
    assert single["generated_token_count"] == 1
    assert single["max_consecutive_token_repeats"] == 1
    assert single["longest_repeated_token_ngram_length"] == 0


def test_repetition_statistics_deterministic() -> None:
    sequence = PREFIX_IDS + [100, 101, 102] * 53
    assert token_repetition_statistics(sequence, decoder_prefix_tokens=3) == (
        token_repetition_statistics(sequence, decoder_prefix_tokens=3)
    )


# ---------------------------------------------------------------------------
# analyze_generation exact cases
# ---------------------------------------------------------------------------


def test_analyze_generation_budget_from_max_new_tokens() -> None:
    result = analyze_generation(
        [50258, 1, 2],
        decoder_prefix_tokens=1,
        configured_max_length=None,
        configured_max_new_tokens=2,
        eos_token_ids=EOS_IDS,
    )
    assert result.effective_token_budget == 2
    assert result.generated_token_count == 2
    assert result.total_sequence_tokens == 3
    assert result.hit_length_limit is True


def test_analyze_generation_budget_from_max_length_subtracts_prefix() -> None:
    result = analyze_generation(
        PREFIX_IDS + [7] * 4,
        decoder_prefix_tokens=3,
        configured_max_length=7,
        configured_max_new_tokens=None,
        eos_token_ids=EOS_IDS,
    )
    assert result.effective_token_budget == 4
    assert result.generated_token_count == 4
    assert result.hit_length_limit is True


def test_analyze_generation_no_budget_reports_none() -> None:
    result = analyze_generation(
        PREFIX_IDS + [1, 2, 3],
        decoder_prefix_tokens=3,
        configured_max_length=None,
        configured_max_new_tokens=None,
        eos_token_ids=EOS_IDS,
    )
    assert result.effective_token_budget is None
    assert result.hit_length_limit is False


def test_analyze_generation_hit_length_limit_without_eos() -> None:
    result = analyze_generation(
        PREFIX_IDS + [10] * 160,
        decoder_prefix_tokens=3,
        configured_max_length=None,
        configured_max_new_tokens=160,
        eos_token_ids=EOS_IDS,
    )
    assert result.generated_token_count == 160
    assert result.effective_token_budget == 160
    assert result.eos_emitted is False
    assert result.eos_token_id is None
    assert result.tokens_after_eos == 0
    assert result.hit_length_limit is True
    assert result.configured_max_new_tokens == 160
    assert result.configured_max_length is None
    assert result.total_sequence_tokens == 163


def test_analyze_generation_short_output_below_budget_is_not_limited() -> None:
    result = analyze_generation(
        [1, 2, 3],
        decoder_prefix_tokens=0,
        configured_max_length=None,
        configured_max_new_tokens=160,
        eos_token_ids=EOS_IDS,
    )
    assert result.generated_token_count == 3
    assert result.hit_length_limit is False


def test_analyze_generation_eos_prevents_length_limit_hit() -> None:
    # Budgeted stop keeps the tokens emitted up to and past the EOS: 159
    # filler tokens, then EOS as the 160th token, then one token after it.
    # The budget is reached but EOS is present, so the run is not classified
    # as a length-limit hit.
    result = analyze_generation(
        [10] * 159 + [50257] + [9],
        decoder_prefix_tokens=0,
        configured_max_length=None,
        configured_max_new_tokens=160,
        eos_token_ids=EOS_IDS,
    )
    assert result.generated_token_count == 161
    assert result.effective_token_budget == 160
    assert result.eos_emitted is True
    assert result.eos_token_id == 50257
    assert result.tokens_after_eos == 1
    assert result.hit_length_limit is False


def test_analyze_generation_tokens_after_eos_uses_first_occurrence() -> None:
    result = analyze_generation(
        [50257, 1, 2, 50257, 3],
        decoder_prefix_tokens=0,
        configured_max_length=None,
        configured_max_new_tokens=None,
        eos_token_ids=EOS_IDS,
    )
    assert result.eos_token_id == 50257
    assert result.tokens_after_eos == 4


def test_analyze_generation_eos_token_id_prefers_largest_matching_id() -> None:
    result = analyze_generation(
        [1, 50258, 2, 50257, 3],
        decoder_prefix_tokens=0,
        configured_max_length=None,
        configured_max_new_tokens=None,
        eos_token_ids=[50257, 50258],
    )
    assert result.eos_emitted is True
    # Two configured EOS ids are both emitted; the contract records the max
    # matching id (50258 > 50257).
    assert result.eos_token_id == 50258


def test_analyze_generation_empty_generated_portion() -> None:
    result = analyze_generation(
        list(PREFIX_IDS),
        decoder_prefix_tokens=3,
        configured_max_length=448,
        configured_max_new_tokens=None,
        eos_token_ids=EOS_IDS,
    )
    assert result == GenerationProbeResult(
        generated_token_count=0,
        decoder_prefix_tokens=3,
        total_sequence_tokens=3,
        effective_token_budget=445,
        hit_length_limit=False,
        configured_max_length=448,
        configured_max_new_tokens=None,
        eos_emitted=False,
        eos_token_id=None,
        tokens_after_eos=0,
        token_repetition={
            "generated_token_count": 0,
            "max_consecutive_token_repeats": 0,
            "longest_repeated_token_ngram_length": 0,
            "longest_repeated_token_ngram_copies": 0,
            "periodic_tail": NOT_APPLICABLE_TAIL,
        },
    )


def test_analyze_generation_without_eos_information_reports_no_eos() -> None:
    result = analyze_generation(
        [10] * 5,
        decoder_prefix_tokens=0,
        configured_max_length=None,
        configured_max_new_tokens=5,
        eos_token_ids=[],
    )
    assert result.eos_emitted is False
    assert result.eos_token_id is None
    assert result.hit_length_limit is True


def test_analyze_generation_deterministic_across_calls() -> None:
    sequence = PREFIX_IDS + [100, 101, 102] * 53
    kwargs = dict(
        decoder_prefix_tokens=3,
        configured_max_length=None,
        configured_max_new_tokens=160,
        eos_token_ids=EOS_IDS,
    )
    assert analyze_generation(sequence, **kwargs) == analyze_generation(
        sequence, **kwargs
    )


# ---------------------------------------------------------------------------
# GenerationCapture
# ---------------------------------------------------------------------------


class _FakeSequences:
    """Minimal stand-in for a generation output tensor of shape (batch, seq)."""

    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[int]]:
        return [list(row) for row in self._rows]


class _FakeGenerateOutput:
    """Minimal stand-in for ``GreedySearchEncoderDecoderOutput``."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.sequences = _FakeSequences(rows)


def test_capture_forces_return_dict_and_passes_kwargs_through() -> None:
    capture = GenerationCapture()
    seen_kwargs: dict = {}

    def generate(features, **kwargs):
        seen_kwargs.update(kwargs)
        return _FakeGenerateOutput([[11, 12, 50257]])

    wrapped = capture.wrap(generate)
    result = wrapped("features", task="transcribe", language="hi")
    # The original callable sees the forced flag plus all passed kwargs.
    assert seen_kwargs == {
        "task": "transcribe",
        "language": "hi",
        "return_dict_in_generate": True,
    }
    # The wrapper's return value is out.sequences (unchanged object).
    assert isinstance(result, _FakeSequences)
    assert result.tolist() == [[11, 12, 50257]]
    assert len(capture.captures) == 1


def test_capture_records_kwargs_sequences_and_error_none() -> None:
    capture = GenerationCapture()
    wrapped = capture.wrap(
        lambda features, **kwargs: _FakeGenerateOutput([[1, 2, 3], [4, 5, 6]])
    )
    returned = wrapped(
        "features",
        max_new_tokens=160,
        language="hi",
        task="transcribe",
        num_beams=5,
    )
    assert capture.captures == [
        {
            "kwargs": {"max_new_tokens": 160, "language": "hi", "task": "transcribe"},
            "sequences": [[1, 2, 3], [4, 5, 6]],
            "error": None,
        }
    ]
    # num_beams is passed through to the model but never recorded.
    assert isinstance(returned, _FakeSequences)


def test_capture_records_only_contract_kwarg_keys() -> None:
    capture = GenerationCapture()
    wrapped = capture.wrap(lambda features, **kwargs: _FakeGenerateOutput([[1]]))
    wrapped("features", do_sample=True, top_k=50, no_repeat_ngram_size=4)
    recorded = capture.captures[0]["kwargs"]
    assert recorded == {"no_repeat_ngram_size": 4}
    assert "do_sample" not in recorded
    assert "top_k" not in recorded
    assert "return_dict_in_generate" not in recorded


def test_capture_returns_sequences_unchanged() -> None:
    capture = GenerationCapture()
    holder: dict = {}

    def generate(features, **kwargs):
        holder["out"] = _FakeGenerateOutput([[7, 8, 9]])
        return holder["out"]

    wrapped = capture.wrap(generate)
    returned = wrapped("features")
    assert returned is holder["out"].sequences


def test_capture_respects_pre_set_truthy_return_dict() -> None:
    capture = GenerationCapture()
    seen_kwargs: dict = {}

    def generate(features, **kwargs):
        seen_kwargs.update(kwargs)
        return _FakeGenerateOutput([[1]])

    capture.wrap(generate)("features", return_dict_in_generate=True)
    assert seen_kwargs["return_dict_in_generate"] is True


def test_capture_exception_records_error_and_reraises() -> None:
    capture = GenerationCapture()

    def generate(features, **kwargs):
        raise RuntimeError("out of memory")

    wrapped = capture.wrap(generate)
    with pytest.raises(RuntimeError, match="out of memory"):
        wrapped("features", max_length=448)
    assert capture.captures == [
        {
            "kwargs": {"max_length": 448},
            "sequences": None,
            "error": "RuntimeError: out of memory",
        }
    ]


def test_capture_reset_clears_captures() -> None:
    capture = GenerationCapture()
    wrapped = capture.wrap(lambda features, **kwargs: _FakeGenerateOutput([[1]]))
    wrapped("features")
    assert len(capture.captures) == 1
    capture.reset()
    assert capture.captures == []
    wrapped("features")
    assert len(capture.captures) == 1


def test_record_sequences_appends_standard_record() -> None:
    capture = GenerationCapture()
    record = capture.record_sequences(
        _FakeSequences([[5, 6, 7]]), kwargs={"max_new_tokens": 160, "do_sample": False}
    )
    assert capture.captures == [
        {
            "kwargs": {"max_new_tokens": 160},
            "sequences": [[5, 6, 7]],
            "error": None,
        }
    ]
    assert record is capture.captures[0]
    # kwargs=None or omitted records an empty kwargs mapping.
    empty = capture.record_sequences(None)
    assert empty == {"kwargs": {}, "sequences": None, "error": None}
    capture.reset()
    assert capture.captures == []


def test_record_sequences_matches_wrap_record_shape() -> None:
    vector_capture = GenerationCapture()
    wrap_capture = GenerationCapture()
    kwargs = {"max_new_tokens": 160, "language": "hi", "task": "transcribe"}
    record = vector_capture.record_sequences(_FakeSequences([[1, 2]]), kwargs)
    wrapped = wrap_capture.wrap(
        lambda features, **kw: _FakeGenerateOutput([[1, 2]])
    )
    wrapped("features", **kwargs)
    assert record == wrap_capture.captures[-1]


def test_capture_rows_feed_probe_statistics() -> None:
    capture = GenerationCapture()
    wrapped = capture.wrap(
        lambda features, **kwargs: _FakeGenerateOutput(
            [PREFIX_IDS + [10, 10, 10], PREFIX_IDS + [1, 2, 3]]
        )
    )
    wrapped("features")
    sequences = capture.captures[0]["sequences"]
    looped = token_repetition_statistics(sequences[0], decoder_prefix_tokens=3)
    assert looped["generated_token_count"] == 3
    assert looped["max_consecutive_token_repeats"] == 3
    clean = analyze_generation(
        sequences[1],
        decoder_prefix_tokens=3,
        configured_max_length=None,
        configured_max_new_tokens=None,
        eos_token_ids=EOS_IDS,
    )
    assert clean.hit_length_limit is False
    assert clean.eos_emitted is False


# ---------------------------------------------------------------------------
# serialisability and module hygiene
# ---------------------------------------------------------------------------


def test_statistics_output_is_json_serialisable() -> None:
    stats = token_repetition_statistics(
        [1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3], decoder_prefix_tokens=0
    )
    encoded = json.loads(json.dumps(stats, sort_keys=True))
    assert encoded == stats


def test_probe_result_is_json_serialisable() -> None:
    result = analyze_generation(
        [1] * 160,
        decoder_prefix_tokens=0,
        configured_max_length=None,
        configured_max_new_tokens=160,
        eos_token_ids=EOS_IDS,
    )
    encoded = json.loads(json.dumps(dataclasses.asdict(result), sort_keys=True))
    assert encoded["hit_length_limit"] is True
    assert encoded["eos_emitted"] is False
    repetition = encoded["token_repetition"]
    assert repetition["generated_token_count"] == 160
    assert repetition["max_consecutive_token_repeats"] == 160


def test_module_source_avoids_heavy_dependencies() -> None:
    source = inspect.getsource(_import_module())
    forbidden = re.findall(
        r"^\s*(?:import|from)\s+(transformers|torch|numpy|datasets|librosa)\b",
        source,
        flags=re.MULTILINE,
    )
    assert forbidden == [], forbidden


def test_module_source_matches_probe_contract_names() -> None:
    source = inspect.getsource(_import_module())
    for symbol in (
        "mendspeech.hinglish_decoder_probe.v1",
        "baseline_generate_kwargs",
        "config_generate_kwargs",
        "token_repetition_statistics",
        "analyze_generation",
        "GenerationProbeResult",
        "GenerationCapture",
        "return_dict_in_generate",
    ):
        assert symbol in source


def _import_module():
    """Import the diagnostics module so ``inspect.getsource`` can read it."""
    import src.hinglish.decoder_diagnostics as module

    return module

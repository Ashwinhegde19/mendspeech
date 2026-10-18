"""Offline instrumentation and analysis for Hinglish decoder generations.

Phase 7.1 controlled decoder probe support. This module is pure python,
offline, deterministic, and network free: it works on plain sequences of
integers and deliberately does not import transformers or torch. The probe
runner imports the public helpers verbatim and applies them to
``model.generate`` outputs from the archived whisper-small baseline.

Probe configs (fixed contract):

- ``"C0"``: baseline generate kwargs, exactly the archived inference call
  (``{"task": "transcribe", "language": "hi"}``, library defaults apply).
- ``"C1"``: ``C0`` plus ``max_new_tokens=160`` (bounded decode length).
- ``"C2"``: ``C0`` plus ``no_repeat_ngram_size=4`` (repetition control).

Definitions used by :func:`token_repetition_statistics` (documented precisely
because downstream interpretation depends on them):

- ``max_consecutive_token_repeats``: length of the longest run of one
  identical token id in the generated portion; ``0`` for an empty generated
  portion and ``1`` when no token ever repeats adjacently.
- ``longest_repeated_token_ngram_length`` / ``..._copies``: the largest
  ``n <= 16`` such that some ``n``-gram occurs at least twice anywhere in the
  generated portion, occurrences at overlapping positions counted
  (e.g. ``aaaa`` yields ``n=3`` with 2 occurrences of ``aaa``). At the largest
  qualifying ``n``, the reported gram is the one with the most occurrences,
  ties broken by the earliest starting index. ``0``/``0`` when no ``n``-gram
  repeats at all.
- ``periodic_tail``: the smallest period ``p`` such that the final
  ``k*p`` generated tokens form ``k >= 2`` exact copies of one ``p``-gram,
  walking the maximal ``k`` backwards from the end. Applicable loops must run
  to the very end of the generated portion (expected for capped, EOS-less
  loops). Not applicable: ``{"period": None, "copies": 0, "coverage_tokens":
  0, "ratio": 0.0}`` (also returned for an empty generated portion).

Version: :data:`PROBE_VERSION`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

PROBE_VERSION = "mendspeech.hinglish_decoder_probe.v1"

#: Fixed baseline generation kwargs reproduced by C0: exactly the kwargs the
#: archived baseline inference path passed to the ``pipeline(...)`` call
#: (verbatim: ``generate_kwargs={"task": "transcribe", "language": "hi"}``).
BASELINE_GENERATE_KWARGS: Dict[str, str] = {"task": "transcribe", "language": "hi"}

#: C1 = C0 + max_new_tokens=160 (bounded decode budget).
C1_MAX_NEW_TOKENS = 160

#: C2 = C0 + no_repeat_ngram_size=4 (repetition suppression).
C2_NO_REPEAT_NGRAM_SIZE = 4

#: Probe config identifiers and one-line meanings (fixed contract).
CONFIG_DEFINITIONS: Dict[str, str] = {
    "C0": (
        "baseline generate kwargs {'task': 'transcribe', 'language': 'hi'}; "
        "library defaults apply (no length or repetition controls)"
    ),
    "C1": "C0 plus max_new_tokens=160 (bounded decode budget)",
    "C2": "C0 plus no_repeat_ngram_size=4 (repetition suppression)",
}

#: Largest ``n`` tested for repeated ``n``-gram reporting.
MAX_NGRAM_LENGTH = 16

#: Generate kwargs that :class:`GenerationCapture` records (length-limit
#: relevant subset; ``return_dict_in_generate`` is forced on the call but is
#: packaging only and is not recorded).
_RECORDED_KWARG_KEYS: Tuple[str, ...] = (
    "max_new_tokens",
    "max_length",
    "no_repeat_ngram_size",
    "language",
    "task",
)


def baseline_generate_kwargs() -> dict:
    """Return the baseline (C0) generate kwargs.

    Returns:
        dict: A fresh ``{"task": "transcribe", "language": "hi"}`` mapping so
        callers may mutate it without affecting later calls.
    """
    return dict(BASELINE_GENERATE_KWARGS)


def config_generate_kwargs(config_name: str) -> dict:
    """Return the full generate kwargs for a named probe config.

    Args:
        config_name: ``"C0"``, ``"C1"``, or ``"C2"``. ``C0`` = baseline
            generate kwargs; ``C1`` = ``C0`` plus ``max_new_tokens=160``;
            ``C2`` = ``C0`` plus ``no_repeat_ngram_size=4``.

    Returns:
        dict: A fresh kwargs mapping for the config (safe to mutate).

    Raises:
        ValueError: If ``config_name`` is not one of the probe configs.
    """
    if config_name == "C0":
        return baseline_generate_kwargs()
    if config_name == "C1":
        kwargs = baseline_generate_kwargs()
        kwargs["max_new_tokens"] = C1_MAX_NEW_TOKENS
        return kwargs
    if config_name == "C2":
        kwargs = baseline_generate_kwargs()
        kwargs["no_repeat_ngram_size"] = C2_NO_REPEAT_NGRAM_SIZE
        return kwargs
    raise ValueError(
        f"unknown decoder probe config {config_name!r}; "
        f"expected one of {sorted(CONFIG_DEFINITIONS)}"
    )


def _generated_portion(sequence, decoder_prefix_tokens: int) -> Tuple[int, ...]:
    """Return the generated tokens as a tuple of ints with a clamped start.

    The slice start is clamped to ``>= 0``; a generator prefix is always
    non-negative, so negative values become ``0`` instead of Python's
    end-anchored slice semantics. A single row of a batched ``sequences``
    tensor must be passed (flat, one-dimensional).
    """
    start = max(0, int(decoder_prefix_tokens))
    return tuple(int(token) for token in sequence[start:])


def _max_consecutive_runs(generated: Tuple[int, ...]) -> int:
    """Return the longest run length of one identical token id."""
    if not generated:
        return 0
    longest = 1
    current = 1
    for previous, token in zip(generated, generated[1:]):
        if token == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _ngram_occurrences(
    generated: Tuple[int, ...], n: int
) -> Dict[Tuple[int, ...], Tuple[int, int]]:
    """Count overlapping occurrences of every ``n``-gram.

    Returns:
        dict: Mapping each ``n``-gram (tuple of int) to ``(count,
        first_start_index)``, preserving first-observation start order.
    """
    occurrences: Dict[Tuple[int, ...], Tuple[int, int]] = {}
    for start in range(len(generated) - n + 1):
        gram = generated[start : start + n]
        count, first_start = occurrences.get(gram, (0, start))
        occurrences[gram] = (count + 1, first_start)
    return occurrences


def _not_applicable_periodic_tail() -> dict:
    """Return the deterministic not-applicable ``periodic_tail`` mapping."""
    return {
        "period": None,
        "copies": 0,
        "coverage_tokens": 0,
        "ratio": 0.0,
    }


def _periodic_tail(generated: Tuple[int, ...]) -> dict:
    """Find the smallest-period suffix loop, if any.

    For each period ``p = 1, 2, ...`` (ascending, ``p <= len // 2``) walk
    ``p``-grams backwards from the end while they equal the final ``p``-gram;
    a period qualifies once two or more copies reach the end. The first
    qualifying ``p`` is returned with its maximal copy count ``k``.
    """
    count = len(generated)
    for period in range(1, count // 2 + 1):
        tail = generated[count - period :]
        copies = 1
        while (copies + 1) * period <= count:
            previous_start = count - (copies + 1) * period
            if generated[previous_start : previous_start + period] != tail:
                break
            copies += 1
        if copies >= 2:
            coverage = copies * period
            return {
                "period": period,
                "copies": copies,
                "coverage_tokens": coverage,
                "ratio": coverage / count if count else 0.0,
            }
    return _not_applicable_periodic_tail()


def token_repetition_statistics(sequence, *, decoder_prefix_tokens: int) -> dict:
    """Summarize repetition inside the generated token portion.

    Args:
        sequence: Flat sequence of ints (the full generated sequence
            including any decoder prompt/prefix tokens).
        decoder_prefix_tokens: Number of leading tokens to exclude before
            analysing (the generated portion is
            ``sequence[decoder_prefix_tokens:]``; clamped to ``>= 0``).

    Returns:
        dict: JSON-serialisable statistics with keys
        ``generated_token_count``,
        ``max_consecutive_token_repeats`` (longest run of one identical token
        id; ``0`` when the generated portion is empty),
        ``longest_repeated_token_ngram_length`` and
        ``longest_repeated_token_ngram_copies`` (largest ``n <= 16`` such that
        some ``n``-gram occurs at least twice, overlapping occurrences
        counted; at that ``n`` the gram with the most occurrences wins, ties
        broken by earliest start) and ``periodic_tail`` (mapping with keys
        ``period``, ``copies``, ``coverage_tokens``, ``ratio`` describing the
        smallest-period suffix loop with at least two copies that reaches the
        end of the generated portion, or the not-applicable mapping
        ``{"period": None, "copies": 0, "coverage_tokens": 0, "ratio": 0.0}``).
    """
    generated = _generated_portion(sequence, decoder_prefix_tokens)

    longest_length = 0
    longest_copies = 0
    if generated:
        for n in range(min(MAX_NGRAM_LENGTH, len(generated) - 1), 0, -1):
            occurrences = _ngram_occurrences(generated, n)
            best_count, best_start = max(
                occurrences.values(), key=lambda item: (item[0], -item[1])
            )
            if best_count >= 2:
                longest_length = n
                longest_copies = best_count
                break

    return {
        "generated_token_count": len(generated),
        "max_consecutive_token_repeats": _max_consecutive_runs(generated),
        "longest_repeated_token_ngram_length": longest_length,
        "longest_repeated_token_ngram_copies": longest_copies,
        "periodic_tail": _periodic_tail(generated),
    }


@dataclass(frozen=True)
class GenerationProbeResult:
    """Full analysis of one generation's token sequence.

    Attribute order follows the probe contract; all values are
    JSON-serialisable (see ``dataclasses.asdict``).

    Attributes:
        generated_token_count: Number of tokens after the decoder prefix.
        decoder_prefix_tokens: Prefix length excluded from the analysis.
        total_sequence_tokens: Length of the full sequence (prefix plus
            generated).
        effective_token_budget: ``configured_max_new_tokens`` when set, else
            ``configured_max_length - decoder_prefix_tokens`` when
            ``configured_max_length`` is set, else ``None``.
        hit_length_limit: True when the budget is set, the generated count
            reaches it, and no EOS token was reported.
        configured_max_length: The ``max_length`` kwarg recorded for the run
            (``None`` when unset).
        configured_max_new_tokens: The ``max_new_tokens`` kwarg recorded for
            the run (``None`` when unset).
        eos_emitted: True when any id from ``eos_token_ids`` appears in the
            generated portion.
        eos_token_id: The largest matching EOS id present (prefers the last
            id when several are configured and emitted), else ``None``.
        tokens_after_eos: Generated tokens strictly after the first EOS
            occurrence (``0`` when no EOS was emitted).
        token_repetition: Mapping from :func:`token_repetition_statistics`.
    """

    generated_token_count: int
    decoder_prefix_tokens: int
    total_sequence_tokens: int
    effective_token_budget: Optional[int]
    hit_length_limit: bool
    configured_max_length: Optional[int]
    configured_max_new_tokens: Optional[int]
    eos_emitted: bool
    eos_token_id: Optional[int]
    tokens_after_eos: int
    token_repetition: dict


def _effective_token_budget(
    decoder_prefix_tokens: int,
    configured_max_length,
    configured_max_new_tokens,
) -> Optional[int]:
    """Resolve the per-generation token budget from configured limits.

    ``max_new_tokens`` wins over ``max_length``; without either, the budget
    is unknown (``None``).
    """
    if configured_max_new_tokens is not None:
        return int(configured_max_new_tokens)
    if configured_max_length is not None:
        return int(configured_max_length) - int(decoder_prefix_tokens)
    return None


def _first_eos_occurrence(
    generated: Tuple[int, ...], eos_token_ids
) -> Tuple[bool, Optional[int], int]:
    """Locate the first EOS in the generated portion.

    Args:
        generated: Generated tokens as a tuple of ints.
        eos_token_ids: Iterable of EOS token ids (a single int is accepted
            for robustness); ``None`` or empty means "no EOS configured".

    Returns:
        tuple: ``(eos_emitted, eos_token_id, tokens_after_eos)`` where
        ``eos_token_id`` is the largest matching id present (preferred over
        earlier ids when several are configured and emitted) and
        ``tokens_after_eos`` counts generated tokens strictly after the
        first EOS occurrence.
    """
    if eos_token_ids is None:
        eos_set = set()
    elif isinstance(eos_token_ids, int):
        eos_set = {int(eos_token_ids)}
    else:
        eos_set = {int(eos_id) for eos_id in eos_token_ids}
    if not eos_set or not generated:
        return False, None, 0

    first_index = None
    for index, token in enumerate(generated):
        if token in eos_set:
            first_index = index
            break
    if first_index is None:
        return False, None, 0

    matching = eos_set.intersection(generated)
    return True, max(matching), len(generated) - (first_index + 1)


def analyze_generation(
    sequence, *, decoder_prefix_tokens: int, configured_max_length, configured_max_new_tokens, eos_token_ids
) -> GenerationProbeResult:
    """Analyze one full generation sequence against configured limits.

    Args:
        sequence: Flat sequence of ints for the full generation (prefix
            included; a batched ``sequences`` tensor row must be passed
            flattened, as :class:`GenerationCapture` records them).
        decoder_prefix_tokens: Number of decoder prefix tokens excluded from
            the generated portion (clamped to ``>= 0``).
        configured_max_length: Recorded ``max_length`` kwarg for the run, or
            ``None`` when unset.
        configured_max_new_tokens: Recorded ``max_new_tokens`` kwarg for the
            run, or ``None`` when unset.
        eos_token_ids: Iterable of EOS token ids for the model, or ``None``
            / empty when unknown.

    Returns:
        GenerationProbeResult: Deterministic per-sequence analysis. Empty
        generated portions yield all-zero counting fields, no EOS, no length
        limit hit, and the not-applicable repetition mapping; no other
        exceptions are raised on degenerate input handled here (see
        :func:`token_repetition_statistics` and
        :func:`_first_eos_occurrence` for details).
    """
    generated = _generated_portion(sequence, decoder_prefix_tokens)
    budget = _effective_token_budget(
        decoder_prefix_tokens, configured_max_length, configured_max_new_tokens
    )
    eos_emitted, eos_token_id, tokens_after_eos = _first_eos_occurrence(
        generated, eos_token_ids
    )

    # An empty generated portion cannot "hit" a limit: all counting fields
    # are zero and no limit is reported.
    if generated:
        hit_length_limit = (
            budget is not None and len(generated) >= budget and not eos_emitted
        )
    else:
        hit_length_limit = False

    return GenerationProbeResult(
        generated_token_count=len(generated),
        decoder_prefix_tokens=int(decoder_prefix_tokens),
        total_sequence_tokens=len(sequence),
        effective_token_budget=budget,
        hit_length_limit=hit_length_limit,
        configured_max_length=(
            configured_max_length
            if configured_max_length is None
            else int(configured_max_length)
        ),
        configured_max_new_tokens=(
            configured_max_new_tokens
            if configured_max_new_tokens is None
            else int(configured_max_new_tokens)
        ),
        eos_emitted=eos_emitted,
        eos_token_id=eos_token_id,
        tokens_after_eos=tokens_after_eos,
        token_repetition=token_repetition_statistics(
            sequence, decoder_prefix_tokens=decoder_prefix_tokens
        ),
    )


def _to_nested_python_lists(sequences):
    """Convert tensor/array/list sequence outputs to nested python lists.

    Args:
        sequences: A ``torch.Tensor``, numpy array, or nested list/tuple of
            token ints (2-D ``(batch, seq)`` in practice; a 1-D flat input
            is treated as one sequence).

    Returns:
        list[list[int]] | None: ``None`` passes through unchanged;
        objects exposing ``.tolist()`` convert directly; nested lists/tuples
        are re-validated to ints; flat 1-D int iterables become a single
        row. Non-convertible values are returned as given.
    """
    if sequences is None:
        return None
    tolist = getattr(sequences, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(sequences, (list, tuple)):
        if sequences and all(isinstance(item, int) for item in sequences):
            return [[int(item) for item in sequences]]
        rows = []
        for row in sequences:
            row_tolist = getattr(row, "tolist", None)
            if callable(row_tolist):
                rows.append(row_tolist())
            elif isinstance(row, (list, tuple)):
                rows.append([int(item) for item in row])
            else:
                rows.append(row)
        return rows
    return sequences


class GenerationCapture:
    """Non-behaviour-changing instrumentation wrapper for ``model.generate``.

    :meth:`wrap` returns a drop-in replacement that only does two things:

    1. It copies the caller's keyword arguments and forces
       ``return_dict_in_generate=True`` when the caller did not already pass
       a truthy value, so ``.sequences`` is available for recording.
    2. It records one entry per call into :attr:`captures`.

    Per upstream generation semantics, ``return_dict_in_generate=True``
    changes only the return packaging (a ``GreedySearchEncoderDecoderOutput``
    instead of the raw ``sequences`` tensor); the sampled token ids and
    stopping behaviour are unchanged, so callers that re-capture the return
    value (this wrapper returns ``out.sequences`` unchanged) observe exactly
    the tokens a baseline call would have produced.

    Recorded entries mirror the call outcomes: on success
    ``{"kwargs": {...}, "sequences": [...], "error": None}``; on exception
    ``{"kwargs": {...}, "sequences": None, "error": "<Type>: <msg>"}``, then
    the exception is re-raised. ``kwargs`` holds only the length-limit
    relevant subset: ``max_new_tokens``, ``max_length``,
    ``no_repeat_ngram_size``, ``language`` and ``task``.
    """

    def __init__(self) -> None:
        self.captures: List[dict] = []

    def reset(self) -> None:
        """Drop all recorded captures."""
        self.captures = []

    def record_sequences(self, sequences, kwargs=None) -> dict:
        """Append a standard record for an already-completed generate call.

        Low-level companion to :meth:`wrap` for callers that need a custom
        wrapper body (for example, a probe that installs its own
        ``model.generate`` wrapper). Produces exactly the record shape
        :meth:`wrap` uses: ``kwargs`` filtered to the contract subset
        (``max_new_tokens``, ``max_length``, ``no_repeat_ngram_size``,
        ``language``, ``task``), ``sequences`` converted to nested python
        lists, and ``error`` ``None``.

        Args:
            sequences: The ``out.sequences`` object from one generate call
                (``torch.Tensor``, numpy array, or nested list/tuple of ints);
                ``None`` is recorded as ``None``.
            kwargs: The generate kwargs actually passed to the call, or
                ``None`` / omitted.

        Returns:
            dict: The record appended to :attr:`captures`.
        """
        recorded_kwargs = (
            {key: kwargs[key] for key in _RECORDED_KWARG_KEYS if key in kwargs}
            if kwargs
            else {}
        )
        record = {
            "kwargs": recorded_kwargs,
            "sequences": _to_nested_python_lists(sequences),
            "error": None,
        }
        self.captures.append(record)
        return record

    def wrap(self, generate_callable: Callable) -> Callable:
        """Return an instrumented replacement for ``generate_callable``.

        Args:
            generate_callable: Typically ``model.generate``; called with the
                original positional and keyword arguments.

        Returns:
            Callable: The wrapper. It has the same ``*args, **kwargs``
            signature as ``generate_callable``; the recorded ``sequences``
            are python lists (never tensors), the only forced kwarg is
            ``return_dict_in_generate`` when missing, and the wrapper's
            return value is ``out.sequences`` unchanged.
        """

        def generate_capture(*args, **kwargs):
            call_kwargs = dict(kwargs)
            if not call_kwargs.get("return_dict_in_generate"):
                call_kwargs["return_dict_in_generate"] = True
            recorded_kwargs = {
                key: call_kwargs[key]
                for key in _RECORDED_KWARG_KEYS
                if key in call_kwargs
            }
            try:
                output = generate_callable(*args, **call_kwargs)
            except Exception as error:
                self.captures.append(
                    {
                        "kwargs": dict(recorded_kwargs),
                        "sequences": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                raise
            sequences = getattr(output, "sequences", None)
            self.captures.append(
                {
                    "kwargs": dict(recorded_kwargs),
                    "sequences": _to_nested_python_lists(sequences),
                    "error": None,
                }
            )
            return sequences

        return generate_capture

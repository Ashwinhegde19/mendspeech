#!/usr/bin/env python3
"""Inspect Whisper decoder failure mechanisms in the Hinglish baseline offline.

The frozen 30-utterance MUCS-Hinglish validation baseline (whisper-small @
973afd24, T4, float16, ``language=hi``, ``task=transcribe``) contains clips whose
hypotheses are dominated by repeated text. This script decides *how* those
hypotheses ended, using only artifacts that already exist: it never runs
inference, never touches the network, and never loads a tokenizer.

Two independent signals are produced per clip:

1. **Output shape.** Word and character counts, words/characters per second of
   audio, and repetition structure: longest consecutive word repeat, longest
   repeated phrase, the share of the hypothesis that sits inside a periodic
   chain, the position and size of the largest loop region, whether that loop
   was still active on the final token, and character-level loops hidden inside
   a single whitespace word (byte-level BPE can emit one token over and over
   without ever inserting a space).
2. **Decode cost.** ``runtime_sec`` values with a cluster search for a
   near-constant runtime ceiling, plus the correlation between runtime and
   output size inside the ceiling. A runtime that stops growing with output
   size is the fingerprint of a fixed decode-step budget (an exhausted
   generation length) rather than of an early EOS.

The emitted JSON is keyed by ``clip_id`` and contains counts, rates, and
statistics only. Transcript text is deliberately excluded so the payload can be
stored inside the repository without leaking private references or hypotheses.

Example:
    python scripts/inspect_hinglish_generation_failures.py \
        --raw-output /private/run/raw_modal_output.json \
        --manifest /private/run/frozen_validation_manifest.csv \
        --diagnostics /private/run/hallucination_diagnostics.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hinglish.metrics import code_switch_bin
from src.hinglish.normalization import normalize_hinglish_text

# A repeated run is reported when it covers at least ``period * min_repeats``
# tokens. The thresholds below were chosen so that ordinary Hindi words and
# normal syllable coincidences stay out of the loop categories, while a
# generation that is still looping on its final token is always counted.
WORD_LOOP_MAX_PERIOD = 8
WORD_LOOP_MIN_REPEATS = 3
PHRASE_LOOP_MIN_COPIES = 2
CHAR_LOOP_MAX_PERIOD = 8
CHAR_LOOP_MIN_REPEATS = 8

# A hypothesis word counts as repetitive when it participates in a periodic
# chain of at most this period. Chains of two copies are included, because a
# two-copy phrase loop is already a decoder defect; the observed separation
# between looping and non-looping clips is wide (>= 67 percent versus <= 18
# percent), so a quarter of the hypothesis is a conservative cut.
CHAIN_MAX_PERIOD_WORDS = 40
CHAIN_MIN_COPIES = 2
LOOP_SHARE_THRESHOLD = 0.25

# A loop whose region ends within this many words of the hypothesis is treated
# as still active when generation stopped. One trailing word is allowed because
# a final copy can be cut mid-way by the generation-length cap.
LOOP_TAIL_TOLERANCE_WORDS = 1

# Runtime clustering tolerances for the decode-cost ceiling search. The frozen
# T4 run shows a 0.047 s spread over 15 clips, so 50 ms and 1.5 percent are
# generous enough to catch the plateau and tight enough to exclude the
# proportional-runtime regime.
CEILING_ABS_TOLERANCE_SEC = 0.05
CEILING_REL_TOLERANCE = 0.015
CEILING_MIN_MEMBERS = 3

# A runtime ceiling counts as "decoupled from output size" when its runtime
# coefficient of variation is at most this fraction of the output-size
# coefficient of variation, and the output size itself still spans at least
# ``DECOUPLED_OUTPUT_SIZE_RATIO``. Observed frozen-run values: runtime CV inside
# the ceiling is about 0.004 while the output-character CV is about 0.10.
DECOUPLING_CV_RATIO = 0.1
DECOUPLED_OUTPUT_SIZE_RATIO = 1.2

CONTRAST_METRICS: Tuple[str, ...] = (
    "runtime_sec",
    "audio_duration_sec",
    "output_words",
    "output_characters",
    "reference_words",
    "words_per_second",
    "runtime_over_audio_duration",
    "max_consecutive_word_repeat",
    "longest_repeated_phrase_words",
    "phrase_repeat_coverage_words",
    "periodic_word_share",
    "loop_region_coverage_words",
    "loop_region_trailing_words",
    "final_word_characters",
    "longest_word_characters",
    "intra_word_loop_characters",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_raw_output(path: Path) -> List[Dict[str, Any]]:
    """Parse the Modal JSON payload, skipping any leading log preamble.

    Args:
        path: Path to ``raw_modal_output.json``.

    Returns:
        List of per-clip records with ``clip_id``, ``transcript``,
        ``runtime_sec``, and ``audio_duration_sec``.

    Raises:
        ValueError: If no JSON array is present in the file.
    """
    text = path.read_text(encoding="utf-8")
    start = text.find("[")
    if start < 0:
        raise ValueError(f"no JSON array found in {path}")
    payload = json.loads(text[start:])
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON array in {path}")
    return [record for record in payload if isinstance(record, dict)]


def load_manifest_rows(path: Path) -> Dict[str, Dict[str, str]]:
    """Load the frozen validation manifest keyed by clip ID.

    Args:
        path: Path to ``frozen_validation_manifest.csv``.

    Returns:
        Mapping of ``clip_id`` to the raw CSV row.
    """
    with path.open(encoding="utf-8") as handle:
        return {row["clip_id"]: row for row in csv.DictReader(handle)}


def load_diagnostics(path: Path) -> Dict[str, Any]:
    """Load offline hallucination diagnostics.

    Args:
        path: Path to ``hallucination_diagnostics.json``.

    Returns:
        The parsed payload with ``summary`` and ``records``.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def flagged_clip_ids(diagnostics: Dict[str, Any]) -> List[str]:
    """Return the clip IDs the offline diagnostics flagged as hallucinated.

    Args:
        diagnostics: Parsed diagnostics payload.

    Returns:
        Sorted list of clip IDs with ``likely_repetitive_hallucination`` set.
    """
    summary = diagnostics.get("summary", {})
    if summary.get("likely_hallucination_ids"):
        return sorted(str(clip_id) for clip_id in summary["likely_hallucination_ids"])
    return sorted(
        str(record["clip_id"])
        for record in diagnostics.get("records", [])
        if record.get("likely_repetitive_hallucination")
    )


def normalize_word_tokens(text: str) -> List[str]:
    """Split text into normalized whitespace tokens.

    Args:
        text: Raw transcript text.

    Returns:
        Tokens produced by the versioned Hinglish normalizer.
    """
    return normalize_hinglish_text(text).split()


def longest_consecutive_repeat(tokens: Sequence[str]) -> int:
    """Return the longest run of identical consecutive tokens.

    Args:
        tokens: Token sequence.

    Returns:
        Run length, or ``0`` for an empty sequence.
    """
    if not tokens:
        return 0
    best = 1
    current = 1
    for previous, current_token in zip(tokens, tokens[1:]):
        if current_token == previous:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def longest_periodic_run(
    tokens: Sequence[str],
    *,
    max_period: int,
    min_repeats: int,
) -> Dict[str, Any]:
    """Find the longest run that repeats a unit of at most ``max_period`` tokens.

    A run of ``covered`` tokens starting at ``start_index`` counts when the
    sequence satisfies ``tokens[i] == tokens[i + period]`` for every position
    inside the run. This detects both single-token loops and short multi-token
    loops, including loops that are still active on the final token.

    Args:
        tokens: Token sequence (words or characters).
        max_period: Largest unit length to consider, in tokens.
        min_repeats: Smallest number of unit repetitions that counts as a run.

    Returns:
        Dict with ``start_index``, ``period_tokens``, ``covered_tokens``,
        ``repeats``, ``reaches_end``, ``trailing_tokens``, and ``found``.
        When nothing qualifies, ``covered_tokens`` is ``0`` and ``found`` is
        ``False``.
    """
    count = len(tokens)
    empty = {
        "start_index": 0,
        "period_tokens": 0,
        "covered_tokens": 0,
        "repeats": 0,
        "reaches_end": False,
        "trailing_tokens": count,
        "found": False,
    }
    if count < 2:
        return empty

    best: Optional[Tuple[int, int, int]] = None  # (covered, period, start)
    for period in range(1, min(max_period, count // min_repeats) + 1):
        matches = [0] * (count + 1)
        for index in range(count - 1 - period, -1, -1):
            if tokens[index] == tokens[index + period]:
                matches[index] = matches[index + 1] + 1
        for start, matched in enumerate(matches):
            if matched <= 0:
                continue
            covered = period + matched
            if covered < period * min_repeats:
                continue
            candidate = (covered, period, start)
            if best is None or covered > best[0] or (
                covered == best[0] and period < best[1]
            ):
                best = candidate

    if best is None:
        return empty
    covered, period, start = best
    return {
        "start_index": start,
        "period_tokens": period,
        "covered_tokens": covered,
        "repeats": covered // period,
        "reaches_end": start + covered == count,
        "trailing_tokens": count - (start + covered),
        "found": True,
    }


def periodic_word_coverage(
    tokens: Sequence[str],
    *,
    max_period: int = CHAIN_MAX_PERIOD_WORDS,
    min_copies: int = CHAIN_MIN_COPIES,
) -> Dict[str, Any]:
    """Measure the share of tokens that participate in a repeating chain.

    Every position that continues a periodic chain of at most ``max_period``
    tokens with at least ``min_copies`` repetitions is marked. This is a
    coverage measure rather than a longest-run measure, so a hypothesis that
    alternates between two loops is still described as repetitive overall.

    Args:
        tokens: Token sequence.
        max_period: Largest unit length considered, in tokens.
        min_copies: Smallest number of unit repetitions for a chain to count.

    Returns:
        Dict with ``covered_tokens``, ``share``, ``max_chain_tokens``,
        ``max_chain_period`` and ``found``.
    """
    count = len(tokens)
    covered = [False] * count
    best_chain = 0
    best_period = 0
    if count >= min_copies:
        for period in range(1, max_period + 1):
            if count < period * min_copies:
                break
            chain_start: Optional[int] = None
            for index in range(count - period):
                if tokens[index] == tokens[index + period]:
                    if chain_start is None:
                        chain_start = index
                    covered[index] = True
                    covered[index + period] = True
                else:
                    if chain_start is not None:
                        length = index - chain_start + period
                        if length >= period * min_copies and length > best_chain:
                            best_chain = length
                            best_period = period
                        chain_start = None
            if chain_start is not None:
                length = count - chain_start
                if length >= period * min_copies and length > best_chain:
                    best_chain = length
                    best_period = period
    total = sum(1 for flag in covered if flag)
    return {
        "covered_tokens": total,
        "share": total / count if count else 0.0,
        "max_chain_tokens": best_chain,
        "max_chain_period": best_period,
        "found": total > 0,
    }


def longest_repeated_phrase(tokens: Sequence[str]) -> Dict[str, Any]:
    """Measure the longest phrase that repeats back to back.

    The phrase must satisfy ``tokens[i : i + k] == tokens[i + k : i + 2k]``.
    The extent of the repetition is then extended copy by copy, so a phrase
    repeated five times reports five copies and the tokens it covers.

    Args:
        tokens: Token sequence.

    Returns:
        Dict with ``phrase_words``, ``copies``, ``coverage_words``,
        ``start_index``, ``reaches_end`` and ``trailing_tokens``.
    """
    count = len(tokens)
    result = {
        "phrase_words": 0,
        "copies": 0,
        "coverage_words": 0,
        "start_index": 0,
        "reaches_end": False,
        "trailing_tokens": count,
    }
    if count < 2:
        return result

    best: Optional[Tuple[int, int, int, int]] = None  # (coverage, words, copies, start)
    for phrase in range(1, count // PHRASE_LOOP_MIN_COPIES + 1):
        for start in range(count - 2 * phrase + 1):
            if tokens[start : start + phrase] != tokens[start + phrase : start + 2 * phrase]:
                continue
            copies = 2
            while (
                start + (copies + 1) * phrase <= count
                and tokens[start : start + phrase]
                == tokens[start + copies * phrase : start + (copies + 1) * phrase]
            ):
                copies += 1
            coverage = copies * phrase
            candidate = (coverage, phrase, copies, start)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return result
    coverage, phrase, copies, start = best
    return {
        "phrase_words": phrase,
        "copies": copies,
        "coverage_words": coverage,
        "start_index": start,
        "reaches_end": start + coverage == count,
        "trailing_tokens": count - (start + coverage),
    }


def intra_word_loops(
    words: Sequence[str],
    *,
    max_period: int = CHAR_LOOP_MAX_PERIOD,
    min_repeats: int = CHAR_LOOP_MIN_REPEATS,
) -> Dict[str, Any]:
    """Find the longest character-level loop hidden inside a single word.

    Byte-level BPE can emit the same sub-word token repeatedly without a space,
    which collapses into one very long whitespace word. A word-level repeat
    detector cannot see that loop, so characters are inspected directly.

    Args:
        words: Whitespace tokens of the hypothesis.
        max_period: Largest repeating character unit.
        min_repeats: Smallest number of character-unit repetitions.

    Returns:
        Dict with ``word_index``, ``word_characters``, ``period_chars``,
        ``covered_characters``, ``repeats`` and ``found``.
    """
    best: Optional[Tuple[int, int, Dict[str, Any]]] = None
    for index, word in enumerate(words):
        run = longest_periodic_run(
            word, max_period=max_period, min_repeats=min_repeats
        )
        if not run["found"]:
            continue
        candidate = (
            int(run["covered_tokens"]),
            index,
            {
                "word_index": index,
                "word_characters": len(word),
                "period_chars": int(run["period_tokens"]),
                "covered_characters": int(run["covered_tokens"]),
                "repeats": int(run["repeats"]),
                "found": True,
            },
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return {
            "word_index": -1,
            "word_characters": 0,
            "period_chars": 0,
            "covered_characters": 0,
            "repeats": 0,
            "found": False,
        }
    return best[2]


def terminal_statistics(transcript: str, words: Sequence[str]) -> Dict[str, Any]:
    """Describe how the hypothesis ends without exposing any of its text.

    A truncated generation characteristically stops without sentence-final
    punctuation, sometimes in the middle of a word. Both facts are measurable
    from Unicode categories alone, so they can be reported as booleans and
    lengths without copying transcript content.

    Args:
        transcript: Raw hypothesis text.
        words: Raw whitespace tokens of the hypothesis.

    Returns:
        Dict with ``ends_with_terminal_punctuation``,
        ``final_word_characters`` and ``final_character_is_combining_mark``.
        A combining mark or virama on the final character means the hypothesis
        stopped in the middle of a word.
    """
    stripped = transcript.strip()
    final_word = words[-1] if words else ""
    final_character = stripped[-1] if stripped else ""
    category = unicodedata.category(final_character) if final_character else ""
    return {
        "ends_with_terminal_punctuation": final_character in {".", "।", "!", "?"},
        "final_word_characters": len(final_word),
        "final_character_is_combining_mark": category in {"Mn", "Mc"},
    }


def output_statistics(transcript: str) -> Dict[str, Any]:
    """Compute repetition and size statistics for one hypothesis.

    Args:
        transcript: Raw hypothesis text.

    Returns:
        Dict with word and character counts, loop structure, and the share of
        the hypothesis that is repetitive.
    """
    words = normalize_word_tokens(transcript)
    raw_words = transcript.split()
    short_loop = longest_periodic_run(
        words, max_period=WORD_LOOP_MAX_PERIOD, min_repeats=WORD_LOOP_MIN_REPEATS
    )
    loop_region = longest_periodic_run(
        words, max_period=CHAIN_MAX_PERIOD_WORDS, min_repeats=WORD_LOOP_MIN_REPEATS
    )
    coverage = periodic_word_coverage(words)
    phrase = longest_repeated_phrase(words)
    loops = intra_word_loops(words)
    repeats = longest_consecutive_repeat(words)

    loop_trailing = int(loop_region["trailing_tokens"]) if loop_region["found"] else len(words)
    char_loop_at_final_word = bool(loops["found"]) and loops["word_index"] == len(words) - 1
    terminal = terminal_statistics(transcript, raw_words)
    loop_active_at_stop = bool(loop_region["found"]) and (
        loop_trailing <= LOOP_TAIL_TOLERANCE_WORDS
    )
    has_word_loop = float(coverage["share"]) >= LOOP_SHARE_THRESHOLD
    has_char_loop = bool(loops["found"])
    return {
        "output_words": len(words),
        "output_characters": len("".join(words)),
        "max_consecutive_word_repeat": repeats,
        "longest_repeated_phrase_words": int(phrase["phrase_words"]),
        "phrase_repeat_copies": int(phrase["copies"]),
        "phrase_repeat_coverage_words": int(phrase["coverage_words"]),
        "phrase_repeat_reaches_end": bool(phrase["reaches_end"]),
        "phrase_repeat_trailing_words": int(phrase["trailing_tokens"]),
        "short_loop_period_words": int(short_loop["period_tokens"]),
        "short_loop_coverage_words": int(short_loop["covered_tokens"]),
        "short_loop_repeats": int(short_loop["repeats"]),
        "periodic_word_coverage_words": int(coverage["covered_tokens"]),
        "periodic_word_share": round(float(coverage["share"]), 4),
        "loop_region_start_words": int(loop_region["start_index"]) if loop_region["found"] else 0,
        "loop_region_period_words": int(loop_region["period_tokens"]),
        "loop_region_copies": int(loop_region["repeats"]),
        "loop_region_coverage_words": int(loop_region["covered_tokens"]),
        "loop_region_trailing_words": loop_trailing,
        "loop_active_at_stop": loop_active_at_stop,
        "longest_word_characters": max((len(word) for word in words), default=0),
        "intra_word_loop_period_chars": int(loops["period_chars"]),
        "intra_word_loop_characters": int(loops["covered_characters"]),
        "intra_word_loop_repeats": int(loops["repeats"]),
        "intra_word_loop_at_final_word": char_loop_at_final_word,
        "degenerate_at_stop": loop_active_at_stop or char_loop_at_final_word,
        "ends_with_terminal_punctuation": bool(terminal["ends_with_terminal_punctuation"]),
        "final_word_characters": int(terminal["final_word_characters"]),
        "final_character_is_combining_mark": bool(
            terminal["final_character_is_combining_mark"]
        ),
        "has_word_loop": has_word_loop,
        "has_char_loop": has_char_loop,
        "has_repetition_evidence": has_word_loop or has_char_loop,
    }


def detect_runtime_ceiling(
    runtimes: Dict[str, float],
    *,
    abs_tolerance_sec: float = CEILING_ABS_TOLERANCE_SEC,
    rel_tolerance: float = CEILING_REL_TOLERANCE,
    min_members: int = CEILING_MIN_MEMBERS,
) -> Dict[str, Any]:
    """Cluster clip runtimes and report the largest near-identical group.

    Args:
        runtimes: Mapping of clip ID to runtime in seconds.
        abs_tolerance_sec: Absolute runtime tolerance for one cluster.
        rel_tolerance: Relative runtime tolerance, applied to the cluster base.
        min_members: Fewest clips that make a cluster reportable.

    Returns:
        Dict with the ceiling ``members``, ``size``, ``min_sec``, ``max_sec``,
        ``spread_sec``, ``spread_relative``, ``gap_below_sec``, and
        ``share_of_clips``. ``members`` is empty when no cluster qualifies.
    """
    empty = {
        "members": [],
        "size": 0,
        "min_sec": 0.0,
        "max_sec": 0.0,
        "spread_sec": 0.0,
        "spread_relative": 0.0,
        "gap_below_sec": 0.0,
        "share_of_clips": 0.0,
        "found": False,
    }
    if not runtimes:
        return empty
    ordered = sorted(runtimes.items(), key=lambda item: item[1])
    best: Optional[Tuple[int, float, List[str]]] = None
    for base_id, base in ordered:
        del base_id
        members = [
            clip_id
            for clip_id, value in ordered
            if 0.0 <= value - base <= max(abs_tolerance_sec, rel_tolerance * base)
        ]
        if len(members) < min_members:
            continue
        spread = max(runtimes[clip_id] for clip_id in members) - base
        if best is None or len(members) > best[0] or (
            len(members) == best[0] and spread < best[1]
        ):
            best = (len(members), spread, members)
    if best is None:
        return empty

    members = best[2]
    values = [runtimes[clip_id] for clip_id in members]
    lowest = min(values)
    outside = [value for clip_id, value in runtimes.items() if clip_id not in members]
    below = [value for value in outside if value < lowest]
    return {
        "members": members,
        "size": len(members),
        "min_sec": round(lowest, 4),
        "max_sec": round(max(values), 4),
        "spread_sec": round(max(values) - lowest, 4),
        "spread_relative": round((max(values) - lowest) / lowest, 5),
        "gap_below_sec": round(lowest - max(below), 4) if below else 0.0,
        "share_of_clips": round(len(members) / len(runtimes), 4),
        "found": True,
    }


def coefficient_of_variation(values: Sequence[float]) -> Optional[float]:
    """Return the coefficient of variation of a series.

    Args:
        values: Numeric series.

    Returns:
        ``stdev / mean``, or ``None`` when it is undefined (fewer than two
        values, or a zero mean).
    """
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return None
    return statistics.stdev(values) / mean


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Return the Pearson correlation of two equal-length series.

    Args:
        xs: First series.
        ys: Second series.

    Returns:
        Correlation coefficient, or ``None`` when it is undefined (fewer than
        two points, or zero variance in either series).
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    return covariance / (variance_x**0.5 * variance_y**0.5)


def decoupling_statistics(
    per_clip: Sequence[Dict[str, Any]],
    member_ids: Iterable[str],
) -> Dict[str, Any]:
    """Compare decode cost with output size inside and outside a runtime ceiling.

    Args:
        per_clip: Per-clip records containing ``clip_id``, ``runtime_sec``,
            ``output_words`` and ``output_characters``.
        member_ids: Clip IDs that belong to the runtime ceiling.

    Returns:
        Dict describing output-size spread, runtime/output correlation inside
        and outside the ceiling, and the ``decoupled`` indicator.
    """
    members = set(member_ids)
    inside = [record for record in per_clip if record["clip_id"] in members]
    outside = [record for record in per_clip if record["clip_id"] not in members]

    def summarise(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {
                "clips": 0,
                "runtime_sec_range": [0.0, 0.0],
                "runtime_sec_coefficient_of_variation": None,
                "runtime_sec_correlation_words": None,
                "runtime_sec_correlation_characters": None,
                "runtime_sec_correlation_audio_duration": None,
                "audio_duration_sec_range": [0.0, 0.0],
                "output_words_range": [0, 0],
                "output_characters_range": [0, 0],
                "output_characters_coefficient_of_variation": None,
                "output_characters_ratio": 0.0,
                "output_words_ratio": 0.0,
            }
        runtimes = [float(record["runtime_sec"]) for record in records]
        words = [float(record["output_words"]) for record in records]
        characters = [float(record["output_characters"]) for record in records]
        durations = [float(record["audio_duration_sec"]) for record in records]
        words_ratio = (
            max(words) / min(words) if min(words) > 0 else 0.0
        )
        characters_ratio = (
            max(characters) / min(characters) if min(characters) > 0 else 0.0
        )
        return {
            "clips": len(records),
            "runtime_sec_range": [round(min(runtimes), 4), round(max(runtimes), 4)],
            "runtime_sec_coefficient_of_variation": coefficient_of_variation(runtimes),
            "runtime_sec_correlation_words": pearson_correlation(runtimes, words),
            "runtime_sec_correlation_characters": pearson_correlation(
                runtimes, characters
            ),
            "runtime_sec_correlation_audio_duration": pearson_correlation(
                runtimes, durations
            ),
            "audio_duration_sec_range": [
                round(min(durations), 4),
                round(max(durations), 4),
            ],
            "output_words_range": [int(min(words)), int(max(words))],
            "output_characters_range": [int(min(characters)), int(max(characters))],
            "output_characters_coefficient_of_variation": coefficient_of_variation(
                characters
            ),
            "output_characters_ratio": round(characters_ratio, 4),
            "output_words_ratio": round(words_ratio, 4),
        }

    inside_summary = summarise(inside)
    outside_summary = summarise(outside)
    runtime_cv = inside_summary["runtime_sec_coefficient_of_variation"]
    output_cv = inside_summary["output_characters_coefficient_of_variation"]
    decoupled = bool(
        inside_summary["clips"] >= CEILING_MIN_MEMBERS
        and inside_summary["output_characters_ratio"] >= DECOUPLED_OUTPUT_SIZE_RATIO
        and runtime_cv is not None
        and output_cv is not None
        and runtime_cv <= DECOUPLING_CV_RATIO * output_cv
    )
    return {
        "ceiling_clips": inside_summary,
        "non_ceiling_clips": outside_summary,
        "decoupled_from_output_size": decoupled,
    }


def _summarise_metric(values: Sequence[float]) -> Dict[str, Any]:
    """Summarise one metric for a clip group.

    Args:
        values: Metric values.

    Returns:
        Dict with ``clips``, ``mean``, ``median``, ``min`` and ``max``.
    """
    if not values:
        return {"clips": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "clips": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def contrast_statistics(
    per_clip: Sequence[Dict[str, Any]],
    flagged_ids: Iterable[str],
    *,
    metrics: Sequence[str] = CONTRAST_METRICS,
) -> Dict[str, Any]:
    """Build a flagged-versus-clean contrast table.

    Args:
        per_clip: Per-clip records with ``clip_id`` and the metrics.
        flagged_ids: Clip IDs that offline diagnostics flagged.
        metrics: Metric names to contrast.

    Returns:
        Dict with ``per_metric`` summaries and ``per_bin`` clip counts. Missing
        metric values are skipped rather than imputed.
    """
    flagged = set(flagged_ids)
    groups = {
        "flagged": [r for r in per_clip if r["clip_id"] in flagged],
        "clean": [r for r in per_clip if r["clip_id"] not in flagged],
    }
    per_metric: Dict[str, Any] = {}
    for metric in metrics:
        group_values = {
            name: [
                float(record[metric])
                for record in records
                if metric in record and record[metric] is not None
            ]
            for name, records in groups.items()
        }
        flagged_summary = _summarise_metric(group_values["flagged"])
        clean_summary = _summarise_metric(group_values["clean"])
        clean_median = clean_summary["median"]
        per_metric[metric] = {
            "flagged": flagged_summary,
            "clean": clean_summary,
            "median_ratio_flagged_over_clean": (
                round(flagged_summary["median"] / clean_median, 4)
                if clean_median
                else None
            ),
        }

    bins: Dict[str, Dict[str, int]] = {}
    for name, records in groups.items():
        for record in records:
            label = str(record.get("english_ratio_bin", "unknown"))
            bins.setdefault(label, {"flagged": 0, "clean": 0})
            bins[label][name] += 1
    return {
        "clip_counts": {name: len(records) for name, records in groups.items()},
        "per_metric": per_metric,
        "per_bin": bins,
    }


def build_inspection_report(
    raw_records: Sequence[Dict[str, Any]],
    manifest_rows: Dict[str, Dict[str, str]],
    diagnostics: Optional[Dict[str, Any]] = None,
    *,
    sources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the clip-ID-keyed inspection payload.

    Args:
        raw_records: Per-clip records parsed from the Modal output.
        manifest_rows: Frozen manifest rows keyed by clip ID.
        diagnostics: Optional offline diagnostics payload.
        sources: Optional provenance metadata (hashes, paths withheld).

    Returns:
        JSON-serialisable report with per-clip statistics, the runtime ceiling,
        decoupling evidence, and the flagged-versus-clean contrast.
    """
    flagged_ids = flagged_clip_ids(diagnostics) if diagnostics else []
    diagnostic_records = {
        str(record["clip_id"]): record
        for record in (diagnostics or {}).get("records", [])
    }

    per_clip: List[Dict[str, Any]] = []
    for record in raw_records:
        clip_id = str(record["clip_id"])
        audio_duration = float(record.get("audio_duration_sec") or 0.0)
        runtime = float(record.get("runtime_sec") or 0.0)
        statistics_payload = output_statistics(str(record.get("transcript", "")))
        manifest_row = manifest_rows.get(clip_id, {})
        reference_words = len(normalize_word_tokens(manifest_row.get("reference_transcript", "")))
        ratio_english = manifest_row.get("ratio_english_words")
        ratio_value = float(ratio_english) if ratio_english not in (None, "") else None
        diagnostic_row = diagnostic_records.get(clip_id, {})
        entry: Dict[str, Any] = {
            "clip_id": clip_id,
            "flagged": clip_id in set(flagged_ids),
            "runtime_sec": round(runtime, 4),
            "audio_duration_sec": round(audio_duration, 4),
            "reference_words": reference_words,
            "english_ratio": ratio_value,
            "english_ratio_bin": (
                code_switch_bin(ratio_value) if ratio_value is not None else "unknown"
            ),
            "words_per_second": (
                round(statistics_payload["output_words"] / audio_duration, 4)
                if audio_duration > 0
                else 0.0
            ),
            "runtime_over_audio_duration": (
                round(runtime / audio_duration, 4) if audio_duration > 0 else 0.0
            ),
        }
        for field in ("insertions", "substitutions", "deletions", "wer"):
            if field in diagnostic_row:
                entry[f"diagnostic_{field}"] = diagnostic_row[field]
        entry.update(statistics_payload)
        entry["output_words_per_second_ratio"] = (
            round(
                (entry["output_words"] / audio_duration) / (reference_words / audio_duration),
                4,
            )
            if audio_duration > 0 and reference_words > 0
            else None
        )
        per_clip.append(entry)

    per_clip.sort(key=lambda item: item["runtime_sec"])
    runtimes = {item["clip_id"]: float(item["runtime_sec"]) for item in per_clip}
    ceiling = detect_runtime_ceiling(runtimes)
    decoupling = decoupling_statistics(per_clip, ceiling["members"])
    ceiling_members = set(ceiling["members"])
    ceiling_repetition = {
        "members": sorted(m for m in ceiling_members),
        "members_with_repetition_evidence": _count_truthy(
            per_clip, ceiling_members, "has_repetition_evidence"
        ),
        "members_with_word_loop": _count_truthy(per_clip, ceiling_members, "has_word_loop"),
        "members_with_char_loop": _count_truthy(per_clip, ceiling_members, "has_char_loop"),
        "members_degenerate_at_stop": _count_truthy(
            per_clip, ceiling_members, "degenerate_at_stop"
        ),
        "members_with_loop_active_at_stop": _count_truthy(
            per_clip, ceiling_members, "loop_active_at_stop"
        ),
        "members_ending_with_sentence_punctuation": _count_truthy(
            per_clip, ceiling_members, "ends_with_terminal_punctuation"
        ),
        "members_ending_on_combining_mark": _count_truthy(
            per_clip, ceiling_members, "final_character_is_combining_mark"
        ),
        "non_members_ending_with_sentence_punctuation": _count_truthy(
            per_clip,
            [item["clip_id"] for item in per_clip if item["clip_id"] not in ceiling_members],
            "ends_with_terminal_punctuation",
        ),
        "loop_region_trailing_words_histogram": _histogram(
            per_clip, ceiling_members, "loop_region_trailing_words"
        ),
        "members_flagged": sum(1 for item in per_clip if item["clip_id"] in ceiling_members and item["flagged"]),
    }
    ceiling["repetition"] = ceiling_repetition

    evidence = {
        "clips_with_repetition_evidence": sum(
            1 for item in per_clip if item["has_repetition_evidence"]
        ),
        "clips_with_repetition_evidence_inside_ceiling": ceiling_repetition[
            "members_with_repetition_evidence"
        ],
        "clips_with_repetition_evidence_outside_ceiling": sum(
            1
            for item in per_clip
            if item["clip_id"] not in ceiling_members and item["has_repetition_evidence"]
        ),
        "max_output_words": max((item["output_words"] for item in per_clip), default=0),
        "max_output_characters": max(
            (item["output_characters"] for item in per_clip), default=0
        ),
        "max_output_words_per_second": round(
            max((item["words_per_second"] for item in per_clip), default=0.0), 4
        ),
    }

    report = {
        "clip_count": len(per_clip),
        "flagged_clip_count": len(flagged_ids),
        "flagged_clip_ids": flagged_ids,
        "sources": sources or {},
        "runtime_ceiling": ceiling,
        "decoupling": decoupling,
        "evidence_counts": evidence,
        "contrast": (
            contrast_statistics(per_clip, flagged_ids) if flagged_ids else None
        ),
        "per_clip": per_clip,
    }
    return report


def _count_truthy(
    per_clip: Sequence[Dict[str, Any]],
    member_ids: Iterable[str],
    field: str,
) -> int:
    """Count members whose boolean field is true.

    Args:
        per_clip: Per-clip records.
        member_ids: Clip IDs to restrict the count to.
        field: Boolean field name.

    Returns:
        Number of matching clips.
    """
    members = set(member_ids)
    return sum(
        1 for item in per_clip if item["clip_id"] in members and item.get(field)
    )


def _histogram(
    per_clip: Sequence[Dict[str, Any]],
    member_ids: Iterable[str],
    field: str,
) -> Dict[str, int]:
    """Count integer field values for a clip group.

    Args:
        per_clip: Per-clip records.
        member_ids: Clip IDs to restrict the count to.
        field: Integer field name.

    Returns:
        Mapping of stringified value to occurrence count, sorted by value.
    """
    members = set(member_ids)
    counts: Dict[int, int] = {}
    for item in per_clip:
        if item["clip_id"] not in members:
            continue
        value = int(item.get(field, 0))
        counts[value] = counts.get(value, 0) + 1
    return {str(key): counts[key] for key in sorted(counts)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Network-free inspection of Whisper decoder failure mechanisms in "
            "the frozen Hinglish baseline. Emits clip-ID-keyed statistics only."
        )
    )
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON report here. Defaults to stdout.",
    )
    args = parser.parse_args()

    raw_records = parse_raw_output(args.raw_output)
    manifest_rows = load_manifest_rows(args.manifest)
    diagnostics = load_diagnostics(args.diagnostics) if args.diagnostics else None
    sources = {
        "raw_output_sha256": sha256_file(args.raw_output),
        "manifest_sha256": sha256_file(args.manifest),
        "diagnostics_sha256": (
            sha256_file(args.diagnostics) if args.diagnostics else None
        ),
        "note": (
            "Private run archive; transcript text is intentionally absent from "
            "this report."
        ),
    }
    report = build_inspection_report(
        raw_records, manifest_rows, diagnostics, sources=sources
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(f"Inspection report written to {output_path}")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

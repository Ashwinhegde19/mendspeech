"""Hinglish dataset metadata helpers.

These helpers intentionally work with dataset metadata only. They do not
download audio and do not create permanent Hinglish training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


MUCS_HINGLISH_DATASET = "dianavdavidson/MUCS-Hinglish"

REQUIRED_MUCS_COLUMNS = frozenset(
    {
        "segment_id",
        "speaker_id",
        "transcript",
        "audio",
        "ratio_english_words",
        "ratio_hindi_words",
    }
)


@dataclass(frozen=True)
class MucsHinglishRecord:
    """One metadata record from the MUCS Hinglish dataset."""

    clip_id: str
    speaker_id: str
    split: str
    transcript: str
    ratio_english_words: float
    ratio_hindi_words: float
    english_words: str
    count_english_words: int
    count_hindi_words: int
    segment_part: str


@dataclass(frozen=True)
class HinglishSplitAudit:
    """Result of auditing speaker and clip separation across splits."""

    speaker_overlaps: tuple[str, ...]
    duplicate_segment_ids: tuple[str, ...]
    missing_speakers: tuple[str, ...]
    checked_splits: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether all audit checks passed."""
        return (
            not self.speaker_overlaps
            and not self.duplicate_segment_ids
            and not self.missing_speakers
        )


def _load_dataset(
    dataset_name: str,
    *,
    split: str,
    streaming: bool = True,
):
    """Load a Hugging Face dataset, importing the dependency lazily."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for Hinglish metadata loading. "
            "Install with: pip install -e '.[hinglish]'"
        ) from exc
    return load_dataset(dataset_name, split=split, streaming=streaming)


def validate_mucs_row(row: Mapping[str, Any]) -> None:
    """Validate one MUCS Hinglish row."""
    missing = REQUIRED_MUCS_COLUMNS.difference(row)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    if not str(row.get("segment_id", "")).strip():
        raise ValueError("segment_id must not be empty")
    if not str(row.get("speaker_id", "")).strip():
        raise ValueError("speaker_id must not be empty")
    if not isinstance(row.get("transcript"), str) or not row["transcript"].strip():
        raise ValueError("transcript must be a non-empty string")
    english = float(row["ratio_english_words"])
    hindi = float(row["ratio_hindi_words"])
    if not 0.0 <= english <= 1.0 or not 0.0 <= hindi <= 1.0:
        raise ValueError("language ratios must be between 0 and 1")


def mucs_row_to_record(
    row: Mapping[str, Any],
    *,
    split: str,
) -> MucsHinglishRecord:
    """Convert a MUCS row into a normalized manifest record."""
    validate_mucs_row(row)
    return MucsHinglishRecord(
        clip_id=str(row["segment_id"]),
        speaker_id=str(row["speaker_id"]),
        split=split,
        transcript=str(row["transcript"]).strip(),
        ratio_english_words=float(row["ratio_english_words"]),
        ratio_hindi_words=float(row["ratio_hindi_words"]),
        english_words=str(row.get("english_words", "")).strip(),
        count_english_words=int(row.get("count_english_words", 0)),
        count_hindi_words=int(row.get("count_dev_words", 0)),
        segment_part=str(row.get("segment_part", "")).strip(),
    )


def load_mucs_hinglish_manifest(
    split: str,
    *,
    max_items: int | None = None,
    dataset_name: str = MUCS_HINGLISH_DATASET,
    loader: Callable[[str, str, bool], Any] | None = None,
    streaming: bool = True,
) -> list[MucsHinglishRecord]:
    """Load Hinglish metadata without downloading the full audio collection.

    ``streaming=True`` keeps the first load lightweight. ``max_items`` bounds
    local or smoke-test work.
    """
    if max_items is not None and max_items < 0:
        raise ValueError("max_items must be non-negative")

    resolved_loader = loader
    if resolved_loader is None:
        resolved_loader = lambda name, split_name, use_streaming: _load_dataset(
            name, split=split_name, streaming=use_streaming
        )

    dataset = resolved_loader(dataset_name, split, streaming)
    records: list[MucsHinglishRecord] = []
    for index, row in enumerate(dataset):
        if max_items is not None and index >= max_items:
            break
        records.append(mucs_row_to_record(row, split=split))

    return records


def export_mucs_manifest(records: Iterable[MucsHinglishRecord]) -> list[dict[str, Any]]:
    """Convert records into a plain CSV-compatible dictionary list."""
    return [
        {
            "clip_id": record.clip_id,
            "speaker_id": record.speaker_id,
            "split": record.split,
            "transcript": record.transcript,
            "ratio_english_words": record.ratio_english_words,
            "ratio_hindi_words": record.ratio_hindi_words,
            "english_words": record.english_words,
            "count_english_words": record.count_english_words,
            "count_hindi_words": record.count_hindi_words,
            "segment_part": record.segment_part,
        }
        for record in records
    ]


def validate_mucs_manifest(records: Iterable[MucsHinglishRecord]) -> list[MucsHinglishRecord]:
    """Return validated records or raise on invalid manifest rows."""
    materialized = list(records)
    for record in materialized:
        if not record.clip_id:
            raise ValueError("clip_id must not be empty")
        if not record.speaker_id:
            raise ValueError("speaker_id must not be empty")
    return materialized


def audit_split_records(
    split_records: dict[str, Iterable[MucsHinglishRecord]],
) -> HinglishSplitAudit:
    """Audit Hinglish splits for speaker and clip leakage."""
    checked_splits = tuple(sorted(split_records))
    speakers_by_split: dict[str, set[str]] = {}
    segments: dict[str, set[str]] = {}
    for split, records in split_records.items():
        materialized = list(records)
        speaker_set: set[str] = set()
        segment_set: set[str] = set()
        for record in materialized:
            speaker_set.add(record.speaker_id)
            segment_set.add(record.clip_id)
        speakers_by_split[split] = speaker_set
        segments[split] = segment_set

    overlaps: list[str] = []
    for first, second in _pairs(checked_splits):
        overlaps.extend(sorted(speakers_by_split[first] & speakers_by_split[second]))

    duplicates: set[str] = set()
    seen: set[str] = set()
    for segment_set in segments.values():
        duplicates.update(segment_set & seen)
        seen.update(segment_set)

    missing = tuple(sorted(speakers_by_split.get("validation", set()) - set()))
    audit = HinglishSplitAudit(
        speaker_overlaps=tuple(overlaps),
        duplicate_segment_ids=tuple(sorted(duplicates)),
        missing_speakers=missing,
        checked_splits=checked_splits,
    )
    return audit


def split_speakers_for_validation(
    speakers: Iterable[str],
    *,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Deterministically split training speakers into train and validation.

    The official test partition is intentionally untouched by this function.
    """
    unique_speakers = sorted(set(speakers))
    if not unique_speakers:
        raise ValueError("speakers must not be empty")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    import random

    shuffled = list(unique_speakers)
    random.Random(seed).shuffle(shuffled)
    count = max(1, int(len(shuffled) * validation_ratio))
    validation = sorted(shuffled[:count])
    training = sorted(shuffled[count:])
    if not training:
        raise ValueError("validation_ratio leaves no training speakers")
    return training, validation


def _pairs(values: tuple[str, ...]) -> Iterable[tuple[str, str]]:
    for first_index, first in enumerate(values):
        for second in values[first_index + 1 :]:
            yield first, second

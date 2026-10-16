"""Hinglish dataset metadata helpers.

These helpers intentionally work with dataset metadata only. They do not
download audio and do not create permanent Hinglish training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


MUCS_HINGLISH_DATASET = "dianavdavidson/MUCS-Hinglish"
# Tested against the Hugging Face dataset revision recorded by
# scripts/prepare_mucs_hinglish_manifest.py on October 16, 2026.
MUCS_HINGLISH_REVISION = "3c66ee127a7b0e0a746d4df1002858567135edc1"

# The audio column is intentionally discarded after schema validation so its
# values are never decoded. These required columns are the metadata contract.
REQUIRED_MUCS_COLUMNS = frozenset(
    {
        "segment_id",
        "speaker_id",
        "transcript",
        "ratio_english_words",
        "ratio_hindi_words",
    }
)

MUCS_LICENSE_NOTE = (
    "Hugging Face declares CC BY 4.0 for this dataset copy, while OpenSLR "
    "originals are CC BY-SA 4.0. Treat the original OpenSLR CC BY-SA 4.0 "
    "requirement as controlling until the upstream license is clarified."
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
    metadata_valid: bool = True
    invalid_fields: tuple[str, ...] = ()
    recomputed_ratio_english_words: float | None = None


@dataclass(frozen=True)
class HinglishSplitAudit:
    """Result of auditing speaker and clip separation across splits."""

    speaker_overlaps: tuple[str, ...]
    duplicate_segment_ids: tuple[str, ...]
    duplicate_within_split: tuple[str, ...]
    cross_split_duplicates: tuple[str, ...]
    missing_speakers: tuple[str, ...]
    empty_speaker_ids: tuple[str, ...]
    checked_splits: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether all audit checks passed."""
        return not (
            self.speaker_overlaps
            or self.duplicate_segment_ids
            or self.duplicate_within_split
            or self.cross_split_duplicates
            or self.missing_speakers
            or self.empty_speaker_ids
        )


def _load_dataset(
    dataset_name: str,
    *,
    split: str,
    streaming: bool,
    revision: str | None,
):
    """Load a Hugging Face dataset, importing the dependency lazily."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for Hinglish metadata loading. "
            "Install with: pip install -e '.[hinglish]'"
        ) from exc
    return load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        revision=revision,
    )


def validate_mucs_row(row: Mapping[str, Any]) -> None:
    """Validate the structural parts of one MUCS Hinglish row."""
    missing = REQUIRED_MUCS_COLUMNS.difference(row)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    if not str(row.get("segment_id", "")).strip():
        raise ValueError("segment_id must not be empty")
    if not str(row.get("speaker_id", "")).strip():
        raise ValueError("speaker_id must not be empty")
    if not isinstance(row.get("transcript"), str) or not row["transcript"].strip():
        raise ValueError("transcript must be a non-empty string")

    # Language ratios are audited, not silently repaired.
    try:
        float(row["ratio_english_words"])
        float(row["ratio_hindi_words"])
    except (TypeError, ValueError) as exc:
        raise ValueError("language ratios must be numeric") from exc


def invalid_mucs_fields(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return invalid metadata fields without changing their raw values."""
    invalid: list[str] = []
    english = row.get("ratio_english_words")
    hindi = row.get("ratio_hindi_words")
    try:
        english_value = float(english)
        if not 0.0 <= english_value <= 1.0:
            invalid.append("ratio_english_words")
    except (TypeError, ValueError):
        invalid.append("ratio_english_words")

    try:
        hindi_value = float(hindi)
        if not 0.0 <= hindi_value <= 1.0:
            invalid.append("ratio_hindi_words")
    except (TypeError, ValueError):
        invalid.append("ratio_hindi_words")

    # A valid language composition should not represent more than all words.
    counts = None
    try:
        counts = int(row.get("count_english_words", 0)) + int(row.get("count_dev_words", 0))
    except (TypeError, ValueError):
        invalid.append("word_counts")
    if counts is not None and counts > 0:
        english_number = row.get("count_english_words", 0)
        try:
            english_count = int(english_number)
            if english_count < 0 or english_count > counts:
                invalid.append("count_english_words")
        except (TypeError, ValueError):
            invalid.append("count_english_words")

    # The HF metadata can disagree with counts; record this without changing
    # either value. A recomputation policy is available separately.
    try:
        declared_english = float(english)
        counted_ratio = None
        if counts and counts > 0:
            counted_ratio = int(row.get("count_english_words", 0)) / counts
        if counted_ratio is not None and abs(declared_english - counted_ratio) > 0.05:
            invalid.append("ratio_english_words")
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    return tuple(dict.fromkeys(invalid))


def recompute_english_ratio_from_counts(
    row: Mapping[str, Any],
    *,
    policy: str = "count_ratio",
) -> float | None:
    """Recompute English ratio using an explicit documented policy."""
    if policy != "count_ratio":
        raise ValueError(f"unknown recompute policy: {policy}")
    try:
        english_count = int(row.get("count_english_words", 0))
        hindi_count = int(row.get("count_dev_words", 0))
    except (TypeError, ValueError):
        return None
    total = english_count + hindi_count
    if total <= 0:
        return None
    if english_count < 0:
        return None
    return english_count / total


def mucs_row_to_record(
    row: Mapping[str, Any],
    *,
    split: str,
    recompute_policy: str | None = None,
) -> MucsHinglishRecord:
    """Convert a MUCS row into a normalized manifest record."""
    validate_mucs_row(row)
    invalid_fields = invalid_mucs_fields(row)
    recomputed = None
    if recompute_policy is not None:
        recomputed = recompute_english_ratio_from_counts(row, policy=recompute_policy)
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
        metadata_valid=not invalid_fields,
        invalid_fields=invalid_fields,
        recomputed_ratio_english_words=recomputed,
    )


def load_mucs_hinglish_manifest(
    split: str,
    *,
    max_items: int | None = None,
    dataset_name: str = MUCS_HINGLISH_DATASET,
    revision: str | None = MUCS_HINGLISH_REVISION,
    loader: Callable[[str, str, bool], Any] | None = None,
    streaming: bool = True,
    recompute_policy: str | None = None,
) -> list[MucsHinglishRecord]:
    """Load metadata without decoding or downloading audio."""
    if max_items is not None and max_items < 0:
        raise ValueError("max_items must be non-negative")

    resolved_loader = loader
    if resolved_loader is None:
        resolved_loader = lambda name, split_name, use_streaming: _load_dataset(
            name,
            split=split_name,
            streaming=use_streaming,
            revision=revision,
        )

    dataset = resolved_loader(dataset_name, split, streaming)
    features = getattr(dataset, "features", None)
    if isinstance(features, dict) and "audio" not in features:
        raise ValueError("dataset is missing required audio column")
    # Hugging Face's Audio feature decodes on access. Removing the column makes
    # this manifest operation metadata-only and prevents accidental downloads.
    if hasattr(dataset, "remove_columns"):
        dataset = dataset.remove_columns(["audio"])

    records: list[MucsHinglishRecord] = []
    for index, row in enumerate(dataset):
        if max_items is not None and index >= max_items:
            break
        records.append(
            mucs_row_to_record(row, split=split, recompute_policy=recompute_policy)
        )
    return records


def validate_mucs_manifest(records: Iterable[MucsHinglishRecord]) -> list[MucsHinglishRecord]:
    """Return materialized validated records."""
    materialized = list(records)
    for record in materialized:
        if not record.clip_id:
            raise ValueError("clip_id must not be empty")
        if not record.speaker_id:
            raise ValueError("speaker_id must not be empty")
    return materialized


def summarize_mucs_data_quality(records: Iterable[MucsHinglishRecord]) -> dict[str, object]:
    """Summarize metadata anomalies before evaluation."""
    materialized = list(records)
    invalid_records = [record for record in materialized if not record.metadata_valid]
    field_counts: dict[str, int] = {}
    for record in invalid_records:
        for field_name in record.invalid_fields:
            field_counts[field_name] = field_counts.get(field_name, 0) + 1

    recomputed_count = sum(
        record.recomputed_ratio_english_words is not None for record in materialized
    )
    return {
        "records": len(materialized),
        "valid_records": len(materialized) - len(invalid_records),
        "invalid_records": len(invalid_records),
        "invalid_field_counts": field_counts,
        "recomputed_ratio_records": recomputed_count,
        "license_note": MUCS_LICENSE_NOTE,
    }


def export_mucs_manifest(records: Iterable[MucsHinglishRecord]) -> list[dict[str, object]]:
    """Convert records into a plain dictionary list."""
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
            "metadata_valid": record.metadata_valid,
            "invalid_fields": ";".join(record.invalid_fields),
            "recomputed_ratio_english_words": record.recomputed_ratio_english_words,
        }
        for record in records
    ]


def audit_split_records(
    split_records: dict[str, Iterable[MucsHinglishRecord]],
    *,
    expected_speakers: set[str] | None = None,
) -> HinglishSplitAudit:
    """Audit Hinglish splits for speaker and clip leakage."""
    checked_splits = tuple(sorted(split_records))
    speakers_by_split: dict[str, set[str]] = {}
    segments_by_split: dict[str, list[MucsHinglishRecord]] = {}
    empty_speaker_ids: list[str] = []

    for split, records in split_records.items():
        materialized = list(records)
        speakers: set[str] = set()
        for record in materialized:
            if not record.speaker_id:
                empty_speaker_ids.append(record.clip_id)
                continue
            speakers.add(record.speaker_id)
        speakers_by_split[split] = speakers
        segments_by_split[split] = materialized

    speaker_overlaps: list[str] = []
    for first, second in _pairs(checked_splits):
        speaker_overlaps.extend(
            sorted(speakers_by_split[first] & speakers_by_split[second])
        )

    counts_by_split: dict[str, dict[str, int]] = {}
    for split in checked_splits:
        counts: dict[str, int] = {}
        for record in segments_by_split[split]:
            counts[record.clip_id] = counts.get(record.clip_id, 0) + 1
        counts_by_split[split] = counts

    within_duplicates: set[str] = set()
    for split in checked_splits:
        within_duplicates.update(
            clip_id for clip_id, count in counts_by_split[split].items() if count > 1
        )

    split_for_clip: dict[str, set[str]] = {}
    for split in checked_splits:
        for record in segments_by_split[split]:
            split_for_clip.setdefault(record.clip_id, set()).add(split)
    cross_split_duplicates: set[str] = {
        clip_id for clip_id, splits in split_for_clip.items() if len(splits) > 1
    }

    all_duplicate_ids = within_duplicates | cross_split_duplicates
    assigned_speakers = set().union(*speakers_by_split.values()) if speakers_by_split else set()
    missing_speakers = []
    if expected_speakers is not None:
        missing_speakers = sorted(expected_speakers - assigned_speakers)

    return HinglishSplitAudit(
        speaker_overlaps=tuple(sorted(set(speaker_overlaps))),
        duplicate_segment_ids=tuple(sorted(all_duplicate_ids)),
        duplicate_within_split=tuple(sorted(within_duplicates)),
        cross_split_duplicates=tuple(sorted(cross_split_duplicates)),
        missing_speakers=tuple(missing_speakers),
        empty_speaker_ids=tuple(sorted(set(empty_speaker_ids))),
        checked_splits=checked_splits,
    )


def select_one_per_speaker(
    records: Iterable[MucsHinglishRecord],
    *,
    speaker_limit: int | None = None,
) -> list[MucsHinglishRecord]:
    """Select one deterministic record per speaker.

    Speakers are considered in first-seen order. The function does not alter
    transcripts. It can return a small cohort for diagnostics or all available
    speakers for validation-set construction.
    """
    if speaker_limit is not None and speaker_limit < 0:
        raise ValueError("speaker_limit must be non-negative")

    selected: list[MucsHinglishRecord] = []
    seen: set[str] = set()
    for record in records:
        if not record.speaker_id:
            continue
        if record.speaker_id in seen:
            continue
        seen.add(record.speaker_id)
        selected.append(record)
        if speaker_limit is not None and speaker_limit > 0 and len(selected) >= speaker_limit:
            break
    if not selected:
        raise ValueError("records must contain at least one speaker")
    return selected


def split_speakers_for_validation(
    speakers: Iterable[str],
    *,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Deterministically split training speakers into train and validation."""
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

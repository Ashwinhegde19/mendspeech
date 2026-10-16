"""Unit tests for Hinglish metadata helpers."""

from __future__ import annotations

import pytest

from src.hinglish.data import (
    MUCS_HINGLISH_DATASET,
    HinglishSplitAudit,
    MucsHinglishRecord,
    audit_split_records,
    load_mucs_hinglish_manifest,
    mucs_row_to_record,
    split_speakers_for_validation,
    validate_mucs_row,
)


def test_validates_required_columns_and_values() -> None:
    row = {
        "segment_id": "utt_1",
        "speaker_id": "spk_1",
        "transcript": "hello नमस्ते",
        "audio": {},
        "ratio_english_words": 0.5,
        "ratio_hindi_words": 0.5,
    }
    validate_mucs_row(row)
    record = mucs_row_to_record(row, split="train")
    assert isinstance(record, MucsHinglishRecord)
    assert record.clip_id == "utt_1"
    assert record.speaker_id == "spk_1"


def test_validation_rejects_invalid_language_ratio() -> None:
    row = {
        "segment_id": "utt_1",
        "speaker_id": "spk_1",
        "transcript": "hello",
        "audio": {},
        "ratio_english_words": 1.5,
        "ratio_hindi_words": 0.5,
    }
    with pytest.raises(ValueError):
        validate_mucs_row(row)


def test_loader_with_fake_iterator_respects_maximum_items() -> None:
    rows = [
        {
            "segment_id": f"utt_{index}",
            "speaker_id": f"spk_{index}",
            "transcript": f"hello {index}",
            "audio": {},
            "ratio_english_words": 0.25,
            "ratio_hindi_words": 0.75,
        }
        for index in range(5)
    ]
    records = load_mucs_hinglish_manifest(
        "train",
        max_items=3,
        loader=lambda name, split, streaming: iter(rows),
    )
    assert len(records) == 3
    assert records[0].clip_id == "utt_0"


def test_loader_rejects_negative_maximum_items() -> None:
    with pytest.raises(ValueError):
        load_mucs_hinglish_manifest(
            "train",
            max_items=-1,
            loader=lambda name, split, streaming: iter([]),
        )


def test_split_speakers_is_deterministic_and_speaker_disjoint() -> None:
    speakers = [f"spk_{index}" for index in range(20)]
    train_first, validation_first = split_speakers_for_validation(
        speakers, validation_ratio=0.25, seed=7
    )
    train_second, validation_second = split_speakers_for_validation(
        speakers, validation_ratio=0.25, seed=7
    )
    assert train_first == train_second
    assert validation_first == validation_second
    assert set(train_first).isdisjoint(validation_first)


def test_audit_detects_speaker_overlap_and_duplicates() -> None:
    train_records = [
        MucsHinglishRecord(
            clip_id="utt_1",
            speaker_id="spk_1",
            split="train",
            transcript="hello",
            ratio_english_words=0.5,
            ratio_hindi_words=0.5,
            english_words="hello",
            count_english_words=1,
            count_hindi_words=1,
            segment_part="0000",
        )
    ]
    validation_records = [
        MucsHinglishRecord(
            clip_id="utt_1",
            speaker_id="spk_1",
            split="validation",
            transcript="hello",
            ratio_english_words=0.5,
            ratio_hindi_words=0.5,
            english_words="hello",
            count_english_words=1,
            count_hindi_words=1,
            segment_part="0000",
        )
    ]
    audit = audit_split_records({"train": train_records, "validation": validation_records})
    assert "spk_1" in audit.speaker_overlaps
    assert "utt_1" in audit.duplicate_segment_ids
    assert audit.passed is False


def test_mucs_dataset_name_constant() -> None:
    assert MUCS_HINGLISH_DATASET == "dianavdavidson/MUCS-Hinglish"

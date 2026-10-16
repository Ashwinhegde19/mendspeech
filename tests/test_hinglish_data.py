"""Unit tests for Hinglish metadata helpers."""

from __future__ import annotations

import pytest

from src.hinglish.data import (
    MUCS_HINGLISH_DATASET,
    MucsHinglishRecord,
    audit_split_records,
    load_mucs_hinglish_manifest,
    mucs_row_to_record,
    split_speakers_for_validation,
    summarize_mucs_data_quality,
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
    assert record.clip_id == "utt_1"
    assert record.speaker_id == "spk_1"


def test_validation_flags_invalid_language_ratio() -> None:
    row = {
        "segment_id": "utt_1",
        "speaker_id": "spk_1",
        "transcript": "hello",
        "audio": {},
        "ratio_english_words": 1.5,
        "ratio_hindi_words": 0.5,
        "count_english_words": 1,
        "count_dev_words": 1,
    }
    validate_mucs_row(row)
    record = mucs_row_to_record(row, split="train")
    assert record.ratio_english_words == 1.5
    assert record.metadata_valid is False
    assert "ratio_english_words" in record.invalid_fields
    assert record.recomputed_ratio_english_words is None


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


def test_audit_passes_for_valid_speaker_disjoint_split() -> None:
    train_records = [
        MucsHinglishRecord(
            clip_id="train_1",
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
            clip_id="val_1",
            speaker_id="spk_2",
            split="validation",
            transcript="hello नमस्ते",
            ratio_english_words=0.5,
            ratio_hindi_words=0.5,
            english_words="hello",
            count_english_words=1,
            count_hindi_words=1,
            segment_part="0000",
        )
    ]
    audit = audit_split_records(
        {"train": train_records, "validation": validation_records}
    )
    assert audit.passed is True
    assert audit.speaker_overlaps == ()
    assert audit.cross_split_duplicates == ()
    assert audit.missing_speakers == ()

    missing_audit = audit_split_records(
        {"train": train_records, "validation": validation_records},
        expected_speakers={"spk_1", "spk_2", "spk_3"},
    )
    assert "spk_3" in missing_audit.missing_speakers
    assert missing_audit.passed is False


def test_audit_detects_speaker_overlap() -> None:
    records = [
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
    audit = audit_split_records({"train": records, "validation": records})
    assert "spk_1" in audit.speaker_overlaps
    assert audit.passed is False


def test_audit_detects_duplicate_clip_across_splits() -> None:
    record = MucsHinglishRecord(
        clip_id="shared",
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
    validation_record = MucsHinglishRecord(
        clip_id="shared",
        speaker_id="spk_2",
        split="validation",
        transcript="hello",
        ratio_english_words=0.5,
        ratio_hindi_words=0.5,
        english_words="hello",
        count_english_words=1,
        count_hindi_words=1,
        segment_part="0000",
    )
    audit = audit_split_records(
        {"train": [record], "validation": [validation_record]}
    )
    assert "shared" in audit.cross_split_duplicates
    assert "shared" in audit.duplicate_segment_ids
    assert audit.passed is False


def test_recomputes_ratio_using_documented_count_policy() -> None:
    row = {
        "segment_id": "utt_1",
        "speaker_id": "spk_1",
        "transcript": "hello",
        "audio": {},
        "ratio_english_words": 0.99,
        "ratio_hindi_words": 0.01,
        "count_english_words": 1,
        "count_dev_words": 3,
    }
    record = mucs_row_to_record(row, split="train", recompute_policy="count_ratio")
    assert record.ratio_english_words == 0.99
    assert record.recomputed_ratio_english_words == 0.25


def test_data_quality_summary_preserves_invalid_values() -> None:
    row = {
        "segment_id": "utt_1",
        "speaker_id": "spk_1",
        "transcript": "hello",
        "audio": {},
        "ratio_english_words": 1.2,
        "ratio_hindi_words": 0.5,
        "count_english_words": 5,
        "count_dev_words": 5,
    }
    record = mucs_row_to_record(row, split="train")
    assert record.ratio_english_words == 1.2
    summary = summarize_mucs_data_quality([record])
    assert summary["invalid_records"] == 1
    assert summary["invalid_field_counts"]["ratio_english_words"] == 1
    assert "OpenSLR" in summary["license_note"]


def test_mucs_dataset_name_constant() -> None:
    assert MUCS_HINGLISH_DATASET == "dianavdavidson/MUCS-Hinglish"

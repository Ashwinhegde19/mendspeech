#!/usr/bin/env python3
"""Create the speaker-disjoint MUCS Hinglish validation cohort.

This script consumes the complete locally collected MUCS training metadata
inventory (no audio), validates it, deterministically splits speakers into
train/validation cohorts with seed 42, excludes previously analyzed
diagnostic speakers from the validation pool, stratifies the selected
validation speakers across language-composition groups, and exports the
frozen manifest plus machine-readable audits outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hinglish.data import (
    MUCS_HINGLISH_REVISION,
    audit_split_records,
    mucs_row_to_record,
    split_speakers_for_validation,
)

SELECTION_VERSION = "mendspeech.mucs_validation.v1"
DIAGNOSTIC_SPEAKERS_DEFAULT = (
    "100051",
    "100356",
    "102503",
    "104662",
    "106319",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_inventory(
    rows: list[dict[str, str]],
    *,
    expected_rows: int | None,
    expected_checksum: str | None,
    path: Path,
) -> dict[str, object]:
    actual_checksum = _sha256(path)
    if expected_checksum is not None and actual_checksum != expected_checksum:
        raise SystemExit(
            f"inventory checksum mismatch: expected {expected_checksum}, "
            f"actual {actual_checksum}"
        )
    if expected_rows is not None and len(rows) != expected_rows:
        raise SystemExit(
            f"inventory row mismatch: expected {expected_rows}, actual {len(rows)}"
        )

    segment_ids = [row["segment_id"] for row in rows]
    speaker_ids = [row["speaker_id"] for row in rows]
    duplicated_segments = [
        clip for clip, count in Counter(segment_ids).items() if count > 1
    ]
    if duplicated_segments:
        raise SystemExit(f"inventory contains duplicate segment ids: {duplicated_segments[:10]}")

    empty_segments = [row["segment_id"] for row in rows if not row["segment_id"].strip()]
    empty_speakers = [row["speaker_id"] for row in rows if not row["speaker_id"].strip()]
    empty_transcripts = [row["segment_id"] for row in rows if not row["transcript"].strip()]
    if empty_segments or empty_speakers:
        raise SystemExit(
            "inventory contains empty identifiers: "
            f"segment={len(empty_segments)}, speaker={len(empty_speakers)}"
        )

    shards = sorted({int(row["shard"]) for row in rows})
    expected_shards = list(range(len(shards)))
    if shards != expected_shards:
        raise SystemExit(f"inventory shard boundaries are not contiguous: {shards}")
    shards = [str(index) for index in shards]

    invalid_ratio_rows = []
    for row in rows:
        try:
            english = float(row["ratio_english_words"])
            hindi = float(row["ratio_hindi_words"])
        except (TypeError, ValueError):
            invalid_ratio_rows.append(row["segment_id"])
            continue
        if not (0.0 <= english <= 1.0 and 0.0 <= hindi <= 1.0):
            invalid_ratio_rows.append(row["segment_id"])
        elif abs(english + hindi - 1.0) > 1e-6:
            invalid_ratio_rows.append(row["segment_id"])

    return {
        "rows": len(rows),
        "distinct_speakers": len(set(speaker_ids)),
        "distinct_segments": len(set(segment_ids)),
        "duplicate_segments": [],  # already rejected above
        "empty_transcripts": len(empty_transcripts),
        "empty_speaker_ids": [],  # already rejected above
        "shards": shards,
        "invalid_ratio_rows": invalid_ratio_rows,
        "checksum": actual_checksum,
    }


def _convert(rows: list[dict[str, str]]):
    return [
        mucs_row_to_record(
            {
                "segment_id": row["segment_id"],
                "speaker_id": row["speaker_id"],
                "transcript": row["transcript"],
                "ratio_english_words": float(row["ratio_english_words"]),
                "ratio_hindi_words": float(row["ratio_hindi_words"]),
                "count_english_words": int(row["count_english_words"] or 0),
                "count_dev_words": int(row["count_dev_words"] or 0),
                "segment_part": row.get("segment_part", ""),
                "english_words": row.get("english_words", "") if "english_words" in row else "",
            },
            split="train",
        )
        for row in rows
    ]


def _bucket(ratio_english: float) -> str:
    if ratio_english >= 0.6:
        return "english_dominant"
    if ratio_english <= 0.4:
        return "hindi_dominant"
    return "mixed_code"


def _speaker_buckets(records) -> dict[str, str]:
    sums = defaultdict(float)
    counts = defaultdict(int)
    for record in records:
        if not record.metadata_valid:
            continue
        sums[record.speaker_id] += record.ratio_english_words
        counts[record.speaker_id] += 1
    return {
        speaker: _bucket(sums[speaker] / counts[speaker])
        for speaker in counts
    }


def _select_speakers_stratified(
    validation_speakers: list[str],
    records,
    *,
    target: int,
) -> list[str]:
    global_buckets = _speaker_buckets(records)
    global_counts = Counter(global_buckets.values())
    total_buck = sum(global_counts.values())


    # Speaker-level bucket assignment (with a deterministic fallback for
    # speakers missing from the global statistics).
    per_speaker_bucket = {
        speaker: global_buckets.get(speaker, "mixed_code")
        for speaker in validation_speakers
    }
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for speaker in validation_speakers:
        by_bucket[per_speaker_bucket[speaker]].append(speaker)

    # Allocate proportional to global inventory frequencies, restricted to
    # buckets actually present in the validation pool.
    allowed_buckets = sorted(by_bucket)
    proportions = {
        bucket: global_counts.get(bucket, 0) for bucket in allowed_buckets
    }
    total_global = sum(proportions.values())
    if total_global == 0:
        allocations = {bucket: 1 for bucket in allowed_buckets}
    else:
        allocations = {
            bucket: max(1, round((count / total_global) * target))
            for bucket, count in proportions.items()
        }

    # Adjust allocation to exactly `target`.
    current = sum(allocations.values())
    ordered_buckets = allowed_buckets
    while current != target:
        for bucket in ordered_buckets:
            if current == target:
                break
            if current < target:
                allocations[bucket] += 1
                current += 1
            elif allocations[bucket] > 1:
                allocations[bucket] -= 1
                current -= 1

    selected: list[str] = []
    leftovers: list[str] = []
    for bucket in ordered_buckets:
        pool = sorted(by_bucket[bucket])
        take = min(allocations[bucket], len(pool))
        selected.extend(pool[:take])
        leftovers.extend(pool[take:])
    # Backfill from leftovers if a bucket was short while another held extra
    # capacity implied by unselected speakers.
    filler_needed = target - len(selected)
    if filler_needed > 0:
        selected.extend(leftovers[:filler_needed])
        if len(selected) < target:
            raise SystemExit(
                "validation pool does not contain enough speakers for stratified selection"
            )
    if len(selected) > target:
        raise SystemExit("stratified selection over-allocated speakers")
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically freeze the MUCS Hinglish validation cohort"
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument(
        "--validation-speakers", type=int, default=30,
        help="Number of held-out validation speakers to select.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-speakers",
        default=",".join(DIAGNOSTIC_SPEAKERS_DEFAULT),
        help="Speakers reserved for the training pool (never validation).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.validation_speakers < 1:
        raise SystemExit("--validation-speakers must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_inventory(args.inventory)
    inventory_stats = _validate_inventory(
        rows,
        expected_rows=args.expected_rows,
        expected_checksum=args.inventory_sha256,
        path=args.inventory,
    )
    inventory_stats["inventory_path"] = str(args.inventory.resolve())

    records = _convert(rows)

    excluded = [value.strip() for value in args.exclude_speakers.split(",") if value.strip()]
    all_speakers = sorted({record.speaker_id for record in records})
    missing_exclusions = [speaker for speaker in excluded if speaker not in all_speakers]
    if missing_exclusions:
        raise SystemExit(
            f"excluded speakers not present in inventory: {missing_exclusions}"
        )

    # Step 1: deterministic per-seed split.
    train_assignment, validation_assignment = split_speakers_for_validation(
        [speaker for speaker in all_speakers if speaker not in excluded],
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    # Step 2: force diagnostic speakers into the training pool and backfill
    # validation if any of them would otherwise have landed there.
    retained_validation = [
        speaker for speaker in validation_assignment if speaker not in excluded
    ]
    forced_into_training = sorted(
        set(validation_assignment) & set(excluded)
    )
    if forced_into_training:
        validation_pool_ranked = sorted(
            set(all_speakers) - set(excluded) - set(train_assignment) - set(retained_validation)
        )
        retained_validation.extend(
            validation_pool_ranked[: len(forced_into_training)]
        )

    validation_sorted = sorted(set(retained_validation))
    training_sorted = sorted(
        set(train_assignment) | (set(all_speakers) - set(validation_sorted) - set(excluded))
    )
    # The excluded diagnostic speakers stay in the training pool.
    training_sorted = sorted(set(training_sorted) | set(excluded))

    # Validate speaker disjointness.
    if set(validation_sorted) & set(training_sorted):
        raise SystemExit("speaker split produced overlapping train and validation pools")

    # Step 3: stratify validation-speaker selection across language-composition
    # groups using the per-speaker average ratio. Then select one record per
    # chosen speaker deterministically by smallest clip_id.
    chosen_speakers = _select_speakers_stratified(
        validation_sorted,
        records,
        target=args.validation_speakers,
    )

    by_speaker: dict[str, list] = defaultdict(list)
    for record in records:
        by_speaker[record.speaker_id].append(record)

    stratification_buckets = _speaker_buckets(records)
    manifest_rows: list[dict[str, object]] = []
    for speaker in chosen_speakers:
        pool = by_speaker[speaker]
        eligible = [record for record in pool if record.metadata_valid]
        if not eligible:
            invalid_fields = sorted(
                {field for record in pool for field in record.invalid_fields}
            )
            eligible = [
                record
                for record in pool
                if record.transcript.strip() and "transcript" not in invalid_fields
            ]
        if not eligible:
            raise SystemExit(f"speaker {speaker} has no eligible validation record")
        chosen = min(eligible, key=lambda record: record.clip_id)
        stratification_bucket = stratification_buckets.get(speaker, "mixed_code")
        manifest_rows.append(
            {
                "clip_id": chosen.clip_id,
                "speaker_id": chosen.speaker_id,
                "source_partition": "train",
                "assigned_role": "validation",
                "reference_transcript": chosen.transcript,
                "ratio_english_words": chosen.ratio_english_words,
                "ratio_hindi_words": chosen.ratio_hindi_words,
                "stratification_bucket": stratification_bucket,
                "record_language_bucket": _bucket(chosen.ratio_english_words),
                "count_english_words": chosen.count_english_words,
                "count_hindi_words": chosen.count_hindi_words,
                "segment_part": chosen.segment_part,
                "metadata_valid": chosen.metadata_valid,
                "invalid_fields": ";".join(chosen.invalid_fields) or "none",
                "dataset_revision": MUCS_HINGLISH_REVISION,
                "split_seed": args.seed,
                "selection_version": SELECTION_VERSION,
            }
        )

    manifest_rows.sort(key=lambda row: str(row["clip_id"]))

    # Step 4: audit leakage. The validation side covers every record from the
    # complete 77-speaker held-out pool, not only the 30 sampled clips.
    split_audit = audit_split_records(
        {
            "train": _split_records(records, training_sorted),
            "validation": validation_pool_records(records, validation_sorted),
        },
        expected_speakers=set(all_speakers),
    )
    validation_pool = list(validation_pool_records(records, validation_sorted))
    pool_stats = {
        "validation_pool_speakers": len(validation_sorted),
        "validation_pool_records": len(validation_pool),
        "validation_pool_empty_transcripts": sum(
            1 for record in validation_pool if not record.transcript.strip()
        ),
        "validation_pool_invalid_metadata_records": sum(
            1 for record in validation_pool if not record.metadata_valid
        ),
        "validation_pool_invalid_clip_ids": sorted(
            record.clip_id for record in validation_pool if not record.metadata_valid
        ),
        "validation_pool_bucket_distribution": dict(
            Counter(
                _speaker_buckets(records).get(speaker, "mixed_code")
                for speaker in validation_sorted
            )
        ),
        "sampled_clip_bucket_distribution": dict(
            Counter(
                stratification_buckets.get(speaker, "mixed_code")
                for speaker in chosen_speakers
            )
        ),
    }
    leakage_audit = {
        "speaker_overlap": split_audit.speaker_overlaps,
        "duplicate_clips_within_split": split_audit.duplicate_within_split,
        "duplicate_clips_across_split": split_audit.cross_split_duplicates,
        "missing_speakers_not_assigned": split_audit.missing_speakers,
        "diagnostic_speakers_in_validation": sorted(
            set(chosen_speakers) & set(excluded)
        ),
        "diagnostic_speakers_all_in_training": sorted(set(excluded) - set(training_sorted)),
        "recording_separation_limitation": (
            "MUCS-Hinglish metadata exposes segment_part and speaker_id but not "
            "the underlying source recording identifier. Speaker-level "
            "separation was audited; recording-level overlap cannot be "
            "verified from the available fields."
        ),
        "parquet_source_revision_limitation": (
            "The Hugging Face refs%2Fconvert%2Fparquet conversion URL is not "
            "itself revision-pinned to the dataset revision; the collected "
            "inventory and the frozen cohort are protected by SHA-256 "
            "checksums instead."
        ),
        "validation_pool_audit": pool_stats,
    }

    language_distribution = Counter(
        str(row["stratification_bucket"]) for row in manifest_rows
    )
    record_language_distribution = Counter(
        str(row["record_language_bucket"]) for row in manifest_rows
    )

    outputs = {
        "inventory_stats.json": {
            "dataset": "dianavdavidson/MUCS-Hinglish",
            "revision": MUCS_HINGLISH_REVISION,
            "split": "train",
            **inventory_stats,
        },
        "speaker_split.json": {
            "seed": args.seed,
            "validation_ratio": args.validation_ratio,
            "training_speakers": training_sorted,
            "validation_speakers": validation_sorted,
            "diagnostic_speakers_reserved_for_training": excluded,
            "forced_into_training_from_seed_split": forced_into_training,
            "selected_validation_speakers": chosen_speakers,
            "stratification_bucket_distribution": dict(
                Counter(
                    _speaker_buckets(records).get(speaker, "mixed_code")
                    for speaker in chosen_speakers
                )
            ),
        },
        "leakage_audit.json": leakage_audit,
        "selection_policy.json": {
            "selection_version": SELECTION_VERSION,
            "dataset_revision": MUCS_HINGLISH_REVISION,
            "seed": args.seed,
            "validation_ratio": args.validation_ratio,
            "selection_policy": (
                "1. Load the complete training-split metadata inventory and "
                "validate row count, checksum, shard coverage, and identifier "
                "uniqueness.\n"
                "2. Apply a seeded deterministic speaker split (seed 42, "
                "target ratio 0.15) over the full speaker inventory.\n"
                "3. Reserve the previously analyzed diagnostic speakers for "
                "the training pool.\n"
                "4. Stratify the 30 held-out validation speakers across "
                "Hindi-dominant, mixed-code, and English-dominant buckets "
                "proportional to their global inventory frequency. The "
                "stratification key is the per-speaker average English ratio; "
                "the manifest separately records the selected utterance's "
                "own composition.\n"
                "5. For each chosen speaker, select the eligible record with "
                "the smallest clip_id.\n"
                "6. Audit speaker and clip overlap; flag recording-level "
                "separation as not verifiable because the source-recording "
                "identifier is not exposed."
            ),
        },
        "selected_validation_manifest.csv": manifest_rows,
    }

    frozen_manifest_path = output_dir / "frozen_validation_manifest.csv"
    fieldnames = list(manifest_rows[0].keys())
    with frozen_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    manifest_checksum = _sha256(frozen_manifest_path)
    (output_dir / "frozen_validation_manifest.sha256").write_text(
        f"{manifest_checksum}  frozen_validation_manifest.csv\n", encoding="utf-8"
    )

    # Machine-readable "clip -> audio filename" mapping.
    clips_for_audio = [
        {"clip_id": row["clip_id"], "audio_filename": f"{row['clip_id']}.wav"}
        for row in manifest_rows
    ]
    with (output_dir / "clips_for_audio.json").open("w", encoding="utf-8") as handle:
        json.dump(clips_for_audio, handle, indent=2)
        handle.write("\n")

    for filename, payload in outputs.items():
        path = output_dir / filename
        if filename.endswith(".json"):
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Inventory stats: {len(records)} rows, {len(all_speakers)} speakers")
    print(f"Training pool: {len(training_sorted)} speakers")
    print(f"Validation pool (before selection): {len(validation_sorted)} speakers")
    print(f"Stratified selected validation speakers: {len(chosen_speakers)}")
    print(f"Selected clips: {len(manifest_rows)}")
    print(f"Stratification bucket distribution: {dict(language_distribution)}")
    print(f"Selected-utterance record-level bucket distribution: "
          f"{dict(record_language_distribution)}")
    print(f"Speaker overlap: {split_audit.speaker_overlaps}")
    print(f"Duplicate clips: {split_audit.duplicate_segment_ids}")
    print(f"Excluded diagnostic speakers all in training pool: "
          f"{not (set(excluded) & set(chosen_speakers))}")
    print(f"Frozen manifest: {frozen_manifest_path}")
    print(f"Manifest sha256: {manifest_checksum}")
    return 0


def _split_records(records, training_speakers):
    for record in records:
        if record.speaker_id in training_speakers:
            yield record


def validation_pool_records(records, validation_speakers):
    """Yield every record belonging to validation-pool speakers."""
    pool = set(validation_speakers)
    for record in records:
        if record.speaker_id in pool:
            yield record


if __name__ == "__main__":
    raise SystemExit(main())

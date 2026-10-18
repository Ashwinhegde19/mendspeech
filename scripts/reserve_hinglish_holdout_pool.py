#!/usr/bin/env python3
"""Reserve the final held-out MUCS Hinglish candidate evaluation pool.

The phase-6.x validation work carved a speaker-disjoint validation pool of 77
MUCS training-split speakers out of the 520-speaker inventory. Thirty of those
speakers were evaluated as development data, so their results may be inspected
to guide product choices. The remaining 47 speakers are the candidate final
held-out evaluation pool: they must stay untouched until the final evaluation.

This script prepares and freezes that reserve using metadata only. It never
reads, prints, or stores transcripts, and it never touches the official MUCS
test partition. Outputs are private artifacts that must live outside the
repository; the frozen JSON carries the selected clip per speaker plus
eligibility metadata, and a sibling SHA-256 checksum pins the freeze.

Freeze step (run once; the resulting pool must never be modified afterwards):

    scripts/reserve_hinglish_holdout_pool.py \\
        --inventory /path/outside/repo/mucs-training-metadata.csv \\
        --speaker-split /path/outside/repo/speaker_split.json \\
        --inventory-sha256 <inventory digest> \\
        --freeze \\
        --output /path/outside/repo/holdout_pool_v0.json \\
        --checksum /path/outside/repo/holdout_pool_v0.sha256

`--dry-run` prints the same composition summary without writing anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.create_mucs_validation_split import (  # noqa: E402
    _bucket,
    _sha256,
)
from src.hinglish.data import (  # noqa: E402
    MUCS_HINGLISH_REVISION,
    invalid_mucs_fields,
)

POOL_VERSION = "mendspeech.mucs_holdout_pool.v0"
DATASET = "dianavdavidson/MUCS-Hinglish"

# Canonical phase-6.x counts. Overridable on the command line so the same code
# path can be exercised against synthetic fixtures in tests.
EXPECTED_ROWS = 52825
EXPECTED_TOTAL_SPEAKERS = 520
EXPECTED_TRAINING_SPEAKERS = 443
EXPECTED_VALIDATION_SPEAKERS = 77
EXPECTED_SELECTED_SPEAKERS = 30
EXPECTED_REMAINING_SPEAKERS = 47

# This script deliberately does not require the transcript column: the reserve
# is built from metadata alone so transcripts can never influence selection.
REQUIRED_COLUMNS = (
    "segment_id",
    "speaker_id",
    "ratio_english_words",
    "ratio_hindi_words",
    "count_english_words",
    "count_dev_words",
)

SELECTION_POLICY = (
    "1. Candidate speakers are the validation-pool speakers that are not part "
    "of the 30 already-evaluated development speakers.\n"
    "2. Only metadata columns are read (segment_id, speaker_id, language "
    "ratios, word counts, segment_part, shard). Transcripts are never read, "
    "printed, or stored.\n"
    "3. A record is eligible when its metadata is valid under the frozen v1 "
    "validity rule: ratios within [0, 1], ratios summing to 1, and the "
    "declared English ratio consistent with the word counts within 0.05.\n"
    "4. For each candidate speaker, the eligible record with the "
    "lexicographically smallest segment_id (clip_id) is selected. Selection "
    "is deterministic; no randomness or seed is involved.\n"
    "5. If a candidate speaker has no eligible record, the speaker is dropped "
    "and recorded in dropped_ineligible_speakers. Transcript-based fallback "
    "inspection is forbidden for the reserve, unlike the development cohort.\n"
    "6. Speakers are reported in stratified language buckets derived from "
    "their average English ratio over eligible records, using the v1 "
    "boundaries (>= 0.6 english, <= 0.4 hindi, otherwise mixed). Buckets are "
    "for stratified reporting only and do not change the selected clips."
)


def _load_inventory(path: Path) -> list[dict[str, str]]:
    """Load the metadata inventory CSV without requiring transcript columns."""
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise SystemExit(f"inventory is missing required columns: {missing}")
        return list(reader)


def _load_speaker_split(path: Path) -> dict[str, object]:
    """Load the frozen speaker split JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_keys = (
        "training_speakers",
        "validation_speakers",
        "selected_validation_speakers",
    )
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise SystemExit(f"speaker split is missing keys: {missing}")
    return payload


def _compute_state(
    rows: list[dict[str, str]],
    split: dict[str, object],
) -> dict[str, object]:
    """Derive the split state used for validation and selection."""
    inventory_speakers = {row["speaker_id"] for row in rows}
    training = list(split["training_speakers"])
    validation = list(split["validation_speakers"])
    selected = list(split["selected_validation_speakers"])
    diagnostics = list(split.get("diagnostic_speakers_reserved_for_training", []))
    remaining = sorted(set(validation) - set(selected))
    return {
        "rows": len(rows),
        "inventory_speakers": inventory_speakers,
        "training_speakers": training,
        "validation_speakers": validation,
        "selected_speakers": selected,
        "remaining_speakers": remaining,
        "diagnostic_speakers": diagnostics,
    }


def _validate_state(
    state: dict[str, object],
    *,
    expected_rows: int,
    expected_total_speakers: int,
    expected_training_speakers: int,
    expected_validation_speakers: int,
    expected_selected_speakers: int,
    expected_remaining_speakers: int,
) -> None:
    """Validate the frozen counts and speaker-role disjointness."""
    for label in ("training", "validation", "selected"):
        values = state[f"{label}_speakers"]
        duplicates = sorted(
            speaker for speaker, count in Counter(values).items() if count > 1
        )
        if duplicates:
            raise SystemExit(f"speaker split contains duplicate {label} speakers: {duplicates}")

    training = set(state["training_speakers"])
    validation = set(state["validation_speakers"])
    selected = set(state["selected_speakers"])
    remaining = set(state["remaining_speakers"])
    diagnostics = set(state["diagnostic_speakers"])

    overlap = sorted(training & validation)
    if overlap:
        raise SystemExit(
            f"training and validation speakers overlap: {overlap}"
        )

    inventory_speakers = set(state["inventory_speakers"])
    unassigned = sorted(inventory_speakers - training - validation)
    unknown = sorted((training | validation) - inventory_speakers)
    if unassigned or unknown:
        raise SystemExit(
            "speaker split does not match the inventory: "
            f"unassigned={unassigned[:10]}, unknown={unknown[:10]}"
        )

    if state["rows"] != expected_rows:
        raise SystemExit(
            f"inventory row mismatch: expected {expected_rows}, actual {state['rows']}"
        )
    if len(inventory_speakers) != expected_total_speakers:
        raise SystemExit(
            "inventory speaker mismatch: "
            f"expected {expected_total_speakers}, actual {len(inventory_speakers)}"
        )
    if len(training) != expected_training_speakers:
        raise SystemExit(
            "training speaker count mismatch: "
            f"expected {expected_training_speakers}, actual {len(training)}"
        )
    if len(validation) != expected_validation_speakers:
        raise SystemExit(
            "validation speaker count mismatch: "
            f"expected {expected_validation_speakers}, actual {len(validation)}"
        )
    if len(selected) != expected_selected_speakers:
        raise SystemExit(
            "selected development speaker count mismatch: "
            f"expected {expected_selected_speakers}, actual {len(selected)}"
        )
    if not selected <= validation:
        raise SystemExit(
            "selected development speakers are not a subset of the validation "
            f"pool: {sorted(selected - validation)}"
        )
    if len(remaining) != expected_remaining_speakers:
        raise SystemExit(
            "held-out candidate speaker count mismatch: "
            f"expected {expected_remaining_speakers}, actual {len(remaining)}"
        )
    if remaining & training:
        raise SystemExit(
            f"held-out candidates overlap training speakers: {sorted(remaining & training)}"
        )
    if remaining & selected:
        raise SystemExit(
            f"held-out candidates overlap development speakers: {sorted(remaining & selected)}"
        )
    stray_diagnostics = sorted(diagnostics - training)
    if stray_diagnostics:
        raise SystemExit(
            f"diagnostic speakers are not reserved for training: {stray_diagnostics}"
        )


def _select_pool(
    rows: list[dict[str, str]],
    remaining_speakers: list[str],
) -> tuple[list[dict[str, object]], list[str]]:
    """Select one eligible clip per held-out speaker. Metadata only."""
    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)

    entries: list[dict[str, object]] = []
    dropped: list[str] = []
    for speaker in sorted(remaining_speakers):
        speaker_rows = by_speaker.get(speaker, [])
        invalid_ids = sorted(
            row["segment_id"]
            for row in speaker_rows
            if invalid_mucs_fields(row)
        )
        invalid_set = set(invalid_ids)
        eligible = [row for row in speaker_rows if row["segment_id"] not in invalid_set]
        if not eligible:
            dropped.append(speaker)
            continue

        chosen = min(eligible, key=lambda row: row["segment_id"])
        average_english = sum(
            float(row["ratio_english_words"]) for row in eligible
        ) / len(eligible)
        entries.append(
            {
                "speaker_id": speaker,
                "stratification_bucket": _bucket(average_english),
                "total_clip_count": len(speaker_rows),
                "eligible_clip_count": len(eligible),
                "invalid_metadata_clip_ids": invalid_ids,
                "selected_clip_id": chosen["segment_id"],
                "selected_clip_ratio_english_words": float(
                    chosen["ratio_english_words"]
                ),
                "selected_clip_ratio_hindi_words": float(
                    chosen["ratio_hindi_words"]
                ),
                "selected_clip_language_bucket": _bucket(
                    float(chosen["ratio_english_words"])
                ),
                "selected_clip_segment_part": chosen.get("segment_part", ""),
            }
        )
    return entries, dropped


def _build_pool_payload(
    state: dict[str, object],
    rows: list[dict[str, str]],
    *,
    inventory_path: Path,
    inventory_sha256: str,
    speaker_split_path: Path,
    speaker_split_sha256: str,
) -> dict[str, object]:
    """Build the frozen pool payload (no transcripts anywhere)."""
    entries, dropped = _select_pool(rows, state["remaining_speakers"])
    bucket_distribution = Counter(
        str(entry["stratification_bucket"]) for entry in entries
    )
    clip_bucket_distribution = Counter(
        str(entry["selected_clip_language_bucket"]) for entry in entries
    )
    eligible_counts = [int(entry["eligible_clip_count"]) for entry in entries]
    return {
        "pool_version": POOL_VERSION,
        "dataset": DATASET,
        "dataset_revision": MUCS_HINGLISH_REVISION,
        "source_partition": "train",
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": inventory_sha256,
        "speaker_split_path": str(speaker_split_path.resolve()),
        "speaker_split_sha256": speaker_split_sha256,
        "counts": {
            "inventory_rows": state["rows"],
            "total_speakers": len(state["inventory_speakers"]),
            "training_speakers": len(state["training_speakers"]),
            "validation_pool_speakers": len(state["validation_speakers"]),
            "selected_development_speakers": len(state["selected_speakers"]),
            "held_out_candidate_speakers": len(state["remaining_speakers"]),
        },
        "bucket_distribution": dict(sorted(bucket_distribution.items())),
        "selected_clip_bucket_distribution": dict(
            sorted(clip_bucket_distribution.items())
        ),
        "eligible_clip_count_summary": {
            "total": sum(eligible_counts),
            "min": min(eligible_counts, default=0),
            "max": max(eligible_counts, default=0),
        },
        "dropped_ineligible_speakers": dropped,
        "selection_policy": SELECTION_POLICY,
        "speakers": entries,
    }


def _print_summary(payload: dict[str, object], *, mode: str, output: Path | None) -> None:
    counts = payload["counts"]
    print(f"Mode: {mode}")
    print(
        "Counts: "
        f"training={counts['training_speakers']} "
        f"validation_pool={counts['validation_pool_speakers']} "
        f"development={counts['selected_development_speakers']} "
        f"held_out={counts['held_out_candidate_speakers']} "
        f"total={counts['total_speakers']}"
    )
    print("Held-out candidate speakers:")
    for entry in payload["speakers"]:
        print(
            f"  {entry['speaker_id']}"
            f"  bucket={entry['stratification_bucket']}"
            f"  clips={entry['total_clip_count']}"
            f"  eligible={entry['eligible_clip_count']}"
            f"  selected={entry['selected_clip_id']}"
        )
    dropped = payload["dropped_ineligible_speakers"]
    if dropped:
        print(f"Dropped ineligible speakers: {dropped}")
    print(f"Speaker bucket distribution: {payload['bucket_distribution']}")
    print(
        "Selected-clip bucket distribution: "
        f"{payload['selected_clip_bucket_distribution']}"
    )
    summary = payload["eligible_clip_count_summary"]
    print(
        "Eligible clips: "
        f"total={summary['total']} min={summary['min']} max={summary['max']}"
    )
    if output is not None:
        print(f"Would write pool file to (not written in dry-run): {output}")


def _ensure_outside_repo(path: Path) -> None:
    """Private pool artifacts must never land inside the public repository."""
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return
    raise SystemExit(
        "refusing to write private pool artifacts inside the repository: "
        f"{resolved}"
    )


def _write_pool(
    payload: dict[str, object],
    *,
    output: Path,
    checksum_path: Path,
    force: bool,
) -> str:
    """Write the frozen pool JSON plus its SHA-256 checksum file."""
    if output.resolve() == checksum_path.resolve():
        raise SystemExit("--output and --checksum must be different paths")
    _ensure_outside_repo(output)
    _ensure_outside_repo(checksum_path)
    if not force:
        for path in (output, checksum_path):
            if path.exists():
                raise SystemExit(
                    f"refusing to overwrite existing pool file: {path} "
                    "(pass --force to replace it)"
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    digest = _sha256(output)
    checksum_path.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reserve and freeze the final held-out MUCS Hinglish candidate "
            "evaluation pool (metadata only; transcripts are never read)."
        )
    )
    parser.add_argument(
        "--inventory", type=Path, required=True,
        help="Private MUCS training metadata inventory CSV.",
    )
    parser.add_argument(
        "--speaker-split", type=Path, required=True,
        help="Private speaker_split.json from the frozen v1 cohort.",
    )
    parser.add_argument(
        "--inventory-sha256", default=None,
        help="Expected SHA-256 of the inventory file; verified when given.",
    )
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument(
        "--expected-total-speakers", type=int, default=EXPECTED_TOTAL_SPEAKERS
    )
    parser.add_argument(
        "--expected-training-speakers", type=int, default=EXPECTED_TRAINING_SPEAKERS
    )
    parser.add_argument(
        "--expected-validation-speakers", type=int, default=EXPECTED_VALIDATION_SPEAKERS
    )
    parser.add_argument(
        "--expected-selected-speakers", type=int, default=EXPECTED_SELECTED_SPEAKERS
    )
    parser.add_argument(
        "--expected-remaining-speakers", type=int, default=EXPECTED_REMAINING_SPEAKERS
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Print the pool composition summary without writing anything.",
    )
    mode.add_argument(
        "--freeze", action="store_true",
        help="Write the pool JSON and checksum file (outside the repository).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Pool JSON path; required with --freeze, outside the repository.",
    )
    parser.add_argument(
        "--checksum", type=Path, default=None,
        help="Checksum path; defaults to the output path with .sha256 suffix.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Allow replacing existing pool files. The reserve freeze must not use this.",
    )
    args = parser.parse_args(argv)

    if args.freeze and args.output is None:
        parser.error("--output is required with --freeze")
    if args.dry_run and args.force:
        parser.error("--force is only meaningful with --freeze")

    inventory_sha256 = _sha256(args.inventory)
    if args.inventory_sha256 is not None and inventory_sha256 != args.inventory_sha256:
        raise SystemExit(
            "inventory checksum mismatch: "
            f"expected {args.inventory_sha256}, actual {inventory_sha256}"
        )

    rows = _load_inventory(args.inventory)
    split = _load_speaker_split(args.speaker_split)
    state = _compute_state(rows, split)
    _validate_state(
        state,
        expected_rows=args.expected_rows,
        expected_total_speakers=args.expected_total_speakers,
        expected_training_speakers=args.expected_training_speakers,
        expected_validation_speakers=args.expected_validation_speakers,
        expected_selected_speakers=args.expected_selected_speakers,
        expected_remaining_speakers=args.expected_remaining_speakers,
    )

    payload = _build_pool_payload(
        state,
        rows,
        inventory_path=args.inventory,
        inventory_sha256=inventory_sha256,
        speaker_split_path=args.speaker_split,
        speaker_split_sha256=_sha256(args.speaker_split),
    )

    if args.dry_run:
        _print_summary(payload, mode="dry-run (no files written)", output=args.output)
        return 0

    checksum_path = args.checksum
    if checksum_path is None:
        checksum_path = args.output.with_suffix(".sha256")
    digest = _write_pool(
        payload,
        output=args.output,
        checksum_path=checksum_path,
        force=args.force,
    )
    _print_summary(payload, mode="freeze (pool written)", output=None)
    print(f"Pool file: {args.output}")
    print(f"Checksum file: {checksum_path}")
    print(f"Pool sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

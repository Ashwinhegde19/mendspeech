#!/usr/bin/env python3
"""Select Hinglish training candidates for a bounded adaptation experiment.

The CLI wraps :mod:`src.hinglish.selection` and writes only safe manifests:
clip IDs, speaker IDs, arm, score, rank, composition metadata, duration, and
bucket labels. Reference transcripts and predictions are read when scoring
failure evidence but are never written to any output file.

Prediction handling
-------------------
``--predictions`` is a CSV with ``clip_id`` and ``prediction`` columns and an
optional ``status`` column. Structural validation mirrors
``scripts/evaluate_hinglish_predictions.py``: duplicate clip IDs are rejected,
clip IDs absent from the inventory are rejected, and an optional
``--predictions-manifest`` (using the same reference-column aliases) is
validated as an exact one-to-one match. Without a predictions manifest,
references come from the private inventory; because a candidate clip without a
prediction is a legitimate state for the intelligent arm, partial coverage is
allowed by default and reported. Pass ``--require-full-predictions`` together
with ``--candidate-pool`` to enforce one-to-one coverage of the inference
batch instead.

Speaker safety
--------------
Selection is restricted to the training speakers in ``--speaker-split``.
Validation speakers are always forbidden, diagnostic speakers are forbidden by
default (``--allow-diagnostic-speakers`` overrides), and additional forbidden
speakers (for example official-test speakers when known) can be supplied with
``--forbid-speakers`` or ``--forbid-speakers-file``. An overlap between the
training pool and the validation pool is refused outright.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hinglish.data import invalid_mucs_fields
from src.hinglish.diagnostics import analyze_repetition, likely_repetitive_hallucination
from src.hinglish.evaluation import HinglishPrediction, score_prediction
from src.hinglish.normalization import normalize_hinglish_text
from src.hinglish.selection import (
    ARM_INTELLIGENT,
    ARM_RANDOM,
    SELECTION_VERSION,
    AsrFailureFeatures,
    CandidateRecord,
    SelectionConfig,
    build_paired_selection,
    build_selection_report,
    select_intelligent_arm,
    select_random_arm,
    prepare_candidate_pool,
    write_selection_manifest,
    write_selection_report,
)

INVENTORY_REQUIRED = frozenset(
    {
        "segment_id",
        "speaker_id",
        "transcript",
        "ratio_english_words",
        "ratio_hindi_words",
        "count_english_words",
        "count_dev_words",
    }
)
PREDICTION_COLUMNS = frozenset({"clip_id", "prediction"})
REFERENCE_ALIASES = ("transcript", "reference_transcript")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inventory_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("inventory is empty")
    missing = INVENTORY_REQUIRED.difference(rows[0])
    if missing:
        raise SystemExit(f"inventory missing columns: {sorted(missing)}")
    return rows


def _load_speaker_split(path: Path) -> tuple[set[str], set[str], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    training = {str(value) for value in payload.get("training_speakers", [])}
    validation = {str(value) for value in payload.get("validation_speakers", [])}
    diagnostics = {
        str(value)
        for value in payload.get("diagnostic_speakers_reserved_for_training", [])
    }
    if not training:
        raise SystemExit("speaker split contains no training speakers")
    if training & validation:
        raise SystemExit(
            "speaker split overlaps training and validation pools; refusing to "
            "select candidates from a corrupted split"
        )
    return training, validation, diagnostics


def _load_clip_id_list(path: Path) -> set[str]:
    """Load clip IDs from a CSV (``clip_id`` column), JSON list, or text file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise SystemExit("candidate pool JSON must be a list of clip IDs")
        return {str(value).strip() for value in payload if str(value).strip()}
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if "clip_id" in first_line.split(","):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return {
                str(row["clip_id"]).strip()
                for row in reader
                if str(row.get("clip_id", "")).strip()
            }
    return {line.strip() for line in text.splitlines() if line.strip()}


def _load_duration_manifest(path: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"clip_id", "duration_sec"}.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"duration manifest missing columns: {sorted(missing)}")
        for row in reader:
            clip_id = str(row["clip_id"]).strip()
            try:
                durations[clip_id] = float(row["duration_sec"])
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"duration manifest has a non-numeric duration for {clip_id}"
                ) from exc
    return durations


def _load_predictions(path: Path) -> list[dict[str, str]]:
    """Load and structurally validate a predictions CSV."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = PREDICTION_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"predictions CSV missing columns: {sorted(missing)}")
        for row in reader:
            clip_id = str(row["clip_id"]).strip()
            if clip_id in seen:
                duplicates.append(clip_id)
                continue
            seen.add(clip_id)
            rows.append(row)
    if duplicates:
        raise SystemExit(
            "duplicate clip IDs in predictions: " + ", ".join(sorted(set(duplicates))[:10])
        )
    return rows


def _load_prediction_manifest(path: Path) -> dict[str, str]:
    """Load a reference manifest for exact one-to-one prediction validation."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "clip_id" not in fieldnames:
            raise SystemExit("predictions manifest must contain a clip_id column")
        reference_column = next(
            (name for name in REFERENCE_ALIASES if name in fieldnames), None
        )
        if reference_column is None:
            raise SystemExit(
                "predictions manifest must contain one of: "
                + ", ".join(REFERENCE_ALIASES)
            )
        references: dict[str, str] = {}
        for row in reader:
            clip_id = str(row["clip_id"]).strip()
            if clip_id in references:
                raise SystemExit(f"duplicate clip ID in predictions manifest: {clip_id}")
            reference = str(row[reference_column]).strip()
            if not reference:
                raise SystemExit(f"empty reference transcript for clip: {clip_id}")
            references[clip_id] = reference
    return references


def _validate_one_to_one(prediction_ids: set[str], manifest_ids: set[str]) -> None:
    unknown = prediction_ids - manifest_ids
    if unknown:
        raise SystemExit(
            "unknown prediction clip IDs: " + ", ".join(sorted(unknown)[:10])
        )
    unmatched = manifest_ids - prediction_ids
    if unmatched:
        raise SystemExit(
            "manifest clip IDs without predictions: " + ", ".join(sorted(unmatched)[:10])
        )


def _asr_features(
    prediction: str,
    reference: str,
    ratio_english_words: float,
) -> Optional[AsrFailureFeatures]:
    """Score one prediction against its reference and return failure features.

    Text is consumed and discarded here; only counts and a hallucination
    heuristic flag leave this function.
    """
    if not reference.strip():
        return None
    scored = score_prediction(
        HinglishPrediction(
            clip_id="scoring",
            speaker_id="scoring",
            reference=reference,
            prediction=prediction,
            ratio_english_words=ratio_english_words,
        )
    )
    repetition = analyze_repetition(normalize_hinglish_text(prediction).split())
    hallucination = likely_repetitive_hallucination(
        reference_words=int(scored["reference_words"]),
        prediction_words=int(scored["hypothesis_words"]),
        insertion_count=int(scored["insertions"]),
        max_consecutive_repeat=repetition.max_consecutive_repeat,
        longest_repeated_phrase=repetition.longest_repeated_phrase,
    )
    return AsrFailureFeatures(
        reference_words=int(scored["reference_words"]),
        substitutions=int(scored["substitutions"]),
        deletions=int(scored["deletions"]),
        insertions=int(scored["insertions"]),
        wer=float(scored["wer"]),
        hallucination_flag=hallucination,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically select safe Hinglish training candidates for a "
            "bounded adaptation experiment (never writes transcripts)"
        )
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--speaker-split", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=[ARM_RANDOM, ARM_INTELLIGENT, "both"],
        default=ARM_INTELLIGENT,
        help=(
            "random control, intelligent ranking, or both with matched "
            "duration budgets"
        ),
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-speaker", type=int, default=2)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--preferred-duration", type=float, default=8.0)
    parser.add_argument("--min-total-words", type=int, default=2)
    parser.add_argument("--duration-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--require-duration",
        action="store_true",
        help="Reject candidates without a known duration instead of marking "
        "the duration component unavailable.",
    )
    parser.add_argument(
        "--duration-manifest",
        type=Path,
        default=None,
        help="CSV clip_id,duration_sec with measured clip durations.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=None,
        help=(
            "Restrict candidates to these clip IDs (CSV clip_id column, JSON "
            "list, or one-per-line text). Use this with the exact clip set "
            "sent to the bounded inference job so both arms draw from one pool."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional CSV clip_id,prediction[,status]. Never written out.",
    )
    parser.add_argument(
        "--predictions-manifest",
        type=Path,
        default=None,
        help=(
            "Optional reference manifest for exact one-to-one prediction "
            "validation, mirroring scripts/evaluate_hinglish_predictions.py."
        ),
    )
    parser.add_argument(
        "--require-full-predictions",
        action="store_true",
        help=(
            "Require every clip in --candidate-pool to have a prediction row "
            "(failed rows count as covered)."
        ),
    )
    parser.add_argument(
        "--allow-diagnostic-speakers",
        action="store_true",
        help=(
            "Allow diagnostic speakers into the candidate pool. Default is to "
            "forbid them because their earlier clips were analyzed in smoke "
            "tests."
        ),
    )
    parser.add_argument(
        "--forbid-speakers",
        default="",
        help="Comma-separated speaker IDs to reject (for example official-test "
        "speakers when known).",
    )
    parser.add_argument(
        "--forbid-speakers-file",
        type=Path,
        default=None,
        help="JSON list of additional forbidden speaker IDs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline and print the report without writing files.",
    )
    args = parser.parse_args()

    inventory_path = args.inventory.resolve()
    if args.inventory_sha256 is not None:
        actual = _sha256(inventory_path)
        if actual != args.inventory_sha256:
            raise SystemExit(
                f"inventory checksum mismatch: expected {args.inventory_sha256}, "
                f"actual {actual}"
            )
    rows = _load_inventory_rows(inventory_path)
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise SystemExit(
            f"inventory row mismatch: expected {args.expected_rows}, "
            f"actual {len(rows)}"
        )

    training_speakers, validation_speakers, diagnostic_speakers = _load_speaker_split(
        args.speaker_split
    )
    forbidden_speakers = set(validation_speakers)
    if not args.allow_diagnostic_speakers:
        forbidden_speakers |= diagnostic_speakers
    extra_forbidden = {
        value.strip() for value in args.forbid_speakers.split(",") if value.strip()
    }
    if args.forbid_speakers_file is not None:
        payload = json.loads(args.forbid_speakers_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit("--forbid-speakers-file must contain a JSON list")
        extra_forbidden |= {str(value).strip() for value in payload if str(value).strip()}
    forbidden_speakers |= extra_forbidden

    durations: dict[str, float] = {}
    if args.duration_manifest is not None:
        durations = _load_duration_manifest(args.duration_manifest)

    candidate_pool: Optional[set[str]] = None
    if args.candidate_pool is not None:
        candidate_pool = _load_clip_id_list(args.candidate_pool)
        known_ids = {row["segment_id"] for row in rows}
        unknown_pool_ids = candidate_pool - known_ids
        if unknown_pool_ids:
            raise SystemExit(
                "candidate pool contains clip IDs absent from the inventory: "
                + ", ".join(sorted(unknown_pool_ids)[:10])
            )

    predictions: dict[str, dict[str, str]] = {}
    prediction_manifest_references: dict[str, str] = {}
    if args.predictions is not None:
        prediction_rows = _load_predictions(args.predictions)
        known_ids = {row["segment_id"] for row in rows}
        unknown = {
            str(row["clip_id"]).strip()
            for row in prediction_rows
            if str(row["clip_id"]).strip() not in known_ids
        }
        if unknown:
            raise SystemExit(
                "unknown prediction clip IDs: " + ", ".join(sorted(unknown)[:10])
            )
        if args.predictions_manifest is not None:
            prediction_manifest_references = _load_prediction_manifest(
                args.predictions_manifest
            )
            _validate_one_to_one(
                {str(row["clip_id"]).strip() for row in prediction_rows},
                set(prediction_manifest_references),
            )
        if args.require_full_predictions:
            if candidate_pool is None:
                raise SystemExit(
                    "--require-full-predictions needs --candidate-pool so the "
                    "expected coverage is well defined"
                )
            covered = {str(row["clip_id"]).strip() for row in prediction_rows}
            missing = candidate_pool - covered
            if missing:
                raise SystemExit(
                    "candidate pool clip IDs without predictions: "
                    + ", ".join(sorted(missing)[:10])
                )
        for row in prediction_rows:
            clip_id = str(row["clip_id"]).strip()
            predictions[clip_id] = row
    elif args.predictions_manifest is not None:
        raise SystemExit("--predictions-manifest requires --predictions")

    candidates: list[CandidateRecord] = []
    for row in rows:
        clip_id = row["segment_id"]
        if candidate_pool is not None and clip_id not in candidate_pool:
            continue
        invalid_fields = invalid_mucs_fields(row)
        try:
            ratio_english = float(row["ratio_english_words"])
            ratio_hindi = float(row["ratio_hindi_words"])
        except (TypeError, ValueError):
            ratio_english = float("nan")
            ratio_hindi = float("nan")

        asr: Optional[AsrFailureFeatures] = None
        prediction_row = predictions.get(clip_id) if predictions else None
        if prediction_row is not None:
            status = str(prediction_row.get("status", "ok") or "ok").strip().lower()
            if status not in {"failed", "fail", "error"}:
                reference = prediction_manifest_references.get(clip_id)
                if reference is None:
                    reference = str(row["transcript"])
                asr = _asr_features(
                    str(prediction_row.get("prediction", "")),
                    reference,
                    ratio_english if ratio_english == ratio_english else 0.5,
                )

        try:
            count_english = int(row.get("count_english_words", 0) or 0)
            count_hindi = int(row.get("count_dev_words", 0) or 0)
        except (TypeError, ValueError):
            count_english = 0
            count_hindi = 0

        candidates.append(
            CandidateRecord(
                clip_id=clip_id,
                speaker_id=str(row["speaker_id"]),
                reference_available=bool(str(row["transcript"]).strip()),
                ratio_english_words=ratio_english,
                ratio_hindi_words=ratio_hindi,
                count_english_words=count_english,
                count_hindi_words=count_hindi,
                segment_part=str(row.get("segment_part", "") or ""),
                shard=str(row.get("shard", "") or ""),
                duration_sec=durations.get(clip_id),
                metadata_valid=not invalid_fields,
                invalid_fields=invalid_fields,
                asr=asr,
            )
        )

    config = SelectionConfig(
        target_count=args.count,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        max_per_speaker=args.max_per_speaker,
        duration_tolerance=args.duration_tolerance,
        seed=args.seed,
        preferred_duration_sec=args.preferred_duration,
        min_total_words=args.min_total_words,
        require_duration=args.require_duration,
    )

    selection_kwargs = dict(
        training_speakers=training_speakers,
        forbidden_speakers=forbidden_speakers,
        config=config,
    )

    if args.arm == "both":
        try:
            paired = build_paired_selection(candidates, **selection_kwargs)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        pool = paired.pool
        arms = {
            ARM_INTELLIGENT: paired.intelligent,
            ARM_RANDOM: paired.random,
        }
    else:
        pool = prepare_candidate_pool(candidates, **selection_kwargs)
        selector = (
            select_random_arm if args.arm == ARM_RANDOM else select_intelligent_arm
        )
        try:
            arms = {args.arm: selector(pool.eligible, config)}
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    report = build_selection_report(pool, arms, config)
    report["selection_version"] = SELECTION_VERSION
    report["inputs"] = {
        "inventory_rows": len(rows),
        "inventory_sha256": _sha256(inventory_path),
        "candidate_pool_size": len(candidate_pool) if candidate_pool else None,
        "predictions_provided": bool(predictions),
        "prediction_coverage_fraction": (
            round(len(predictions) / len(candidate_pool), 4)
            if predictions and candidate_pool
            else None
        ),
        "predicted_clips_without_candidates": (
            sum(1 for clip_id in predictions if candidate_pool and clip_id not in candidate_pool)
            if predictions and candidate_pool
            else 0
        ),
        "training_speakers": len(training_speakers),
        "forbidden_speakers": len(forbidden_speakers),
        "diagnostic_speakers_allowed": args.allow_diagnostic_speakers,
        "duration_manifest_clips": len(durations),
    }
    report["transcript_written"] = False
    report["selection_guardrails"] = {
        "training_pool_only": True,
        "validation_speakers_forbidden": True,
        "diagnostic_speakers_forbidden": not args.allow_diagnostic_speakers,
        "official_test_speakers_forbidden_when_supplied": True,
        "duplicate_clip_ids_rejected": True,
        "transcript_free_outputs": True,
        "deterministic_seed": args.seed,
    }

    summary = {
        "pool": report["pool"],
        "arms": {
            name: {
                key: value
                for key, value in arm_summary.items()
                if key in {"size", "distinct_speakers", "total_duration_sec", "budget_match"}
            }
            for name, arm_summary in report["arms"].items()
        },
        "inputs": report["inputs"],
    }
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        print("dry run: no files written; pass without --dry-run to write manifests")
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for arm_name, arm in sorted(arms.items()):
        manifest_path = output_dir / f"selection_manifest_{arm_name}.csv"
        write_selection_manifest(manifest_path, arm)
        written.append(manifest_path)
    report_path = output_dir / "selection_report.json"
    write_selection_report(report_path, report)
    written.append(report_path)
    for path in written:
        checksum = _sha256(path)
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{checksum}  {path.name}\n", encoding="utf-8"
        )
    print("written files:")
    for path in written:
        print(f"  {path}")
    print("transcripts written: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

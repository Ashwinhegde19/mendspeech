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
from typing import Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hinglish.data import invalid_mucs_fields
from src.hinglish.diagnostics import analyze_repetition, likely_repetitive_hallucination
from src.hinglish.evaluation import HinglishPrediction, score_prediction
from src.hinglish.normalization import normalize_hinglish_text
from src.hinglish.selection import (
    ACQUISITION_LABEL_FREE,
    ACQUISITION_MODES,
    ACQUISITION_ORACLE,
    ARM_INTELLIGENT,
    ARM_RANDOM,
    SELECTION_VERSION,
    AsrFailureFeatures,
    AsrOutputFeatures,
    CandidateRecord,
    SelectionConfig,
    assert_finalized_arm_speakers,
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


def _prediction_is_failed(row: "Mapping[str, str]") -> bool:
    """Return True when a prediction row records failed inference.

    A failed row is present but does not describe a usable decode: it must
    never silently count as ASR scoring coverage.
    """
    status = str(row.get("status", "") or "").strip().lower()
    return status in {"failed", "fail", "error"}


POOL_MANIFEST_REQUIRED_COLUMNS = frozenset(
    {"clip_id", "speaker_id", "audio_duration_sec", "shard"}
)
# Transcript-derived columns must never appear in a label-free pool
# manifest; their presence means reference-derived data leaked into the
# acquisition input.
POOL_MANIFEST_FORBIDDEN_COLUMNS = frozenset(
    {
        "transcript",
        "reference_transcript",
        "reference",
        "prediction",
        "hypothesis",
        "ratio_english_words",
        "ratio_hindi_words",
        "count_english_words",
        "count_hindi_words",
        "count_dev_words",
        "audio_duration_corrected",
    }
)
SCORING_STATUS_VALUES = frozenset({"ok", "failed"})


def _load_inventory_clip_ids(path: Path) -> tuple[set[str], int]:
    """Load only the inventory's clip IDs (join check) plus the row count.

    Label-free acquisition must not read transcript-derived inventory
    values; this loader keeps ``segment_id`` values only. The returned row
    count is parsed in the same pass.
    """
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "segment_id" not in fieldnames:
            raise SystemExit("inventory is missing a segment_id column")
        clip_ids = set()
        row_count = 0
        for row in reader:
            row_count += 1
            value = str(row.get("segment_id", "")).strip()
            if value:
                clip_ids.add(value)
    if not clip_ids:
        raise SystemExit("inventory contains no clip IDs")
    return clip_ids, row_count


def _load_pool_manifest(path: Path) -> list[dict[str, str]]:
    """Load and validate the frozen label-free pool manifest."""
    if not path.is_file():
        raise SystemExit(f"pool manifest not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = POOL_MANIFEST_REQUIRED_COLUMNS.difference(fieldnames)
        if missing:
            raise SystemExit(
                "pool manifest missing columns: " + ", ".join(sorted(missing))
            )
        forbidden = POOL_MANIFEST_FORBIDDEN_COLUMNS.intersection(fieldnames)
        if forbidden:
            raise SystemExit(
                "label-free pool manifest must not contain transcript-derived "
                "columns: " + ", ".join(sorted(forbidden))
            )
        rows = list(reader)
    if not rows:
        raise SystemExit("pool manifest is empty")

    seen_ids: set[str] = set()
    normalized: list[dict[str, str]] = []
    for row in rows:
        clip_id = str(row["clip_id"]).strip()
        if not clip_id:
            raise SystemExit("pool manifest contains an empty clip_id")
        if clip_id in seen_ids:
            raise SystemExit(f"duplicate clip IDs in pool manifest: {clip_id}")
        seen_ids.add(clip_id)
        validated = dict(row)
        try:
            duration = float(row["audio_duration_sec"])
        except (TypeError, ValueError):
            raise SystemExit(
                f"pool manifest clip {clip_id}: invalid audio_duration_sec"
            ) from None
        if not duration > 0.0:
            raise SystemExit(
                f"pool manifest clip {clip_id}: audio_duration_sec must be positive"
            )
        validated["audio_duration_sec"] = repr(duration)
        validated["clip_id"] = clip_id
        normalized.append(validated)
    return normalized


def _opt_float(value: object) -> Optional[float]:
    """Coerce an optional JSON number to float or None."""
    if value is None:
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN guard
        return None
    return parsed


def _opt_int(value: object) -> Optional[int]:
    """Coerce an optional JSON number to int or None."""
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _opt_bool(value: object) -> Optional[bool]:
    """Coerce an optional boolean value or None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _load_scoring_records(path: Path) -> dict[str, AsrOutputFeatures]:
    """Load model-derived scoring records from the J2 results JSON."""
    if not path.is_file():
        raise SystemExit(f"scoring results not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise SystemExit("scoring results JSON must contain a non-empty records list")
    features: dict[str, AsrOutputFeatures] = {}
    for record in records:
        clip_id = str(record.get("clip_id", "")).strip()
        if not clip_id:
            raise SystemExit("scoring record without clip_id")
        if clip_id in features:
            raise SystemExit(f"duplicate scoring records for clip {clip_id}")
        status = str(record.get("status", "") or "").strip().lower()
        if status not in SCORING_STATUS_VALUES:
            status = "failed"
        features[clip_id] = AsrOutputFeatures(
            status=status,
            audio_duration_sec=_opt_float(record.get("audio_duration_sec")),
            inference_runtime_sec=_opt_float(record.get("inference_runtime_sec")),
            generated_token_count=_opt_int(record.get("generated_token_count")),
            eos_emitted_effective=_opt_bool(record.get("eos_emitted_effective")),
            hit_length_limit_effective=_opt_bool(
                record.get("hit_length_limit_effective")
            ),
            max_consecutive_repeated_tokens=_opt_int(
                record.get("max_consecutive_repeated_tokens")
            ),
            repeated_token_fraction=_opt_float(record.get("repeated_token_fraction")),
            repetition_flag=_opt_bool(record.get("repetition_flag")),
            mean_logprob=_opt_float(record.get("mean_logprob")),
            min_logprob=_opt_float(record.get("min_logprob")),
            entropy_mean=_opt_float(record.get("entropy_mean")),
        )
    return features


def _build_label_free_candidates(
    pool_rows: list[dict[str, str]],
    scoring_features: dict[str, AsrOutputFeatures],
    candidate_pool: Optional[set[str]] = None,
) -> list[CandidateRecord]:
    """Build label-free CandidateRecords from the frozen pool manifest.

    Transcript-derived fields stay ``None``/0 by construction: the pool
    manifest carries only clip/speaker/shard/duration/quality metadata and
    the scoring records carry only model-derived numeric features.
    """
    candidates: list[CandidateRecord] = []
    for row in pool_rows:
        clip_id = row["clip_id"]
        if candidate_pool is not None and clip_id not in candidate_pool:
            continue
        asr_output = scoring_features.get(clip_id)
        scored_duration = (
            asr_output.audio_duration_sec
            if asr_output is not None and asr_output.status == "ok"
            else None
        )
        manifest_duration = _opt_float(row["audio_duration_sec"])
        candidates.append(
            CandidateRecord(
                clip_id=clip_id,
                speaker_id=str(row["speaker_id"]).strip(),
                reference_available=True,
                ratio_english_words=None,
                ratio_hindi_words=None,
                count_english_words=0,
                count_hindi_words=0,
                segment_part=str(row.get("segment_part", "") or ""),
                shard=str(row.get("shard", "") or ""),
                duration_sec=scored_duration
                if scored_duration is not None
                else manifest_duration,
                metadata_valid=True,
                invalid_fields=(),
                asr=None,
                asr_output=asr_output,
                audio_peak_dbfs=_opt_float(row.get("peak_dbfs")),
                audio_clipping_fraction=_opt_float(row.get("clipping_fraction")),
                audio_mean_rms_dbfs=_opt_float(row.get("mean_rms_dbfs")),
            )
        )
    if not candidates:
        raise SystemExit("pool manifest yielded no candidates")
    return candidates


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
    parser.add_argument("--max-per-speaker", type=int, default=1)
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
            "Require every clip in --candidate-pool to have a usable "
            "(status=ok, nonempty) prediction; missing rows and failed rows "
            "both reject."
        ),
    )
    parser.add_argument(
        "--acquisition-mode",
        choices=list(ACQUISITION_MODES),
        default=ACQUISITION_ORACLE,
        help=(
            "oracle: rank with reference-derived evidence (supervised). "
            "label_free: rank from acquisition-time signals only; requires "
            "--pool-manifest and --scoring, rejects --predictions and "
            "transcript-derived gates."
        ),
    )
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        default=None,
        help=(
            "Label-free pool manifest CSV (clip_id, speaker_id, "
            "audio_duration_sec, shard[, audio quality columns]) prepared "
            "and frozen before the scoring run. Required in label_free "
            "mode."
        ),
    )
    parser.add_argument(
        "--scoring",
        type=Path,
        default=None,
        help=(
            "scoring_results.json from the label-free scoring run; provides "
            "AsrOutputFeatures (model-derived tokens/EOS/cap/repetition/"
            "confidence). Required in label_free mode."
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

    if args.require_full_predictions:
        if args.predictions is None:
            raise SystemExit(
                "--require-full-predictions requires --predictions: no "
                "predictions file was provided, so full coverage cannot be "
                "verified"
            )
        if not args.predictions.is_file():
            raise SystemExit(
                f"--require-full-predictions: predictions file not found: "
                f"{args.predictions}"
            )
    label_free = args.acquisition_mode == ACQUISITION_LABEL_FREE
    if args.acquisition_mode == ACQUISITION_LABEL_FREE:
        if args.predictions is not None:
            raise SystemExit(
                "label_free acquisition prohibits ground-truth WER/S/D/I "
                "features: do not pass --predictions"
            )
        if args.predictions_manifest is not None:
            raise SystemExit(
                "label_free acquisition never reads reference transcripts: "
                "do not pass --predictions-manifest"
            )
        if args.duration_manifest is not None:
            raise SystemExit(
                "label_free durations must come from the frozen --pool-manifest; "
                "do not pass --duration-manifest"
            )
        if not args.require_duration:
            raise SystemExit(
                "label_free acquisition ranks on duration only: pass "
                "--require-duration"
            )
        if args.min_total_words != 0:
            raise SystemExit(
                "label_free acquisition prohibits the transcript-derived "
                "word-count gate: pass --min-total-words 0"
            )
        if args.pool_manifest is None:
            raise SystemExit(
                "label_free acquisition requires --pool-manifest (the frozen "
                "300-clip candidate pool)"
            )
        if args.scoring is None:
            raise SystemExit(
                "label_free acquisition requires --scoring (the completed "
                "candidate-scoring run results)"
            )
        if not args.pool_manifest.is_file():
            raise SystemExit(f"pool manifest not found: {args.pool_manifest}")
        if not args.scoring.is_file():
            raise SystemExit(f"scoring results not found: {args.scoring}")

    inventory_path = args.inventory.resolve()
    if args.inventory_sha256 is not None:
        actual = _sha256(inventory_path)
        if actual != args.inventory_sha256:
            raise SystemExit(
                f"inventory checksum mismatch: expected {args.inventory_sha256}, "
                f"actual {actual}"
            )
    rows: Optional[list[dict[str, str]]] = None
    inventory_row_count: Optional[int] = None
    pool_manifest_rows: list[dict[str, str]] = []
    scoring_features: dict[str, AsrOutputFeatures] = {}
    if label_free:
        pool_manifest_rows = _load_pool_manifest(args.pool_manifest)
        scoring_features = _load_scoring_records(args.scoring)
        inventory_clip_ids, inventory_row_count = _load_inventory_clip_ids(
            inventory_path
        )
        # Join check only: label-free mode never reads inventory values.
        pool_ids = {row["clip_id"] for row in pool_manifest_rows}
        unknown_pool_ids = pool_ids - inventory_clip_ids
        if unknown_pool_ids:
            raise SystemExit(
                "pool manifest clip IDs absent from the training inventory "
                "(reference join check failed; references are loaded only for "
                "the later training stage): "
                + ", ".join(sorted(unknown_pool_ids)[:10])
            )
        if len({row["speaker_id"] for row in pool_manifest_rows}) != len(
            pool_manifest_rows
        ):
            raise SystemExit(
                "pool manifest must list one clip per distinct speaker"
            )
        if len(pool_manifest_rows) < args.count:
            raise SystemExit(
                f"pool manifest has {len(pool_manifest_rows)} clips; fewer than "
                f"the requested arm size {args.count}"
            )
    else:
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
        if label_free:
            known_ids = {row["clip_id"] for row in pool_manifest_rows}
            universe_note = "the pool manifest"
        else:
            known_ids = {row["segment_id"] for row in rows}
            universe_note = "the inventory"
        unknown_pool_ids = candidate_pool - known_ids
        if unknown_pool_ids:
            raise SystemExit(
                f"candidate pool contains clip IDs absent from {universe_note}: "
                + ", ".join(sorted(unknown_pool_ids)[:10])
            )
    if label_free:
        pool_ids = {row["clip_id"] for row in pool_manifest_rows}
        selection_universe = pool_ids & candidate_pool if candidate_pool else pool_ids
        missing_scoring = sorted(selection_universe - set(scoring_features))
        if missing_scoring:
            raise SystemExit(
                f"scoring results missing {len(missing_scoring)} pool clips "
                "required for label-free selection (missing records are never "
                "silently skipped): "
                + ", ".join(missing_scoring[:10])
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
            present = {
                str(row["clip_id"]).strip()
                for row in prediction_rows
            }
            missing = candidate_pool - present
            failed_in_pool = {
                str(row["clip_id"]).strip()
                for row in prediction_rows
                if _prediction_is_failed(row) and str(row["clip_id"]).strip() in candidate_pool
            }
            if failed_in_pool:
                raise SystemExit(
                    "candidate pool clip IDs with failed inference rows: "
                    + ", ".join(sorted(failed_in_pool)[:10])
                )
            if missing:
                raise SystemExit(
                    "candidate pool clip IDs without usable predictions: "
                    + ", ".join(sorted(missing)[:10])
                )
        for row in prediction_rows:
            clip_id = str(row["clip_id"]).strip()
            predictions[clip_id] = row
    elif args.predictions_manifest is not None:
        raise SystemExit("--predictions-manifest requires --predictions")

    if label_free:
        candidates = _build_label_free_candidates(
            pool_manifest_rows, scoring_features, candidate_pool
        )
    else:
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

    try:
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
            acquisition_mode=args.acquisition_mode,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid selection config: {exc}") from exc

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

    try:
        for arm in arms.values():
            assert_finalized_arm_speakers(arm, config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    report = build_selection_report(pool, arms, config)
    report["selection_version"] = SELECTION_VERSION
    report["inputs"] = {
        "inventory_rows": len(rows) if rows is not None else inventory_row_count,
        "inventory_sha256": _sha256(inventory_path),
        "candidate_pool_size": len(candidate_pool) if candidate_pool else None,
        "predictions_provided": bool(predictions),
        "prediction_usable_count": (
            sum(1 for row in predictions.values() if not _prediction_is_failed(row))
            if predictions
            else 0
        ),
        "prediction_failed_clip_count": (
            sum(1 for row in predictions.values() if _prediction_is_failed(row))
            if predictions
            else 0
        ),
        "prediction_failed_clip_ids": sorted(
            clip_id
            for clip_id, row in predictions.items()
            if _prediction_is_failed(row)
        )[:10],
        "prediction_coverage_fraction": (
            round(
                sum(1 for row in predictions.values() if not _prediction_is_failed(row))
                / len(candidate_pool),
                4,
            )
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
    if label_free:
        report["inputs"]["inventory_rows"] = inventory_row_count
        assert args.pool_manifest is not None and args.scoring is not None
        failed_ids = sorted(
            clip_id
            for clip_id, features in scoring_features.items()
            if features.status != "ok"
        )
        report["inputs"]["pool_manifest_rows"] = len(pool_manifest_rows)
        report["inputs"]["pool_manifest_sha256"] = _sha256(args.pool_manifest)
        report["inputs"]["scoring_records"] = len(scoring_features)
        report["inputs"]["scoring_sha256"] = _sha256(args.scoring)
        report["inputs"]["scoring_failed_clip_count"] = len(failed_ids)
        report["inputs"]["scoring_failed_clip_ids"] = failed_ids[:10]
        report["inputs"]["scoring_ok_clip_count"] = (
            len(scoring_features) - len(failed_ids)
        )
    report["transcript_written"] = False
    if len(arms) == 2 and {"random", "intelligent"}.issubset(set(arms)):
        random_ids = {
            entry.candidate.clip_id for entry in arms["random"].ranked
        }
        intelligent_ids = {
            entry.candidate.clip_id for entry in arms["intelligent"].ranked
        }
        overlap_ids = sorted(random_ids & intelligent_ids)
        report["clip_overlap_between_arms"] = {
            "count": len(overlap_ids),
            "clip_ids": overlap_ids[:20],
            "note": (
                "arms draw independently from the identical 300-clip pool; "
                "overlap is allowed and reported, never hidden"
            ),
        }
    report["selection_guardrails"] = {
        "training_pool_only": True,
        "validation_speakers_forbidden": True,
        "diagnostic_speakers_forbidden": not args.allow_diagnostic_speakers,
        "official_test_speakers_forbidden_when_supplied": True,
        "duplicate_clip_ids_rejected": True,
        "transcript_free_outputs": True,
        "deterministic_seed": args.seed,
        "acquisition_mode": args.acquisition_mode,
        "label_free_reference_features_prohibited": label_free,
        "failed_predictions_counted_as_unusable_coverage": True,
        "distinct_speakers_asserted_when_cap_is_one": True,
        "label_free_scoring_records_required": label_free,
        "scoring_failed_rejected_not_zeroed": label_free,
        "pool_manifest_is_single_duration_source": label_free,
        "arms_clip_overlap_reported": len(arms) == 2,
    }

    inputs_for_summary = report["inputs"]
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
        "inputs": inputs_for_summary,
        "clip_overlap_between_arms": report.get("clip_overlap_between_arms"),
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

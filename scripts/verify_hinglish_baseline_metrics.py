#!/usr/bin/env python3
"""Verify a committed Hinglish baseline artifact against private run inputs.

This CLI is an offline integrity check for a committed aggregate such as
``results/hinglish/validation_baseline_metrics.json``. It re-scores the
private prediction CSV against the frozen manifest with the repository's own
call chain (``src.hinglish.evaluation`` over ``src.hinglish.normalization``)
and compares every aggregate, per-clip row, composition bin and
hallucination flag with the committed numbers. Optionally it cross-checks
the private raw inference output and the private diagnostics file.

Privacy: the JSON report contains clip IDs, counts and rates only. Reference
transcripts and raw hypotheses are compared in memory and are never written
to the report. The script performs no network access.

Exit codes:
    0: every checked dimension matched.
    1: at least one checked dimension mismatched.
    2: the inputs could not be read or failed validation (for example
       duplicate or unknown clip IDs).

Example:
    python scripts/verify_hinglish_baseline_metrics.py \
        --predictions predictions.csv \
        --manifest frozen_validation_manifest.csv \
        --metrics results/hinglish/validation_baseline_metrics.json \
        --raw-output raw_modal_output.json \
        --diagnostics hallucination_diagnostics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.hinglish.diagnostics import (
    diagnose_predictions,
    summarize_diagnostics,
)
from src.hinglish.evaluation import (
    HinglishPrediction,
    summarize_predictions,
)
from src.hinglish.normalization import HINGLISH_NORMALIZATION_VERSION

VERIFICATION_VERSION = "mendspeech.hinglish_verification.v1"

PREDICTION_COLUMNS = frozenset({"clip_id", "prediction"})
MANIFEST_REQUIRED = frozenset({"clip_id"})
# The frozen validation manifest names the reference column
# `reference_transcript`; smoke-test manifests call it `transcript`. Both
# aliases mirror scripts/evaluate_hinglish_predictions.py.
MANIFEST_REFERENCE_ALIASES = ("transcript", "reference_transcript")

# Committed artifacts round rates to four decimals; a half-ulp of 5e-5 plus a
# small guard is the smallest tolerance that still compares them exactly.
DEFAULT_TOLERANCE = 5.1e-5

AGGREGATE_FIELDS = (
    ("utterances", "utterances", "int"),
    ("reference_words", "reference_words", "int"),
    ("reference_characters", "reference_characters", "int"),
    ("total_edits", "total_edits", "int"),
    ("substitutions", "substitutions", "int"),
    ("deletions", "deletions", "int"),
    ("insertions", "insertions", "int"),
    ("corpus_wer", "wer", "float"),
    ("corpus_cer", "cer", "float"),
    ("mean_utterance_wer", "mean_utterance_wer", "float"),
    ("mean_utterance_cer", "mean_utterance_cer", "float"),
)

PER_CLIP_FIELDS = (
    ("wer", "wer", "float"),
    ("cer", "cer", "float"),
    ("substitutions", "substitutions", "int"),
    ("deletions", "deletions", "int"),
    ("insertions", "insertions", "int"),
    ("reference_words", "reference_words", "int"),
    ("hypothesis_words", "hypothesis_words", "int"),
    ("ratio_bin", "english_ratio_bin", "str"),
    ("ratio_english_words", "ratio_english_words", "float"),
    ("speaker_id", "speaker_id", "str"),
)
# These per-clip fields must exist in the committed row; optional fields are
# compared only when the committed artifact carries them.
PER_CLIP_REQUIRED = frozenset(
    {
        "wer",
        "cer",
        "substitutions",
        "deletions",
        "insertions",
        "reference_words",
        "hypothesis_words",
    }
)
PER_CLIP_OPTIONAL = ("ratio_bin", "ratio_english_words", "speaker_id", "stratification_bucket", "hallucination_flagged")


def _load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV path and reject an empty file."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV is empty: {path}")
    return frame


def _validate_inputs(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the one-to-one prediction/reference mapping.

    Mirrors the guardrails of ``scripts/evaluate_hinglish_predictions.py``:
    required columns, reference alias, duplicate rejection, unknown clip
    rejection, unmatched manifest rejection and empty-reference rejection.

    Args:
        predictions: Prediction rows from the private run archive.
        manifest: Frozen manifest rows with the reference transcripts.

    Returns:
        A tuple ``(predictions, manifest)`` with the reference column
        normalized to ``transcript``.

    Raises:
        ValueError: If any guardrail fails. Messages contain clip IDs only.
    """
    missing_prediction = PREDICTION_COLUMNS.difference(predictions.columns)
    if missing_prediction:
        raise ValueError(
            "predictions CSV must contain: " + ", ".join(sorted(missing_prediction))
        )
    missing_required = MANIFEST_REQUIRED.difference(manifest.columns)
    if missing_required:
        raise ValueError(
            "manifest CSV must contain: " + ", ".join(sorted(missing_required))
        )
    if not set(MANIFEST_REFERENCE_ALIASES).intersection(manifest.columns):
        raise ValueError(
            "manifest CSV must contain one of: "
            + ", ".join(MANIFEST_REFERENCE_ALIASES)
        )
    if "transcript" not in manifest.columns:
        manifest = manifest.rename(columns={"reference_transcript": "transcript"})

    if predictions["clip_id"].duplicated().any():
        duplicates = (
            predictions.loc[predictions["clip_id"].duplicated(), "clip_id"]
            .astype(str)
            .unique()
        )
        raise ValueError(
            "duplicate clip IDs in predictions: " + ", ".join(map(str, duplicates))
        )
    if manifest["clip_id"].duplicated().any():
        duplicates = (
            manifest.loc[manifest["clip_id"].duplicated(), "clip_id"]
            .astype(str)
            .unique()
        )
        raise ValueError(
            "duplicate clip IDs in manifest: " + ", ".join(map(str, duplicates))
        )

    prediction_ids = set(predictions["clip_id"].astype(str))
    manifest_ids = set(manifest["clip_id"].astype(str))
    missing = prediction_ids - manifest_ids
    if missing:
        raise ValueError("unknown prediction clip IDs: " + ", ".join(sorted(missing)))
    unmatched = manifest_ids - prediction_ids
    if unmatched:
        raise ValueError(
            "manifest clip IDs without predictions: " + ", ".join(sorted(unmatched))
        )

    empty_reference = manifest[
        manifest["transcript"].isna()
        | (manifest["transcript"].astype(str).str.strip() == "")
    ]
    if not empty_reference.empty:
        raise ValueError(
            "empty reference transcripts for: "
            + ", ".join(empty_reference["clip_id"].astype(str))
        )
    return predictions, manifest


def _build_records(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[list[HinglishPrediction], list[dict[str, str]]]:
    """Build scored records with the same defaults as the evaluator.

    Args:
        predictions: Validated prediction rows.
        manifest: Validated manifest rows using the ``transcript`` field.

    Returns:
        A tuple of scored ``HinglishPrediction`` records and the list of
        failed predictions (clip ID plus error text only).
    """
    lookup = manifest.set_index("clip_id").to_dict(orient="index")
    records: list[HinglishPrediction] = []
    failed: list[dict[str, str]] = []
    for row in predictions.to_dict(orient="records"):
        clip_id = str(row["clip_id"])
        manifest_row = lookup[clip_id]
        status = str(row.get("status", "ok")).strip().lower()
        if status == "failed":
            failed.append({"clip_id": clip_id, "error": str(row.get("error", ""))})
            continue
        reference = manifest_row.get("transcript")
        if reference is None:
            raise ValueError(f"missing reference transcript for clip {clip_id}")
        records.append(
            HinglishPrediction(
                clip_id=clip_id,
                speaker_id=str(manifest_row.get("speaker_id", "")),
                reference=str(reference).strip(),
                prediction=str(row["prediction"]),
                ratio_english_words=float(manifest_row.get("ratio_english_words", 0.5)),
                language=str(row.get("language", "hi")),
                task=str(row.get("task", "transcribe")),
                model_id=str(row.get("model_id", "openai/whisper-small")),
                model_revision=str(
                    row.get(
                        "model_revision", "973afd24965f72e36ca33b3055d56a652f456b4d"
                    )
                ),
                status="ok",
                error="",
            )
        )
    return records, failed


def _scalar_match(recomputed: Any, committed: Any, kind: str, tolerance: float) -> bool:
    """Compare one recomputed value against one committed value."""
    if kind == "int":
        return int(recomputed) == int(committed)
    if kind == "float":
        return abs(float(recomputed) - float(committed)) <= tolerance
    return str(recomputed) == str(committed)


def _field_entry(
    recomputed: Any,
    committed: Any,
    kind: str,
    tolerance: float,
) -> dict[str, Any]:
    """Build one field comparison entry."""
    matched = _scalar_match(recomputed, committed, kind, tolerance)
    entry: dict[str, Any] = {
        "match": matched,
        "recomputed": recomputed,
        "committed": committed,
    }
    if kind == "float":
        entry["delta"] = round(float(recomputed) - float(committed), 9)
    return entry


def _compare_section(
    recomputed: dict[str, Any],
    committed: dict[str, Any],
    field_map: tuple[tuple[str, str, str], ...],
    tolerance: float,
    *,
    required: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Compare selected fields of one artifact section."""
    fields: dict[str, Any] = {}
    matched = True
    for committed_name, recomputed_name, kind in field_map:
        if committed_name not in committed:
            if committed_name in required:
                fields[committed_name] = {"match": False, "error": "missing in committed"}
                matched = False
            continue
        entry = _field_entry(
            recomputed[recomputed_name], committed[committed_name], kind, tolerance
        )
        fields[committed_name] = entry
        matched = matched and bool(entry["match"])
    return {"match": matched, "fields": fields}


def _bucket_rows(
    scored_rows: list[dict[str, Any]],
    bucket_of: dict[str, str],
    flags: dict[str, bool],
) -> list[dict[str, Any]]:
    """Summarize corpus-weighted and macro rates for each bucket.

    Args:
        scored_rows: Per-clip score rows from ``summarize_predictions``.
        bucket_of: Clip ID to bucket name mapping.
        flags: Clip ID to hallucination-flag mapping.

    Returns:
        One row per bucket with clip counts, error counts, corpus-weighted
        rates (total edits / total reference units) and macro rates (mean of
        per-clip rates).
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[str(bucket_of[str(row["clip_id"])])].append(row)
    table: list[dict[str, Any]] = []
    for bucket in sorted(grouped):
        rows = grouped[bucket]
        reference_words = sum(int(row["reference_words"]) for row in rows)
        reference_characters = sum(int(row["reference_characters"]) for row in rows)
        substitutions = sum(int(row["substitutions"]) for row in rows)
        deletions = sum(int(row["deletions"]) for row in rows)
        insertions = sum(int(row["insertions"]) for row in rows)
        total_edits = sum(int(row["total_edits"]) for row in rows)
        character_edits = sum(int(row["character_edits"]) for row in rows)
        flagged_rows = [
            row for row in rows if flags.get(str(row["clip_id"]), False)
        ]
        table.append(
            {
                "bucket": bucket,
                "clips": len(rows),
                "reference_words": reference_words,
                "reference_characters": reference_characters,
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
                "total_edits": total_edits,
                "corpus_wer": (
                    total_edits / reference_words if reference_words else 0.0
                ),
                "macro_wer": sum(float(row["wer"]) for row in rows) / len(rows),
                "corpus_cer": (
                    character_edits / reference_characters
                    if reference_characters
                    else 0.0
                ),
                "macro_cer": sum(float(row["cer"]) for row in rows) / len(rows),
                "flagged_clips": len(flagged_rows),
                "clean_clips": len(rows) - len(flagged_rows),
                "insertions_from_flagged_clips": sum(
                    int(row["insertions"]) for row in flagged_rows
                ),
            }
        )
    return table


def _hallucination_view(
    scored_rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    summaries: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the committed hallucination section from diagnostics."""
    flags = {
        str(record["clip_id"]): bool(record["likely_repetitive_hallucination"])
        for record in diagnostics
    }
    flagged_ids = sorted(clip_id for clip_id, flag in flags.items() if flag)
    clean_rows = [row for row in scored_rows if not flags[str(row["clip_id"])]]
    flagged_rows = [row for row in scored_rows if flags[str(row["clip_id"])]]
    clean_reference_words = sum(int(row["reference_words"]) for row in clean_rows)
    clean_edits = sum(int(row["total_edits"]) for row in clean_rows)
    flagged_reference_words = sum(int(row["reference_words"]) for row in flagged_rows)
    flagged_edits = sum(int(row["total_edits"]) for row in flagged_rows)
    flagged_by_bin = Counter(
        str(row["english_ratio_bin"]) for row in flagged_rows
    )
    clean_by_bin = Counter(str(row["english_ratio_bin"]) for row in clean_rows)
    return {
        "likely_hallucination_clips": len(flagged_ids),
        "likely_hallucination_clip_ids": flagged_ids,
        "insertions_from_flagged_clips": sum(
            int(row["insertions"]) for row in flagged_rows
        ),
        "insertions_from_clean_clips": sum(
            int(row["insertions"]) for row in clean_rows
        ),
        "max_consecutive_repeat": int(summaries["max_consecutive_repeat"]),
        "max_repeated_phrase": int(summaries["max_repeated_phrase"]),
        "corpus_wer_excluding_flagged": (
            clean_edits / clean_reference_words if clean_reference_words else 0.0
        ),
        "corpus_wer_flagged_only": (
            flagged_edits / flagged_reference_words
            if flagged_reference_words
            else 0.0
        ),
        "flagged_by_bin": dict(sorted(flagged_by_bin.items())),
        "clean_clip_count_by_bin": dict(sorted(clean_by_bin.items())),
    }


def _compare_hallucination(
    recomputed: dict[str, Any],
    committed: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """Compare the hallucination section field by field."""
    fields: dict[str, Any] = {}
    matched = True
    for name in (
        "likely_hallucination_clips",
        "insertions_from_flagged_clips",
        "insertions_from_clean_clips",
        "max_consecutive_repeat",
        "max_repeated_phrase",
    ):
        if name not in committed:
            fields[name] = {"match": False, "error": "missing in committed"}
            matched = False
            continue
        entry = _field_entry(recomputed[name], committed[name], "int", tolerance)
        fields[name] = entry
        matched = matched and bool(entry["match"])
    for name in ("corpus_wer_excluding_flagged", "corpus_wer_flagged_only"):
        if name not in committed:
            fields[name] = {"match": False, "error": "missing in committed"}
            matched = False
            continue
        entry = _field_entry(recomputed[name], committed[name], "float", tolerance)
        fields[name] = entry
        matched = matched and bool(entry["match"])
    for name in ("flagged_by_bin", "clean_clip_count_by_bin"):
        if name not in committed:
            continue
        entry = _field_entry(recomputed[name], committed[name], "str", tolerance)
        entry["match"] = dict(recomputed[name]) == {
            str(key): int(value) for key, value in dict(committed[name]).items()
        }
        fields[name] = entry
        matched = matched and bool(entry["match"])
    committed_ids = [str(clip) for clip in committed.get("likely_hallucination_clip_ids", [])]
    recomputed_ids = recomputed["likely_hallucination_clip_ids"]
    ids_entry = {
        "match": set(committed_ids) == set(recomputed_ids)
        and len(committed_ids) == len(recomputed_ids),
        "recomputed": recomputed_ids,
        "committed": committed_ids,
    }
    fields["likely_hallucination_clip_ids"] = ids_entry
    matched = matched and bool(ids_entry["match"])
    return {"match": matched, "fields": fields}


def _compare_per_clip(
    scored_rows: list[dict[str, Any]],
    committed_rows: list[dict[str, Any]],
    manifest: pd.DataFrame,
    flags: dict[str, bool],
    tolerance: float,
    max_mismatch_details: int = 200,
) -> dict[str, Any]:
    """Compare every recomputed per-clip row with the committed rows."""
    manifest_lookup = manifest.set_index("clip_id").to_dict(orient="index")
    committed_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in committed_rows:
        clip_id = str(row.get("clip_id"))
        if clip_id in committed_by_id:
            duplicate_ids.append(clip_id)
        committed_by_id[clip_id] = row

    mismatches: list[dict[str, Any]] = []
    matched = not duplicate_ids

    recomputed_by_id = {str(row["clip_id"]): row for row in scored_rows}
    missing_in_committed = sorted(set(recomputed_by_id) - set(committed_by_id))
    extra_in_committed = sorted(set(committed_by_id) - set(recomputed_by_id))
    if missing_in_committed or extra_in_committed:
        matched = False

    for clip_id, recomputed in recomputed_by_id.items():
        committed = committed_by_id.get(clip_id)
        if committed is None:
            continue
        optional_values: dict[str, Any] = {
            "stratification_bucket": manifest_lookup.get(clip_id, {}).get(
                "stratification_bucket"
            ),
            "hallucination_flagged": flags.get(clip_id),
        }
        for committed_name, recomputed_name, kind in PER_CLIP_FIELDS:
            if committed_name not in committed:
                if committed_name in PER_CLIP_REQUIRED:
                    mismatches.append(
                        {
                            "clip_id": clip_id,
                            "field": committed_name,
                            "error": "missing in committed row",
                        }
                    )
                    matched = False
                continue
            entry = _field_entry(
                recomputed[recomputed_name], committed[committed_name], kind, tolerance
            )
            if not entry["match"]:
                entry["clip_id"] = clip_id
                entry["field"] = committed_name
                mismatches.append(entry)
                matched = False
        for committed_name in PER_CLIP_OPTIONAL:
            if committed_name in committed and optional_values.get(committed_name) is not None:
                recomputed_value = optional_values[committed_name]
                kind = "float" if isinstance(recomputed_value, float) else "str"
                if isinstance(recomputed_value, bool):
                    kind = "str"
                entry = _field_entry(
                    recomputed_value, committed[committed_name], kind, tolerance
                )
                if not entry["match"]:
                    entry["clip_id"] = clip_id
                    entry["field"] = committed_name
                    mismatches.append(entry)
                    matched = False

    truncated = len(mismatches) > max_mismatch_details
    return {
        "match": matched,
        "clips_compared": len(recomputed_by_id),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:max_mismatch_details],
        "truncated": truncated,
        "duplicate_committed_clip_ids": sorted(set(duplicate_ids)),
        "missing_in_committed": missing_in_committed,
        "extra_in_committed": extra_in_committed,
    }


def _load_raw_output(path: Path) -> list[dict[str, Any]]:
    """Load the raw Modal output, skipping the leading log line.

    The private archive file starts with a human-readable log line before the
    JSON array; the payload begins at the first ``[``.
    """
    text = path.read_text(encoding="utf-8")
    start = text.find("[")
    if start < 0:
        raise ValueError(f"raw output has no JSON array: {path}")
    payload = json.loads(text[start:])
    if not isinstance(payload, list):
        raise ValueError(f"raw output is not a JSON array: {path}")
    return payload


def _compare_raw_output(
    raw_rows: list[dict[str, Any]],
    predictions: pd.DataFrame,
    committed: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """Cross-check the raw inference output against predictions and artifact.

    Transcripts are compared in memory; only clip IDs and metadata sets are
    reported.
    """
    raw_by_id = {str(row.get("clip_id")): row for row in raw_rows}
    prediction_by_id = {
        str(row["clip_id"]): str(row["prediction"])
        for row in predictions.to_dict(orient="records")
    }
    missing_in_raw = sorted(set(prediction_by_id) - set(raw_by_id))
    extra_in_raw = sorted(set(raw_by_id) - set(prediction_by_id))
    mismatched_transcript_ids = sorted(
        clip_id
        for clip_id in set(prediction_by_id).intersection(raw_by_id)
        if raw_by_id[clip_id].get("transcript") != prediction_by_id[clip_id]
    )
    checks: dict[str, Any] = {
        "clip_set_match": not missing_in_raw and not extra_in_raw,
        "missing_in_raw": missing_in_raw,
        "extra_in_raw": extra_in_raw,
        "transcript_match": not mismatched_transcript_ids,
        "mismatched_transcript_clip_ids": mismatched_transcript_ids,
    }
    matched = checks["clip_set_match"] and checks["transcript_match"]

    for field, key in (
        ("model_id", "model_id"),
        ("task", "task"),
        ("language", "language"),
        ("precision", "precision"),
    ):
        values = sorted({str(row.get(key, "")) for row in raw_rows})
        checks[f"{field}_values"] = values
        inference_config = committed.get("inference_config")
        if isinstance(inference_config, dict) and field in inference_config:
            expected = str(inference_config[field])
            checks[f"{field}_match_config"] = values == [expected]
            matched = matched and checks[f"{field}_match_config"]

    per_clip = {
        str(row.get("clip_id")): row
        for row in committed.get("per_clip", [])
        if isinstance(row, dict)
    }
    runtime_mismatches: list[str] = []
    duration_mismatches: list[str] = []
    for clip_id, raw_row in raw_by_id.items():
        committed_row = per_clip.get(clip_id)
        if committed_row is None:
            continue
        if "runtime_sec" in raw_row and "inference_runtime_sec" in committed_row:
            if abs(float(raw_row["runtime_sec"]) - float(committed_row["inference_runtime_sec"])) > 1e-3:
                runtime_mismatches.append(clip_id)
        if "audio_duration_sec" in raw_row and "audio_duration_sec" in committed_row:
            if abs(float(raw_row["audio_duration_sec"]) - float(committed_row["audio_duration_sec"])) > 1e-3:
                duration_mismatches.append(clip_id)
    checks["per_clip_runtime_mismatch_ids"] = sorted(runtime_mismatches)
    checks["per_clip_duration_mismatch_ids"] = sorted(duration_mismatches)
    matched = matched and not runtime_mismatches and not duration_mismatches

    inference_runtime = committed.get("inference_runtime")
    if isinstance(inference_runtime, dict) and raw_by_id:
        total_runtime = sum(float(row.get("runtime_sec", 0.0)) for row in raw_rows)
        total_audio = sum(float(row.get("audio_duration_sec", 0.0)) for row in raw_rows)
        totals_ok = True
        if "total_gpu_inference_sec" in inference_runtime:
            totals_ok = totals_ok and abs(
                total_runtime - float(inference_runtime["total_gpu_inference_sec"])
            ) <= tolerance
        if "total_audio_sec" in inference_runtime:
            totals_ok = totals_ok and abs(
                total_audio - float(inference_runtime["total_audio_sec"])
            ) <= tolerance
        if "rtf" in inference_runtime and total_audio:
            totals_ok = totals_ok and abs(
                total_runtime / total_audio - float(inference_runtime["rtf"])
            ) <= tolerance
        checks["totals_match"] = totals_ok
        checks["recomputed_total_gpu_inference_sec"] = total_runtime
        checks["recomputed_total_audio_sec"] = total_audio
        checks["recomputed_rtf"] = (
            total_runtime / total_audio if total_audio else 0.0
        )
        matched = matched and totals_ok

        for map_key, raw_key in (
            ("per_clip_runtime_sec", "runtime_sec"),
            ("per_clip_audio_duration_sec", "audio_duration_sec"),
        ):
            committed_map = inference_runtime.get(map_key)
            if not isinstance(committed_map, dict):
                continue
            map_mismatches = sorted(
                clip_id
                for clip_id, value in committed_map.items()
                if clip_id in raw_by_id
                and abs(float(raw_by_id[clip_id].get(raw_key, 0.0)) - float(value)) > 1e-3
            )
            checks[f"{map_key}_mismatch_ids"] = map_mismatches
            matched = matched and not map_mismatches

    checks["match"] = matched
    return checks


def _compare_diagnostics(
    diagnostics: dict[str, Any],
    scored_rows: list[dict[str, Any]],
    recomputed_flags: dict[str, bool],
    diagnosis_summary: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """Cross-check the private diagnostics file against recomputation."""
    records = diagnostics.get("records", [])
    summary = diagnostics.get("summary", {})
    record_flags = {
        str(record.get("clip_id")): bool(record.get("likely_repetitive_hallucination"))
        for record in records
    }
    flag_mismatches = sorted(
        clip_id
        for clip_id in set(record_flags).union(recomputed_flags)
        if record_flags.get(clip_id) != recomputed_flags.get(clip_id)
    )
    checks: dict[str, Any] = {
        "records": len(records),
        "record_flag_mismatch_ids": flag_mismatches,
        "record_flags_match": not flag_mismatches,
    }
    matched = not flag_mismatches

    fields: dict[str, Any] = {}
    summary_fields = (
        ("utterances", "int"),
        ("total_reference_words", "int"),
        ("total_edits", "int"),
        ("corpus_wer", "float"),
        ("total_insertions", "int"),
        ("likely_hallucination_clips", "int"),
        ("max_consecutive_repeat", "int"),
        ("max_repeated_phrase", "int"),
    )
    total_reference_words = sum(int(row["reference_words"]) for row in scored_rows)
    total_edits = sum(int(row["total_edits"]) for row in scored_rows)
    total_insertions = sum(int(row["insertions"]) for row in scored_rows)
    recomputed_summary = {
        "utterances": len(scored_rows),
        "total_reference_words": total_reference_words,
        "total_edits": total_edits,
        "corpus_wer": (
            total_edits / total_reference_words if total_reference_words else 0.0
        ),
        "total_insertions": total_insertions,
        "likely_hallucination_clips": sum(1 for flag in recomputed_flags.values() if flag),
        "max_consecutive_repeat": int(diagnosis_summary["max_consecutive_repeat"]),
        "max_repeated_phrase": int(diagnosis_summary["max_repeated_phrase"]),
    }
    for name, kind in summary_fields:
        if name not in summary:
            continue
        expected = recomputed_summary.get(name)
        if expected is None:
            continue
        entry = _field_entry(expected, summary[name], kind, tolerance)
        fields[name] = entry
        matched = matched and bool(entry["match"])
    committed_ids = sorted(str(clip) for clip in summary.get("likely_hallucination_ids", []))
    recomputed_ids = sorted(
        clip_id for clip_id, flag in recomputed_flags.items() if flag
    )
    ids_entry = {
        "match": committed_ids == recomputed_ids,
        "recomputed": recomputed_ids,
        "committed": committed_ids,
    }
    fields["likely_hallucination_ids"] = ids_entry
    matched = matched and bool(ids_entry["match"])
    checks["summary_fields"] = fields
    checks["match"] = matched
    return checks


def _build_report(
    predictions_path: Path,
    manifest_path: Path,
    metrics_path: Path,
    raw_output_path: Optional[Path],
    diagnostics_path: Optional[Path],
    tolerance: float,
) -> dict[str, Any]:
    """Recompute everything and build the verification report."""
    predictions = _load_csv(predictions_path)
    manifest = _load_csv(manifest_path)
    predictions, manifest = _validate_inputs(predictions, manifest)
    committed = json.loads(metrics_path.read_text(encoding="utf-8"))

    records, failed = _build_records(predictions, manifest)
    summary = summarize_predictions(records)
    scored_rows = list(summary["results"])
    diagnostics = diagnose_predictions(records)
    diagnostics_summary = summarize_diagnostics(diagnostics)
    flags = {
        str(record["clip_id"]): bool(record["likely_repetitive_hallucination"])
        for record in diagnostics
    }

    recomputed_aggregate: dict[str, Any] = {
        "utterances": int(summary["utterances"]),
        "reference_words": int(summary["reference_words"]),
        "reference_characters": int(summary["reference_characters"]),
        "total_edits": int(summary["total_edits"]),
        "substitutions": sum(int(row["substitutions"]) for row in scored_rows),
        "deletions": sum(int(row["deletions"]) for row in scored_rows),
        "insertions": sum(int(row["insertions"]) for row in scored_rows),
        "wer": float(summary["wer"]),
        "cer": float(summary["cer"]),
        "mean_utterance_wer": float(summary["mean_utterance_wer"]),
        "mean_utterance_cer": float(summary["mean_utterance_cer"]),
        "normalization_version": HINGLISH_NORMALIZATION_VERSION,
    }

    checks: dict[str, Any] = {}

    committed_aggregate = committed.get("aggregate")
    if not isinstance(committed_aggregate, dict):
        raise ValueError("committed metrics file has no 'aggregate' object")
    aggregate_check = _compare_section(
        recomputed_aggregate,
        committed_aggregate,
        AGGREGATE_FIELDS,
        tolerance,
        required=frozenset(name for name, _, _ in AGGREGATE_FIELDS),
    )
    if (
        "normalization_version" in committed_aggregate
        or "normalization_version" in recomputed_aggregate
    ):
        entry = _field_entry(
            recomputed_aggregate["normalization_version"],
            committed_aggregate.get("normalization_version", ""),
            "str",
            tolerance,
        )
        aggregate_check["fields"]["normalization_version"] = entry
        aggregate_check["match"] = aggregate_check["match"] and bool(entry["match"])
    checks["aggregate"] = aggregate_check

    committed_per_clip = committed.get("per_clip")
    if isinstance(committed_per_clip, list):
        checks["per_clip"] = _compare_per_clip(
            scored_rows, committed_per_clip, manifest, flags, tolerance
        )
    else:
        checks["per_clip"] = {"skipped": True, "reason": "no per_clip in committed"}

    macro_wer_by_bin: dict[str, float] = {}
    macro_cer_by_bin: dict[str, float] = {}
    weighted_wer_by_bin: dict[str, float] = {}
    weighted_cer_by_bin: dict[str, float] = {}
    for row in _bucket_rows(scored_rows, {str(r["clip_id"]): str(r["english_ratio_bin"]) for r in scored_rows}, flags):
        bucket = str(row["bucket"])
        macro_wer_by_bin[bucket] = float(row["macro_wer"])
        macro_cer_by_bin[bucket] = float(row["macro_cer"])
        weighted_wer_by_bin[bucket] = float(row["corpus_wer"])
        weighted_cer_by_bin[bucket] = float(row["corpus_cer"])

    committed_bins = committed.get("composition_bins")
    if isinstance(committed_bins, dict):
        fields: dict[str, Any] = {}
        matched = True
        expected_keys = {f"wer_{bucket}" for bucket in macro_wer_by_bin}
        for key, committed_value in committed_bins.items():
            key = str(key)
            if key.startswith("wer_") and key[4:] in macro_wer_by_bin:
                entry = _field_entry(
                    macro_wer_by_bin[key[4:]], committed_value, "float", tolerance
                )
            elif key.startswith("cer_") and key[4:] in macro_cer_by_bin:
                entry = _field_entry(
                    macro_cer_by_bin[key[4:]], committed_value, "float", tolerance
                )
            else:
                entry = {"match": False, "error": "unknown composition bin key", "committed": committed_value}
            fields[key] = entry
            matched = matched and bool(entry["match"])
        missing_keys = sorted(expected_keys - {str(key) for key in committed_bins})
        if missing_keys:
            matched = False
            fields["_missing_bin_keys"] = {"match": False, "missing": missing_keys}
        checks["composition_bins"] = {
            "match": matched,
            "fields": fields,
            "note": "committed composition bins are macro (mean-of-utterance) rates; corpus-weighted rates are reported in bucket_tables",
        }
    else:
        checks["composition_bins"] = {"skipped": True, "reason": "no composition_bins in committed"}

    recomputed_hallucination = _hallucination_view(
        scored_rows, diagnostics, diagnostics_summary
    )
    committed_hallucination = committed.get("hallucination_diagnostics")
    if isinstance(committed_hallucination, dict):
        checks["hallucination_diagnostics"] = _compare_hallucination(
            recomputed_hallucination, committed_hallucination, tolerance
        )
    else:
        checks["hallucination_diagnostics"] = {
            "skipped": True,
            "reason": "no hallucination_diagnostics in committed",
        }

    committed_dataset = committed.get("dataset")
    if isinstance(committed_dataset, dict):
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        recomputed_dataset: dict[str, Any] = {
            "manifest_sha256": digest,
            "manifest_clips": len(manifest),
            "manifest_speakers": int(manifest["speaker_id"].nunique())
            if "speaker_id" in manifest.columns
            else None,
        }
        dataset_fields: dict[str, Any] = {}
        matched = True
        for name, kind in (
            ("manifest_sha256", "str"),
            ("manifest_clips", "int"),
            ("manifest_speakers", "int"),
        ):
            if name not in committed_dataset:
                continue
            expected = recomputed_dataset[name]
            if expected is None:
                continue
            entry = _field_entry(expected, committed_dataset[name], kind, tolerance)
            dataset_fields[name] = entry
            matched = matched and bool(entry["match"])
        for name, column in (
            ("stratification_bucket_distribution", "stratification_bucket"),
            ("record_level_bucket_distribution", "record_language_bucket"),
        ):
            if name in committed_dataset and column in manifest.columns:
                computed = dict(
                    sorted(Counter(manifest[column].astype(str)).items())
                )
                expected = {
                    str(key): int(value)
                    for key, value in dict(committed_dataset[name]).items()
                }
                entry = {
                    "match": computed == expected,
                    "recomputed": computed,
                    "committed": expected,
                }
                dataset_fields[name] = entry
                matched = matched and bool(entry["match"])
        checks["dataset"] = {
            "match": matched,
            "fields": dataset_fields,
            "failed_predictions": failed,
            "failed_prediction_count": len(failed),
        }
    else:
        checks["dataset"] = {"skipped": True, "reason": "no dataset in committed"}

    bucket_tables: dict[str, Any] = {
        "english_ratio_bin": {
            "source": "evaluation (code_switch_bin on ratio_english_words)",
            "rows": _bucket_rows(
                scored_rows,
                {str(row["clip_id"]): str(row["english_ratio_bin"]) for row in scored_rows},
                flags,
            ),
        }
    }
    if "stratification_bucket" in manifest.columns:
        bucket_tables["stratification_bucket"] = {
            "source": "manifest speaker-level stratification bucket",
            "rows": _bucket_rows(
                scored_rows,
                {
                    str(row["clip_id"]): str(row["stratification_bucket"])
                    for row in manifest.to_dict(orient="records")
                },
                flags,
            ),
        }
    if "record_language_bucket" in manifest.columns:
        bucket_tables["record_language_bucket"] = {
            "source": "manifest record-level language bucket",
            "rows": _bucket_rows(
                scored_rows,
                {
                    str(row["clip_id"]): str(row["record_language_bucket"])
                    for row in manifest.to_dict(orient="records")
                },
                flags,
            ),
        }

    if raw_output_path is not None:
        raw_rows = _load_raw_output(raw_output_path)
        checks["raw_output"] = _compare_raw_output(
            raw_rows, predictions, committed, tolerance
        )
    else:
        checks["raw_output"] = {"skipped": True, "reason": "no --raw-output given"}

    if diagnostics_path is not None:
        diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        checks["diagnostics_file"] = _compare_diagnostics(
            diagnostics_payload, scored_rows, flags, diagnostics_summary, tolerance
        )
    else:
        checks["diagnostics_file"] = {
            "skipped": True,
            "reason": "no --diagnostics given",
        }

    overall_match = all(
        bool(check.get("match"))
        for check in checks.values()
        if not check.get("skipped", False)
    )
    report = {
        "verification_version": VERIFICATION_VERSION,
        "privacy_note": (
            "Clip IDs, counts and rates only; reference transcripts and raw "
            "hypotheses are never included in this report."
        ),
        "inputs": {
            "predictions": str(predictions_path),
            "manifest": str(manifest_path),
            "metrics": str(metrics_path),
            "raw_output": str(raw_output_path) if raw_output_path else None,
            "diagnostics": str(diagnostics_path) if diagnostics_path else None,
        },
        "tolerance": tolerance,
        "overall_match": overall_match,
        "checks": checks,
        "recomputed": {
            "aggregate": recomputed_aggregate,
            "hallucination_diagnostics": recomputed_hallucination,
        },
        "bucket_tables": bucket_tables,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a committed Hinglish baseline metrics artifact against "
            "private predictions, the frozen manifest and optional raw outputs"
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, default=None)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to also write the JSON verification report",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            "Absolute tolerance for committed rates that are rounded to four "
            f"decimals (default {DEFAULT_TOLERANCE})"
        ),
    )
    args = parser.parse_args()

    try:
        report = _build_report(
            args.predictions.resolve(),
            args.manifest.resolve(),
            args.metrics.resolve(),
            args.raw_output.resolve() if args.raw_output else None,
            args.diagnostics.resolve() if args.diagnostics else None,
            args.tolerance,
        )
    except (ValueError, KeyError, OSError) as exc:
        print(f"verification input error: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        print(f"verification report written to {output_path}", file=sys.stderr)

    if report["overall_match"]:
        print("verification passed: all checks matched", file=sys.stderr)
        return 0
    failed_checks = [
        name
        for name, check in report["checks"].items()
        if not check.get("skipped", False) and not check.get("match", False)
    ]
    print(
        "verification FAILED: mismatched checks: " + ", ".join(failed_checks),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

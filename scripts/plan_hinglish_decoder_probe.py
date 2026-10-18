#!/usr/bin/env python3
"""Freeze the Phase 7.1 controlled decoder-probe plan (probe J1) offline.

This builder derives the frozen 16-clip development-cohort selection for the
single paid Modal T4 decoder probe from artifacts that already exist:

* the committed per-clip failure intelligence
  (``results/hinglish/failure_intelligence.json``) for insertions, loop
  flags, heuristic flags, and failure classes;
* the archived raw Modal baseline output (``raw_modal_output.json``) for
  the per-clip measured baseline runtimes and audio durations;
* the archived frozen validation manifest and holdout reserve pool for
  development-cohort membership and reserve exclusion (R4);
* the private WAV archive for the SHA-256 of every selected clip's audio.

No inference runs, no network access, no tokenizer or model downloads, and
no audio is read into any output. The committed plan contains clip IDs,
numeric statistics, booleans, and checksums only; transcript text and
private filesystem paths are excluded by construction and verified by an
explicit safety scan before writing.

Pre-registered selection rules (identical text lives in the integrity
subagent's prompt) are implemented verbatim in :func:`derive_categories`,
:func:`build_selection`, and :func:`validate_membership`. If the derivation
contradicts the mandate (8 ceiling-word-loop / 3 ceiling-char-loop /
2 early-loop / 3 control clips, 16 total, all inside the development
cohort, none in the holdout reserve), the builder stops with exit code 3
and never improvises a substitute selection.

Cost and time estimates are computed with conservative per-configuration
clamps (C0 = archived measured runtime, C1 = ``min(measured, 2.0)``,
C2 = ``max(measured, 3.75)`` seconds) plus one-off constants (model load
90 s, container overhead 60 s). Hard gates fail with exit code 2 when the
estimated wall exceeds 420 s or the estimated cost exceeds $0.10.

Exit codes:
    0: derivation and mandate checks passed; dry run printed or freeze
       written and self-checked.
    2: cost/timeout gate failure (wall > 420 s or cost > $0.10).
    3: the derivation contradicts the pre-registered mandate.
    4: bad input (missing or unreadable files, malformed payloads,
       overwrite refusal without ``--force``).
    5: post-freeze self-check failed (committed plan does not match the
       private freeze copy).

Example (dry run, the default)::

    python scripts/plan_hinglish_decoder_probe.py

Example (freeze; refuses to overwrite without ``--force``)::

    python scripts/plan_hinglish_decoder_probe.py --freeze
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROBE_VERSION = "mendspeech.hinglish_decoder_probe.v1"
GPU = "T4"
RATE_USD_PER_SEC = 0.000164
FUNCTION_TIMEOUT_SEC = 420
PLANNING_BUDGET_USD = 0.10
MODEL_ID = "openai/whisper-small"
MODEL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
LANGUAGE = "hi"
TASK = "transcribe"
PRECISION = "float16"
NORMALIZER_VERSION = "mendspeech.hinglish.v1"

CONFIG_ORDER: Tuple[str, ...] = ("C0", "C1", "C2")
CONFIG_PURPOSES: Dict[str, str] = {
    "C0": "baseline decoding + instrumentation",
    "C1": "single-parameter delta max_new_tokens=160",
    "C2": "single-parameter delta no_repeat_ngram_size=4",
}
C1_MAX_PER_CLIP_SEC = 2.0
C2_MIN_PER_CLIP_SEC = 3.75
MODEL_LOAD_SEC = 90.0
CONTAINER_OVERHEAD_SEC = 60.0
MAX_TOTAL_WALL_SEC = 420.0
MAX_ESTIMATED_COST_USD = 0.10

# R1: the runtime ceiling window observed in the frozen T4 baseline
# (inclusive on both ends): 420389... starts at 3.687 s and 344267... ends
# at 3.734 s; the next runtimes below and above the window are far away.
CEILING_RUNTIME_MIN_SEC = 3.687
CEILING_RUNTIME_MAX_SEC = 3.734
CEILING_CLASS = "generation_cap_with_repetition"

# R2 category names (the plan uses the short "control" label for the
# category after the R2 control_candidate criteria are satisfied).
CATEGORY_CEILING_WORD_LOOP = "ceiling_word_loop"
CATEGORY_CEILING_CHAR_LOOP = "ceiling_char_loop"
CATEGORY_EARLY_LOOP = "early_loop"
CATEGORY_CONTROL = "control"
CATEGORY_EXCLUDED = "excluded"
MANDATED_CATEGORY_COUNTS: Dict[str, int] = {
    CATEGORY_CEILING_WORD_LOOP: 8,
    CATEGORY_CEILING_CHAR_LOOP: 3,
    CATEGORY_EARLY_LOOP: 2,
    CATEGORY_CONTROL: 3,
}
MANDATED_TOTAL_CLIPS = sum(MANDATED_CATEGORY_COUNTS.values())

# R3: the early-loop set is fully pre-registered, and the three controls are
# pre-registered by clip-ID prefix (slowest clean decode, median clean,
# shortest clip per the Phase 7.0 design).
EARLY_LOOP_ENFORCED_IDS: Tuple[str, ...] = (
    "304558_c8Ba68BmVqNaBudE_0000",
    "399287_1TuGgwSGEBVVydVA_0000",
)
CONTROL_CLIP_ID_PREFIXES: Tuple[str, ...] = ("111627_", "136706_", "415128_")

SELECTION_RULES: Dict[str, str] = {
    "R1": (
        "source = results/hinglish/failure_intelligence.json per-clip records; "
        f"ceiling membership = baseline runtime within "
        f"[{CEILING_RUNTIME_MIN_SEC}, {CEILING_RUNTIME_MAX_SEC}] s inclusive "
        f"OR failure class {CEILING_CLASS}; both signals must agree for "
        "every development-cohort clip."
    ),
    "R2": (
        "ceiling_word_loop = ceiling AND word-loop; ceiling_char_loop = "
        "ceiling AND char-loop AND NOT word-loop; early_loop = NOT ceiling "
        "AND (word-loop OR char-loop); control = none of those categories "
        "AND not heuristic-flagged."
    ),
    "R3": (
        "frozen set = top-8 ceiling_word_loop by insertion count descending "
        "with tie-break clip_id ascending + ALL ceiling_char_loop (exactly 3) "
        "+ ALL early_loop (must be exactly "
        f"{sorted(EARLY_LOOP_ENFORCED_IDS)}) + the 3 controls resolved by "
        f"clip-ID prefixes {list(CONTROL_CLIP_ID_PREFIXES)} (each must satisfy "
        "the control criteria)."
    ),
    "R4": (
        "all frozen clips must be development-cohort manifest clip IDs whose "
        "WAV files hash-verify, and none of their speakers may appear in the "
        "frozen holdout reserve pool."
    ),
    "note": (
        "Control prefixes were pre-registered in the Phase 7.0 design "
        "(111627 slowest clean decode, 136706 median clean, 415128 shortest "
        "clip). The plan never modifies the normalizer "
        f"{NORMALIZER_VERSION}."
    ),
}

PRIVATE_ONLY_PLAN_FIELDS: FrozenSet[str] = frozenset({"private_sources"})

ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {
        "probe_version",
        "created_utc",
        "gpu",
        "rate_usd_per_sec",
        "function_timeout_sec",
        "planning_budget_usd",
        "model",
        "language",
        "task",
        "precision",
        "normalizer",
        "configs",
        "estimator_constants",
        "selection_rules",
        "provenance",
        "clips",
        "estimate",
    }
)
ALLOWED_CLIP_KEYS = frozenset(
    {
        "clip_id",
        "category",
        "baseline_runtime_sec",
        "insertions",
        "has_word_loop",
        "has_char_loop",
        "heuristic_flagged",
        "duration_sec",
        "audio_sha256",
    }
)

INTELLIGENCE_REQUIRED_KEYS = (
    "clip_id",
    "failure_class",
    "insertions",
    "has_word_loop",
    "has_char_loop",
    "heuristic_flag",
)

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
FORBIDDEN_COMMITTED_SUBSTRINGS = ("/home/", "backups/", "\\", "reference_transcript")

DEFAULT_INTELLIGENCE_JSON = REPO_ROOT / "results" / "hinglish" / "failure_intelligence.json"
DEFAULT_AUDIO_DIR = Path("/home/ashwin/backups/mucs-validation/audio")
DEFAULT_MANIFEST = Path("/home/ashwin/backups/mucs-validation/frozen_validation_manifest.csv")
DEFAULT_RESERVE_JSON = Path("/home/ashwin/backups/mucs-validation/holdout_pool_v0.json")
DEFAULT_RAW_OUTPUT_JSON = Path(
    "/home/ashwin/backups/mucs-validation-t4-run-2026-10-16/raw_modal_output.json"
)
DEFAULT_PRIVATE_DIR = Path("/home/ashwin/backups/mucs-j1-probe-2026-10-18/freeze")
DEFAULT_PLAN_OUT = REPO_ROOT / "results" / "hinglish" / "decoder_probe_plan.json"
DEFAULT_REPORT_OUT = Path("/tmp/opencode/phase7.1/plan_report.json")


class InputError(RuntimeError):
    """Raised when an input file is missing, unreadable, or malformed."""


class GateError(RuntimeError):
    """Raised when the estimated wall time or cost exceeds a hard gate."""


class MandateError(RuntimeError):
    """Raised when the derivation contradicts the pre-registered mandate."""


class SelfCheckError(RuntimeError):
    """Raised when the post-freeze self-check fails."""


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


@dataclass(frozen=True)
class SelectionFeatures:
    """Per-clip features used by the pre-registered selection rules.

    Attributes:
        clip_id: Development-cohort clip identifier.
        failure_class: Failure class from the committed intelligence JSON.
        insertions: Strict insertion count from the intelligence JSON.
        has_word_loop: Word-level repetition evidence flag.
        has_char_loop: Character-level repetition evidence flag.
        heuristic_flag: Content-heuristic flag.
        baseline_runtime_sec: Archived measured baseline runtime in seconds.
        audio_duration_sec: Archived audio duration in seconds.
    """

    clip_id: str
    failure_class: str
    insertions: int
    has_word_loop: bool
    has_char_loop: bool
    heuristic_flag: bool
    baseline_runtime_sec: float
    audio_duration_sec: float


@dataclass(frozen=True)
class SelectedClip:
    """One frozen plan clip (no transcript text, private paths, or audio).

    Attributes:
        clip_id: Development-cohort clip identifier.
        category: One of the four mandated categories.
        baseline_runtime_sec: Archived measured baseline runtime in seconds.
        insertions: Strict insertion count from the intelligence JSON.
        has_word_loop: Word-level repetition evidence flag.
        has_char_loop: Character-level repetition evidence flag.
        heuristic_flag: Content-heuristic flag.
        duration_sec: Archived audio duration in seconds.
        audio_sha256: SHA-256 of the development-cohort WAV file.
    """

    clip_id: str
    category: str
    baseline_runtime_sec: float
    insertions: int
    has_word_loop: bool
    has_char_loop: bool
    heuristic_flag: bool
    duration_sec: float
    audio_sha256: str


def _read_json(path: Path) -> Dict[str, Any]:
    """Read and parse a JSON object file.

    Args:
        path: JSON file path.

    Returns:
        Parsed dictionary payload.

    Raises:
        InputError: If the file is missing or not a JSON object.
    """
    if not path.is_file():
        raise InputError(f"missing JSON input: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"unreadable JSON input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputError(f"expected a JSON object in {path}")
    return payload


def load_intelligence(path: Path) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Load the committed failure intelligence per-clip records.

    Args:
        path: Path to ``failure_intelligence.json``.

    Returns:
        Tuple of (mapping of clip ID to per-clip record, SHA-256 of the
        file bytes). Records are copied; the payload is never mutated.

    Raises:
        InputError: If the file is missing or per-clip records are malformed.
    """
    payload = _read_json(path)
    records = payload.get("per_clip")
    if not isinstance(records, list) or not records:
        raise InputError(f"{path}: missing or empty per_clip records")
    by_clip: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise InputError(f"{path}: per_clip entries must be objects")
        clip_id = record.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise InputError(f"{path}: per_clip entry without a clip_id")
        missing = [key for key in INTELLIGENCE_REQUIRED_KEYS if key not in record]
        if missing:
            raise InputError(f"{path}: clip {clip_id} missing keys {missing}")
        if clip_id in by_clip:
            raise InputError(f"{path}: duplicate per_clip record for {clip_id}")
        by_clip[clip_id] = dict(record)
    return by_clip, sha256_file(path)


def load_manifest(path: Path) -> Tuple[Dict[str, str], str]:
    """Load the frozen development-cohort manifest clip/speaker IDs.

    Only the ``clip_id`` and ``speaker_id`` columns are extracted. All other
    columns (including any reference transcript column) are ignored and are
    never copied anywhere.

    Args:
        path: Path to the frozen validation manifest CSV.

    Returns:
        Tuple of (mapping of clip ID to speaker ID, SHA-256 of the file
        bytes).

    Raises:
        InputError: If the file is missing, malformed, or has duplicates.
    """
    if not path.is_file():
        raise InputError(f"missing manifest input: {path}")
    speakers: Dict[str, str] = {}
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "clip_id" not in reader.fieldnames:
                raise InputError(f"{path}: manifest has no clip_id column")
            if "speaker_id" not in reader.fieldnames:
                raise InputError(f"{path}: manifest has no speaker_id column")
            for row in reader:
                clip_id = (row.get("clip_id") or "").strip()
                if not clip_id:
                    raise InputError(f"{path}: empty clip_id row")
                if clip_id in speakers:
                    raise InputError(f"{path}: duplicate clip_id row for {clip_id}")
                speakers[clip_id] = (row.get("speaker_id") or "").strip()
    except (OSError, csv.Error) as exc:
        raise InputError(f"unreadable manifest {path}: {exc}") from exc
    if not speakers:
        raise InputError(f"{path}: manifest has no rows")
    return speakers, sha256_file(path)


def parse_raw_output(path: Path) -> Tuple[Dict[str, Dict[str, float]], str]:
    """Parse the archived raw Modal output, skipping any log preamble.

    Only ``runtime_sec`` and ``audio_duration_sec`` are extracted per clip.
    The ``transcript`` key is never carried beyond this function.

    Args:
        path: Path to ``raw_modal_output.json``.

    Returns:
        Tuple of (mapping of clip ID to ``{"runtime_sec": ..., 
        "audio_duration_sec": ...}``, SHA-256 of the file bytes).

    Raises:
        InputError: If no JSON array is present, a record is malformed, or a
            runtime/duration is not a positive finite number.
    """
    if not path.is_file():
        raise InputError(f"missing raw-output input: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        start = text.find("[")
        if start < 0:
            raise ValueError("no JSON array found")
        payload = json.loads(text[start:])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise InputError(f"unreadable raw output {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise InputError(f"expected a JSON array in {path}")
    extracted: Dict[str, Dict[str, float]] = {}
    for record in payload:
        if not isinstance(record, dict) or "clip_id" not in record:
            raise InputError(f"{path}: malformed raw-output record")
        clip_id = str(record["clip_id"])
        try:
            runtime = float(record["runtime_sec"])
            duration = float(record["audio_duration_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InputError(
                f"{path}: clip {clip_id} missing numeric runtime/duration"
            ) from exc
        if not runtime > 0.0 or not duration > 0.0:
            raise InputError(
                f"{path}: clip {clip_id} has non-positive runtime/duration"
            )
        if clip_id in extracted:
            raise InputError(f"{path}: duplicate raw-output record for {clip_id}")
        extracted[clip_id] = {
            "runtime_sec": runtime,
            "audio_duration_sec": duration,
        }
    return extracted, sha256_file(path)


def load_reserve_speakers(path: Path) -> Tuple[Set[str], str]:
    """Load the frozen holdout reserve speaker IDs.

    Args:
        path: Path to ``holdout_pool_v0.json``.

    Returns:
        Tuple of (set of reserve speaker IDs, SHA-256 of the file bytes).

    Raises:
        InputError: If the payload is malformed.
    """
    payload = _read_json(path)
    entries = payload.get("speakers")
    if not isinstance(entries, list):
        raise InputError(f"{path}: missing speakers list")
    speakers: Set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("speaker_id"), str):
            raise InputError(f"{path}: malformed speaker entry")
        speaker_id = entry["speaker_id"]
        if speaker_id in speakers:
            raise InputError(f"{path}: duplicate reserve speaker {speaker_id}")
        speakers.add(speaker_id)
    return speakers, sha256_file(path)


def derive_categories(
    intel: Dict[str, Dict[str, Any]],
    raw: Dict[str, Dict[str, float]],
    manifest: Dict[str, str],
) -> Tuple[Dict[str, SelectionFeatures], Dict[str, int]]:
    """Apply R1/R2 and return per-clip features with derived categories.

    R1 requires that, for every development-cohort clip, membership in the
    runtime-ceiling window agrees with the ``generation_cap_with_repetition``
    failure class. R2 then assigns each clip to exactly one category:
    ``ceiling_word_loop``, ``ceiling_char_loop``, ``early_loop``,
    ``control`` (satisfying the control-candidate criteria), or ``excluded``.

    Args:
        intel: Per-clip intelligence records keyed by clip ID.
        raw: Archived per-clip runtimes and durations keyed by clip ID.
        manifest: Manifest clip IDs (development cohort) mapped to speakers.

    Returns:
        Tuple of (features keyed by clip ID, category counts).

    Raises:
        MandateError: If the two ceiling signals disagree, or the manifest
            and intelligence clip populations differ.
        InputError: If a clip's numeric features cannot be interpreted.
    """
    if set(intel) != set(manifest):
        only_intel = sorted(set(intel) - set(manifest))
        only_manifest = sorted(set(manifest) - set(intel))
        raise MandateError(
            "R1 population mismatch between manifest and intelligence JSON "
            f"(manifest-only: {only_manifest}; intelligence-only: {only_intel})"
        )
    features: Dict[str, SelectionFeatures] = {}
    disagreements: List[str] = []
    for clip_id, manifest_speaker in manifest.items():
        del manifest_speaker  # speaker IDs are checked later against the reserve
        record = intel[clip_id]
        raw_row = raw.get(clip_id)
        if raw_row is None:
            raise InputError(
                f"archived raw output has no record for manifest clip {clip_id}"
            )
        runtime = raw_row["runtime_sec"]
        duration = raw_row["audio_duration_sec"]
        in_window = CEILING_RUNTIME_MIN_SEC <= runtime <= CEILING_RUNTIME_MAX_SEC
        by_class = record["failure_class"] == CEILING_CLASS
        if in_window != by_class:
            disagreements.append(
                f"{clip_id}: runtime {runtime:.4f} s in-window={in_window} "
                f"failure_class={record['failure_class']}"
            )
        try:
            insertions = int(record["insertions"])
        except (TypeError, ValueError) as exc:
            raise InputError(f"clip {clip_id} has non-integer insertions") from exc
        features[clip_id] = SelectionFeatures(
            clip_id=clip_id,
            failure_class=str(record["failure_class"]),
            insertions=insertions,
            has_word_loop=bool(record["has_word_loop"]),
            has_char_loop=bool(record["has_char_loop"]),
            heuristic_flag=bool(record["heuristic_flag"]),
            baseline_runtime_sec=runtime,
            audio_duration_sec=duration,
        )
    if disagreements:
        raise MandateError(
            "R1 ceiling agreement failed for "
            f"{len(disagreements)} clip(s): {'; '.join(disagreements)}"
        )

    counts: Dict[str, int] = {}
    for feature in features.values():
        ceiling = (
            CEILING_RUNTIME_MIN_SEC
            <= feature.baseline_runtime_sec
            <= CEILING_RUNTIME_MAX_SEC
        )
        if ceiling and feature.has_word_loop:
            category = CATEGORY_CEILING_WORD_LOOP
        elif ceiling and feature.has_char_loop and not feature.has_word_loop:
            category = CATEGORY_CEILING_CHAR_LOOP
        elif not ceiling and (feature.has_word_loop or feature.has_char_loop):
            category = CATEGORY_EARLY_LOOP
        elif not feature.heuristic_flag:
            category = CATEGORY_CONTROL
        else:
            category = CATEGORY_EXCLUDED
        counts[category] = counts.get(category, 0) + 1
    return features, counts


def build_selection(
    features: Dict[str, SelectionFeatures],
    counts: Dict[str, int],
) -> List[SelectedClip]:
    """Build the frozen 16-clip selection per rule R3.

    Args:
        features: Per-clip features keyed by clip ID (categories must have
            been derived already).
        counts: Category counts from :func:`derive_categories`.

    Returns:
        The ordered frozen selection (ceiling word-loop top-8 first, then
        ceiling char-loop, early-loop, and pre-registered controls).

    Raises:
        MandateError: If any mandated count, membership rule, or exact-set
            requirement is violated.
    """
    def _category_members(category: str) -> List[SelectionFeatures]:
        return [f for f in features.values() if _classify(features, f) == category]

    ceiling_word_loop = _category_members(CATEGORY_CEILING_WORD_LOOP)
    if len(ceiling_word_loop) < MANDATED_CATEGORY_COUNTS[CATEGORY_CEILING_WORD_LOOP]:
        raise MandateError(
            f"{CATEGORY_CEILING_WORD_LOOP} has {len(ceiling_word_loop)} clips; "
            f"the mandate requires at least "
            f"{MANDATED_CATEGORY_COUNTS[CATEGORY_CEILING_WORD_LOOP]}"
        )
    top8 = sorted(
        ceiling_word_loop, key=lambda f: (-f.insertions, f.clip_id)
    )[: MANDATED_CATEGORY_COUNTS[CATEGORY_CEILING_WORD_LOOP]]

    ceiling_char_loop = sorted(
        _category_members(CATEGORY_CEILING_CHAR_LOOP), key=lambda f: f.clip_id
    )
    if len(ceiling_char_loop) != MANDATED_CATEGORY_COUNTS[CATEGORY_CEILING_CHAR_LOOP]:
        raise MandateError(
            f"{CATEGORY_CEILING_CHAR_LOOP} count is {len(ceiling_char_loop)}; "
            f"mandate requires exactly "
            f"{MANDATED_CATEGORY_COUNTS[CATEGORY_CEILING_CHAR_LOOP]}"
        )

    early_loop = sorted(_category_members(CATEGORY_EARLY_LOOP), key=lambda f: f.clip_id)
    early_ids = tuple(sorted(f.clip_id for f in early_loop))
    if early_ids != tuple(sorted(EARLY_LOOP_ENFORCED_IDS)):
        raise MandateError(
            f"early_loop set {early_ids} does not match the pre-registered "
            f"mandate {sorted(EARLY_LOOP_ENFORCED_IDS)}"
        )

    controls: List[SelectionFeatures] = []
    for prefix in CONTROL_CLIP_ID_PREFIXES:
        matches = [f for f in features.values() if f.clip_id.startswith(prefix)]
        if len(matches) != 1:
            resolved = sorted(f.clip_id for f in matches)
            raise MandateError(
                f"control prefix {prefix} resolves to {resolved}; expected "
                "exactly one development-cohort clip"
            )
        match = matches[0]
        if _classify(features, match) != CATEGORY_CONTROL:
            raise MandateError(
                f"control clip {match.clip_id} does not satisfy the control "
                "criteria (none of the loop categories and not heuristic-flagged)"
            )
        controls.append(match)
    if len(controls) != MANDATED_CATEGORY_COUNTS[CATEGORY_CONTROL]:
        raise MandateError(
            f"resolved {len(controls)} controls; mandate requires exactly "
            f"{MANDATED_CATEGORY_COUNTS[CATEGORY_CONTROL]}"
        )

    ordered: List[SelectionFeatures] = top8 + ceiling_char_loop + early_loop + controls
    selected_ids = [f.clip_id for f in ordered]
    if len(set(selected_ids)) != len(selected_ids):
        raise MandateError(f"selection contains duplicate clip IDs: {selected_ids}")
    if len(selected_ids) != MANDATED_TOTAL_CLIPS:
        raise MandateError(
            f"selection has {len(selected_ids)} clips; mandate requires "
            f"{MANDATED_TOTAL_CLIPS}"
        )
    final_counts: Dict[str, int] = {}
    for feature in ordered:
        category = _classify(features, feature)
        final_counts[category] = final_counts.get(category, 0) + 1
    if final_counts != MANDATED_CATEGORY_COUNTS:
        raise MandateError(
            f"final category counts {final_counts} != mandate {MANDATED_CATEGORY_COUNTS}"
        )

    return [
        SelectedClip(
            clip_id=f.clip_id,
            category=_category_name(_classify(features, f)),
            baseline_runtime_sec=round(f.baseline_runtime_sec, 4),
            insertions=f.insertions,
            has_word_loop=f.has_word_loop,
            has_char_loop=f.has_char_loop,
            heuristic_flag=f.heuristic_flag,
            duration_sec=round(f.audio_duration_sec, 4),
            audio_sha256="",
        )
        for f in ordered
    ]


def _classify(features: Dict[str, SelectionFeatures], feature: SelectionFeatures) -> str:
    """Re-derive the R2 category for one feature.

    Args:
        features: Full feature mapping (unused; kept for symmetry).
        feature: Feature to classify.

    Returns:
        One of the five category labels.
    """
    del features
    ceiling = (
        CEILING_RUNTIME_MIN_SEC <= feature.baseline_runtime_sec <= CEILING_RUNTIME_MAX_SEC
    )
    if ceiling and feature.has_word_loop:
        return CATEGORY_CEILING_WORD_LOOP
    if ceiling and feature.has_char_loop and not feature.has_word_loop:
        return CATEGORY_CEILING_CHAR_LOOP
    if not ceiling and (feature.has_word_loop or feature.has_char_loop):
        return CATEGORY_EARLY_LOOP
    if not feature.heuristic_flag:
        return CATEGORY_CONTROL
    return CATEGORY_EXCLUDED


def _category_name(internal: str) -> str:
    """Map the internal category label to the plan category name.

    Args:
        internal: Internal category label.

    Returns:
        Plan category name (``excluded`` never reaches the plan).
    """
    if internal == CATEGORY_EXCLUDED:
        raise MandateError("excluded clips must never enter the frozen selection")
    return internal


def validate_membership(
    selected: Sequence[SelectedClip],
    manifest: Dict[str, str],
    reserve_speakers: Set[str],
) -> None:
    """Apply rule R4: development-cohort membership and reserve exclusion.

    Args:
        selected: Frozen selection.
        manifest: Manifest clip IDs mapped to development-cohort speakers.
        reserve_speakers: Holdout reserve speaker IDs.

    Raises:
        MandateError: If a selected clip is outside the development cohort
            or one of its speakers is in the holdout reserve.
    """
    for clip in selected:
        if clip.clip_id not in manifest:
            raise MandateError(
                f"selected clip {clip.clip_id} is not a development-cohort "
                "manifest clip ID (R4)"
            )
        speaker_id = manifest[clip.clip_id]
        if speaker_id in reserve_speakers:
            raise MandateError(
                f"selected clip {clip.clip_id} speaker {speaker_id or '<unknown>'} "
                "appears in the holdout reserve pool (R4)"
            )


def attach_audio_hashes(selected: List[SelectedClip], audio_dir: Path) -> List[SelectedClip]:
    """Compute the SHA-256 of every selected clip's WAV file.

    Args:
        selected: Frozen selection.
        audio_dir: Directory holding ``{clip_id}.wav`` files.

    Returns:
        The selection with ``audio_sha256`` populated.

    Raises:
        InputError: If a selected clip's WAV file is missing.
    """
    with_hashes: List[SelectedClip] = []
    for clip in selected:
        audio_path = audio_dir / f"{clip.clip_id}.wav"
        if not audio_path.is_file():
            raise InputError(f"missing selected audio file: {audio_path}")
        with_hashes.append(
            SelectedClip(
                clip_id=clip.clip_id,
                category=clip.category,
                baseline_runtime_sec=clip.baseline_runtime_sec,
                insertions=clip.insertions,
                has_word_loop=clip.has_word_loop,
                has_char_loop=clip.has_char_loop,
                heuristic_flag=clip.heuristic_flag,
                duration_sec=clip.duration_sec,
                audio_sha256=sha256_file(audio_path),
            )
        )
    return with_hashes


def clip_config_seconds(measured: float) -> Tuple[float, float, float]:
    """Return per-clip (C0, C1, C2) estimate seconds for one clip.

    C0 uses the archived measured runtime; C1 is clamped at
    ``C1_MAX_PER_CLIP_SEC`` from above (``min``); C2 is floored at
    ``C2_MIN_PER_CLIP_SEC`` from below (``max``).

    Args:
        measured: Archived measured baseline runtime in seconds.

    Returns:
        Tuple of (C0 seconds, C1 seconds, C2 seconds).
    """
    return (measured, min(measured, C1_MAX_PER_CLIP_SEC), max(measured, C2_MIN_PER_CLIP_SEC))


def build_estimate(selected: Sequence[SelectedClip]) -> Dict[str, Any]:
    """Build the conservative wall-time and cost estimate block.

    Args:
        selected: Frozen selection with measured baseline runtimes.

    Returns:
        Estimate dictionary with per-config sums, per-clip values, the total
        wall time, the estimated GPU cost, and the timeout worst case.

    Raises:
        GateError: If the estimated wall exceeds ``MAX_TOTAL_WALL_SEC`` or
            the estimated cost exceeds ``MAX_ESTIMATED_COST_USD``.
    """
    per_clip: Dict[str, Dict[str, float]] = {}
    config_sum: Dict[str, float] = {name: 0.0 for name in CONFIG_ORDER}
    for clip in selected:
        c0, c1, c2 = clip_config_seconds(clip.baseline_runtime_sec)
        per_clip[clip.clip_id] = {"C0": round(c0, 4), "C1": round(c1, 4), "C2": round(c2, 4)}
        for name, value in zip(CONFIG_ORDER, (c0, c1, c2)):
            config_sum[name] += value
    decoded_wall = sum(config_sum[name] for name in CONFIG_ORDER)
    total_wall = decoded_wall + MODEL_LOAD_SEC + CONTAINER_OVERHEAD_SEC
    cost = total_wall * RATE_USD_PER_SEC
    timeout_worst_case = FUNCTION_TIMEOUT_SEC * RATE_USD_PER_SEC
    if total_wall > MAX_TOTAL_WALL_SEC:
        raise GateError(
            f"estimated wall {total_wall:.3f} s exceeds the function-timeout "
            f"gate {MAX_TOTAL_WALL_SEC:.1f} s"
        )
    if cost > MAX_ESTIMATED_COST_USD:
        raise GateError(
            f"estimated GPU cost ${cost:.5f} exceeds the planning budget "
            f"gate ${MAX_ESTIMATED_COST_USD:.2f}"
        )
    return {
        "config_inference_seconds": {
            name: round(config_sum[name], 4) for name in CONFIG_ORDER
        },
        "per_clip_config_seconds": per_clip,
        "model_load_sec": MODEL_LOAD_SEC,
        "container_overhead_sec": CONTAINER_OVERHEAD_SEC,
        "decoded_wall_sec": round(decoded_wall, 4),
        "total_wall_sec": round(total_wall, 4),
        "estimated_gpu_cost_usd": round(cost, 6),
        "timeout_worst_case_usd": round(timeout_worst_case, 6),
        "clips_total_audio_sec": round(
            sum(clip.duration_sec for clip in selected), 4
        ),
    }


def build_plan(
    selected: Sequence[SelectedClip],
    created_utc: str,
    provenance: Dict[str, Any],
    estimate: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the committed plan payload.

    Args:
        selected: Frozen selection with audio hashes.
        created_utc: Pinned or current UTC timestamp (``YYYY-MM-DDTHH:MM:SSZ``).
        provenance: Provenance block (hashes and source notes only).
        estimate: Estimate block from :func:`build_estimate`.

    Returns:
        The committed plan dictionary (safe content only).
    """
    return {
        "probe_version": PROBE_VERSION,
        "created_utc": created_utc,
        "gpu": GPU,
        "rate_usd_per_sec": RATE_USD_PER_SEC,
        "function_timeout_sec": FUNCTION_TIMEOUT_SEC,
        "planning_budget_usd": PLANNING_BUDGET_USD,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "language": LANGUAGE,
        "task": TASK,
        "precision": PRECISION,
        "normalizer": NORMALIZER_VERSION,
        "configs": {
            "C0": {
                "purpose": CONFIG_PURPOSES["C0"],
                "generate_kwargs": {"task": TASK, "language": LANGUAGE},
            },
            "C1": {
                "purpose": CONFIG_PURPOSES["C1"],
                "generate_kwargs_overrides": {"max_new_tokens": 160},
            },
            "C2": {
                "purpose": CONFIG_PURPOSES["C2"],
                "generate_kwargs_overrides": {"no_repeat_ngram_size": 4},
            },
        },
        "estimator_constants": {
            "model_load_sec": MODEL_LOAD_SEC,
            "container_overhead_sec": CONTAINER_OVERHEAD_SEC,
            "c1_max_per_clip_sec": C1_MAX_PER_CLIP_SEC,
            "c2_min_per_clip_sec": C2_MIN_PER_CLIP_SEC,
        },
        "selection_rules": dict(SELECTION_RULES),
        "provenance": provenance,
        "clips": [
            {
                "clip_id": clip.clip_id,
                "category": clip.category,
                "baseline_runtime_sec": clip.baseline_runtime_sec,
                "insertions": clip.insertions,
                "has_word_loop": clip.has_word_loop,
                "has_char_loop": clip.has_char_loop,
                "heuristic_flagged": clip.heuristic_flag,
                "duration_sec": clip.duration_sec,
                "audio_sha256": clip.audio_sha256,
            }
            for clip in selected
        ],
        "estimate": estimate,
    }


def build_private_plan(
    plan: Dict[str, Any], private_sources: Dict[str, str]
) -> Dict[str, Any]:
    """Add the private-only source-pointer block to a plan copy.

    Args:
        plan: Committed plan dictionary.
        private_sources: Mapping of source label to private filesystem path.

    Returns:
        Deep copy of the plan with an extra ``private_sources`` key. This
        copy is written only to the private freeze directory.
    """
    private_plan = copy.deepcopy(plan)
    private_plan["private_sources"] = dict(private_sources)
    return private_plan


def _walk_strings(value: Any) -> List[str]:
    """Collect every string inside a nested structure.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        List of strings found anywhere in the structure.
    """
    found: List[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.append(key)
            found.extend(_walk_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_walk_strings(item))
    return found


def assert_committed_plan_safe(plan: Dict[str, Any]) -> None:
    """Refuse to write a committed plan that leaks private content.

    Checks the allowed key sets, forbids Devanagari (transcript script)
    characters, private filesystem path signatures, control characters/
    backslashes, and the reference-transcript column name anywhere in the
    committed payload.

    Args:
        plan: Committed plan dictionary.

    Raises:
        ValueError: If any unsafe string or key is present.
    """
    top_keys = set(plan)
    disallowed_top = top_keys - ALLOWED_TOP_LEVEL_KEYS
    if disallowed_top:
        raise ValueError(f"committed plan has unexpected keys: {sorted(disallowed_top)}")
    forbidden_top = set(ALLOWED_TOP_LEVEL_KEYS) - top_keys
    if forbidden_top:
        raise ValueError(f"committed plan is missing keys: {sorted(forbidden_top)}")
    for index, clip in enumerate(plan.get("clips", [])):
        clip_keys = set(clip)
        if clip_keys != ALLOWED_CLIP_KEYS:
            raise ValueError(
                f"clip {index} keys {sorted(clip_keys)} != allowed "
                f"{sorted(ALLOWED_CLIP_KEYS)}"
            )
    for text in _walk_strings(plan):
        if DEVANAGARI_PATTERN.search(text):
            raise ValueError("committed plan contains Devanagari (transcript) text")
        for signature in FORBIDDEN_COMMITTED_SUBSTRINGS:
            if signature in text:
                raise ValueError(
                    f"committed plan contains forbidden substring {signature!r}"
                )


def serialize_plan(plan: Dict[str, Any]) -> str:
    """Serialize a plan dictionary deterministically.

    Args:
        plan: Plan dictionary.

    Returns:
        Deterministic JSON text with a trailing newline.
    """
    return json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def write_freeze(
    plan: Dict[str, Any],
    private_plan: Dict[str, Any],
    plan_out: Path,
    private_dir: Path,
    report_path: Optional[Path],
    report: Dict[str, Any],
    force: bool,
) -> List[Path]:
    """Write the committed plan, the private freeze copy, and the sums file.

    Args:
        plan: Committed plan dictionary (safety-checked before writing).
        private_plan: Private copy including ``private_sources``.
        plan_out: Committed plan output path.
        private_dir: Private freeze directory.
        report_path: Optional JSON report path (written when provided).
        report: Report payload for ``report_path``.
        force: Whether overwriting a previous freeze is allowed.

    Returns:
        List of every written file path.

    Raises:
        InputError: If a target exists and ``force`` is false.
        ValueError: If the committed plan fails the safety scan.
    """
    assert_committed_plan_safe(plan)
    private_plan_path = private_dir / "decoder_probe_plan.json"
    sums_path = private_dir / "SHA256SUMS.txt"
    targets = [plan_out, private_plan_path, sums_path]
    if report_path is not None:
        targets.append(report_path)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise InputError(
            "refusing to overwrite an existing freeze without --force: "
            + ", ".join(str(path) for path in existing)
        )

    private_dir.mkdir(parents=True, exist_ok=True)
    plan_out.parent.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    plan_out.write_text(serialize_plan(plan), encoding="utf-8")
    written.append(plan_out)
    private_plan_path.write_text(serialize_plan(private_plan), encoding="utf-8")
    written.append(private_plan_path)
    if report_path is not None:
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written.append(report_path)

    lines = []
    for path in written:
        # Absolute paths: the committed plan and the private freeze copy can
        # share a base name but must never hash-ambiguate.
        lines.append(f"{sha256_file(path)}  {path}")
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(sums_path)
    return written


def verify_freeze_selfcheck(plan_out: Path, private_plan_path: Path, sums_path: Path) -> None:
    """Verify that the committed plan matches the private freeze copy.

    The private copy may carry extra private-only fields (documented in
    ``PRIVATE_ONLY_PLAN_FIELDS``); after removing those, both payloads must
    be identical dictionaries, and the sums file must match every written
    file byte for byte.

    Args:
        plan_out: Committed plan path.
        private_plan_path: Private freeze plan path.
        sums_path: Private ``SHA256SUMS.txt`` path.

    Raises:
        SelfCheckError: If any comparison fails.
    """
    committed = json.loads(plan_out.read_text(encoding="utf-8"))
    private = json.loads(private_plan_path.read_text(encoding="utf-8"))
    for field in PRIVATE_ONLY_PLAN_FIELDS:
        private.pop(field, None)
    if committed != private:
        raise SelfCheckError(
            "post-freeze self-check failed: committed plan and private freeze "
            "content diverge beyond the private-only fields"
        )
    extra_private = set(private) - set(committed)
    if extra_private:
        raise SelfCheckError(
            f"private freeze copy introduced unexpected top-level keys: {sorted(extra_private)}"
        )
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        expected_sha, raw_path = line.split("  ", 1)
        actual_path = Path(raw_path)
        if not actual_path.is_file():
            raise SelfCheckError(f"SHA256SUMS references a missing file: {actual_path}")
        actual_sha = sha256_file(actual_path)
        if actual_sha != expected_sha:
            raise SelfCheckError(
                f"SHA256SUMS mismatch for {actual_path}: {actual_sha} != {expected_sha}"
            )


def _now_utc() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``.

    Returns:
        Timestamp string.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_repo_relative(path_value: str) -> Path:
    """Resolve a CLI path argument, anchoring relative paths at the repo root.

    Args:
        path_value: Raw path argument.

    Returns:
        Absolute path (repo-root anchored for relative arguments).
    """
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _print_derivation(
    selected: Sequence[SelectedClip],
    counts: Dict[str, int],
    estimate: Dict[str, Any],
    agreement: int,
    population: int,
) -> None:
    """Print the derived selection and all estimate numbers.

    Args:
        selected: Frozen selection.
        counts: Full category counts for the whole cohort.
        estimate: Estimate block.
        agreement: Number of clips whose R1 signals agree.
        population: Number of development-cohort clips checked.
    """
    print(f"J1 decoder probe plan builder ({PROBE_VERSION})")
    print(
        f"R1 ceiling agreement: {agreement}/{population} clips agree "
        "(runtime window vs generation_cap_with_repetition class)"
    )
    print(
        "R2 categories (whole cohort): "
        + ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    )
    print(
        f"R3/R4 mandate: category counts "
        + ", ".join(
            f"{name}={count}" for name, count in sorted(MANDATED_CATEGORY_COUNTS.items())
        )
        + f" (total {MANDATED_TOTAL_CLIPS})"
    )
    for index, clip in enumerate(selected, start=1):
        print(
            f"  {index:2d}. {clip.clip_id}  category={clip.category}  "
            f"insertions={clip.insertions}  "
            f"baseline_runtime_sec={clip.baseline_runtime_sec}  "
            f"duration_sec={clip.duration_sec}  "
            f"audio_sha256={clip.audio_sha256}"
        )
    print(
        "estimator constants: "
        f"model_load_sec={MODEL_LOAD_SEC} "
        f"container_overhead_sec={CONTAINER_OVERHEAD_SEC} "
        f"c1<=min(measured,{C1_MAX_PER_CLIP_SEC}) "
        f"c2>=max(measured,{C2_MIN_PER_CLIP_SEC})"
    )
    per_config = ", ".join(
        f"{name}={estimate['config_inference_seconds'][name]}" for name in CONFIG_ORDER
    )
    print(f"config inference seconds: {per_config}")
    print(
        f"wall = {estimate['decoded_wall_sec']} + {MODEL_LOAD_SEC} + "
        f"{CONTAINER_OVERHEAD_SEC} = {estimate['total_wall_sec']} s "
        f"(gate <= {MAX_TOTAL_WALL_SEC:g}) "
        + ("OK" if estimate["total_wall_sec"] <= MAX_TOTAL_WALL_SEC else "FAILED")
    )
    print(
        f"estimated_gpu_cost = {estimate['total_wall_sec']} x {RATE_USD_PER_SEC} = "
        f"${estimate['estimated_gpu_cost_usd']} "
        f"(gate <= ${MAX_ESTIMATED_COST_USD:.2f}) "
        + (
            "OK"
            if estimate["estimated_gpu_cost_usd"] <= MAX_ESTIMATED_COST_USD
            else "FAILED"
        )
    )
    print(
        f"timeout_worst_case = {FUNCTION_TIMEOUT_SEC} x {RATE_USD_PER_SEC} = "
        f"${estimate['timeout_worst_case_usd']}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the plan builder.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 success, 2 gate failure, 3 mandate
        contradiction, 4 input error, 5 self-check failure).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the Phase 7.1 J1 decoder-probe plan offline. Dry run is "
            "the default; --freeze performs the writes."
        ),
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="Derive and print only (default behavior).",
    )
    modes.add_argument(
        "--freeze",
        action="store_true",
        help="Write the committed plan, private freeze copy, and sums file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing freeze.",
    )
    parser.add_argument(
        "--intelligence-json",
        type=str,
        default=str(DEFAULT_INTELLIGENCE_JSON),
        help="Committed failure intelligence JSON.",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=str(DEFAULT_AUDIO_DIR),
        help="Private directory with the development-cohort WAV files.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help="Frozen validation manifest CSV (private archive).",
    )
    parser.add_argument(
        "--reserve-json",
        type=str,
        default=str(DEFAULT_RESERVE_JSON),
        help="Frozen holdout reserve pool JSON (private archive).",
    )
    parser.add_argument(
        "--raw-output-json",
        type=str,
        default=str(DEFAULT_RAW_OUTPUT_JSON),
        help="Archived baseline raw Modal output (per-clip runtimes).",
    )
    parser.add_argument(
        "--private-dir",
        type=str,
        default=str(DEFAULT_PRIVATE_DIR),
        help="Private freeze directory.",
    )
    parser.add_argument(
        "--plan-out",
        type=str,
        default=str(DEFAULT_PLAN_OUT),
        help="Committed plan output path.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=str(DEFAULT_REPORT_OUT),
        help="JSON report path (written on --freeze only).",
    )
    parser.add_argument(
        "--created-utc",
        type=str,
        default=None,
        help="Pin the plan created_utc timestamp (for byte-determinism tests).",
    )
    args = parser.parse_args(argv)
    try:
        created_utc = args.created_utc or _now_utc()
        datetime.strptime(created_utc, "%Y-%m-%dT%H:%M:%SZ")
        intelligence_path = _resolve_repo_relative(args.intelligence_json)
        audio_dir = _resolve_repo_relative(args.audio_dir)
        manifest_path = _resolve_repo_relative(args.manifest)
        reserve_path = _resolve_repo_relative(args.reserve_json)
        raw_output_path = _resolve_repo_relative(args.raw_output_json)
        private_dir = _resolve_repo_relative(args.private_dir)
        plan_out = _resolve_repo_relative(args.plan_out)
        report_path = _resolve_repo_relative(args.report_out)

        intel, intelligence_sha = load_intelligence(intelligence_path)
        manifest, manifest_sha = load_manifest(manifest_path)
        raw, raw_sha = parse_raw_output(raw_output_path)
        reserve_speakers, reserve_sha = load_reserve_speakers(reserve_path)

        features, counts = derive_categories(intel, raw, manifest)
        selected = build_selection(features, counts)
        validate_membership(selected, manifest, reserve_speakers)
        selected = attach_audio_hashes(selected, audio_dir)
        estimate = build_estimate(selected)

        population = len(manifest)
        provenance = {
            "manifest_sha256": manifest_sha,
            "reserve_sha256": reserve_sha,
            "intelligence_json_sha256": intelligence_sha,
            "raw_output_sha256": raw_sha,
            "archive_note": "private mucs-validation-t4-run-2026-10-16",
            "baseline_runtime_and_duration_source": (
                "archived per-clip runtime_sec and audio_duration_sec from the "
                "T4 baseline raw Modal output (raw_modal_output.json)"
            ),
            "selection_features_source": (
                "committed per-clip failure intelligence records: insertions, "
                "has_word_loop, has_char_loop, heuristic_flag, failure_class"
            ),
            "runtime_class_window_agreement": {
                "clips_checked": population,
                "disagreements": 0,
            },
        }

        plan = build_plan(selected, created_utc, provenance, estimate)
        assert_committed_plan_safe(plan)

        if not args.freeze:
            _print_derivation(selected, counts, estimate, population, population)
            print("mode: dry-run (nothing written)")
            return 0

        report = {
            "plan_builder_version": PROBE_VERSION,
            "mode": "freeze",
            "created_utc": created_utc,
            "probe_version": PROBE_VERSION,
            "gpu": GPU,
            "rate_usd_per_sec": RATE_USD_PER_SEC,
            "function_timeout_sec": FUNCTION_TIMEOUT_SEC,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "precision": PRECISION,
            "language": LANGUAGE,
            "task": TASK,
            "mandate": {
                "category_counts": dict(MANDATED_CATEGORY_COUNTS),
                "total_clips": MANDATED_TOTAL_CLIPS,
                "runtime_class_window_agreement": provenance[
                    "runtime_class_window_agreement"
                ],
                "selection_rules": dict(SELECTION_RULES),
            },
            "cohort_categories": dict(sorted(counts.items())),
            "clips": [asdict(clip) for clip in selected],
            "estimate": copy.deepcopy(estimate),
            "sources": {
                "manifest_sha256": manifest_sha,
                "reserve_sha256": reserve_sha,
                "intelligence_json_sha256": intelligence_sha,
                "raw_output_sha256": raw_sha,
                "archive_note": "private mucs-validation-t4-run-2026-10-16",
            },
            "outputs": {
                "plan_out": str(plan_out),
                "private_dir": str(private_dir),
                "report_out": str(report_path),
                "force": bool(args.force),
            },
        }
        private_plan = build_private_plan(
            plan,
            {
                "audio_dir": str(audio_dir),
                "manifest": str(manifest_path),
                "reserve_json": str(reserve_path),
                "raw_output_json": str(raw_output_path),
                "intelligence_json": str(intelligence_path),
                "frozen_with": "scripts/plan_hinglish_decoder_probe.py",
            },
        )
        written = write_freeze(
            plan=plan,
            private_plan=private_plan,
            plan_out=plan_out,
            private_dir=private_dir,
            report_path=report_path,
            report=report,
            force=args.force,
        )
        plan_out_result = next(path for path in written if path == plan_out)
        private_plan_path = private_dir / "decoder_probe_plan.json"
        sums_path = private_dir / "SHA256SUMS.txt"
        verify_freeze_selfcheck(plan_out_result, private_plan_path, sums_path)
        for path in written:
            print(f"wrote: {path} (sha256 {sha256_file(path)})")
        _print_derivation(selected, counts, estimate, population, population)
        print("self-check: committed plan matches private freeze copy OK")
        print(
            "FROZEN BEFORE EXECUTION: "
            f"{plan_out} is the execution reference for infra/"
            "modal_hinglish_decoder_probe.py; do not modify it without --force"
        )
        return 0
    except GateError as exc:
        print(f"GATE FAILED: {exc}", file=sys.stderr)
        return 2
    except MandateError as exc:
        print(f"MANDATE CONTRADICTION: {exc}", file=sys.stderr)
        print("Stopping: never improvise a substitute selection.", file=sys.stderr)
        return 3
    except InputError as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 4
    except SelfCheckError as exc:
        print(f"SELF-CHECK FAILED: {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())

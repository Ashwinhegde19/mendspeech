#!/usr/bin/env python3
"""Audit the Phase 7.1 controlled decoder probe (J1) 16-clip selection.

This tool enforces the pre-registered selection rules for the single paid
T4 probe run (plan ``mendspeech.phase7_1_probe_selection.v1``) before that
run is allowed to happen. It never runs inference, never downloads
anything, and never writes transcript or reference text anywhere: its
evidence inputs are clip IDs, numbers, booleans, flag names, and SHA-256
hashes only.

Pre-registered rules enforced here (blockers, not tunable policy)
-----------------------------------------------------------------
R1  Ceiling membership per clip = baseline inference runtime inside the
    window [3.687, 3.734] seconds (inclusive edges, small float epsilon)
    OR committed failure class ``generation_cap_with_repetition``. Both
    definitions must agree for every clip of the 30-clip development
    cohort, else the audit fails.
R2  Categories:
      ceiling_word_loop = ceiling AND word-level loop,
      ceiling_char_loop = ceiling AND char-level loop AND NOT word loop,
      early_loop        = NOT ceiling AND (word OR char loop evidence),
      control_candidate = NOT ceiling, no word loop, no char loop,
                          and not heuristic-flagged.
    Anything else is ``unclassified`` and is an audit violation.
R3  Frozen 16-clip probe set =
      top-8 ``ceiling_word_loop`` by insertion count descending with
        clip_id ascending as the deterministic tie-break,
      + ALL ``ceiling_char_loop`` (must be exactly 3 clips),
      + ALL ``early_loop`` clips (must resolve to exactly the two
        mandated early-stopping clips with prefixes ``304558_`` and
        ``399287_``; any other derivation is a STOP violation),
      + the 3 named normal controls resolved from prefixes ``111627_``,
        ``136706_``, ``415128_``; each must satisfy the R2 control
        criteria (it must actually be a ``control_candidate`` clip).
R4  All 16 clip IDs must belong to the frozen development-cohort
    manifest. Insertion counts and runtimes come from committed per-clip
    artifacts (``results/hinglish/failure_intelligence.json`` per-clip
    ``insertions`` and ``results/hinglish/validation_baseline_metrics.json``
    per-clip ``inference_runtime_sec``), cross-checked against the offline
    inspection report when it is provided.
R5  No selected clip's speaker may appear in the 47-speaker holdout
    reserve (``holdout_pool_v0.json``). No step of this audit reads
    reference-text semantics: the frozen manifest is read for clip IDs
    and speaker IDs only (its reference columns are parsed but never
    extracted into output), loop evidence is read as boolean flags from
    committed artifacts, and the report is structurally scanned so that
    only IDs, numbers, booleans, flag names, and hashes can appear in
    string positions.

Runtime edge tolerance (documented pre-registered choice)
---------------------------------------------------------
Window edges are inclusive. To be robust against binary floating-point
representation of the recorded 3-decimal runtimes, each edge is extended
by an absolute epsilon of 1e-6 seconds only. Therefore:
  3.687 / 3.734      exact edges            -> ceiling,
  3.7339             inside the window      -> ceiling,
  3.7341             0.0001 outside         -> NOT ceiling,
  3.6869999          float noise below edge -> ceiling,
  3.6869             1e-4 below edge        -> NOT ceiling.
Substantive decimals are decision boundaries; only representation noise
is absorbed.

Modes
-----
Derive mode (default): derive the 16 candidates from the committed
artifacts, verify R4/R5, embed audio SHA-256 hashes computed from the
private audio directory, write the audit report JSON (structurally
scanned), print a compact human summary, and exit 0 on pass / 2 on any
violation (exit 1 is reserved for usage errors such as missing files).

Plan-audit mode (``--plan-json``): recompute the expected selection with
the same code path and assert the built plan matches exactly: clip IDs,
categories, per-clip evidence numbers, and audio SHA-256 values when the
plan provides them (absent hashes are recorded, not treated as a
mismatch).

Expected plan JSON schema (documented contract for the plan-builder):

    {
      "plan_schema": "mendspeech.phase7_1_probe_selection.v1",
      "clips": [
        {
          "clip_id": "111627_bYPRSx2sjXooUdv8_0000",
          "category": "control_candidate",
          "runtime_sec": 1.562,
          "insertions": 0,
          "heuristic_flag": false,
          "has_word_loop": false,
          "has_char_loop": false,
          "audio_sha256": "<64 hex chars>"     // optional but recommended
        }
      ]
    }

Extra top-level keys are permitted; every listed clip must match the
recomputed expectation exactly.

Audit report schema: ``mendspeech.phase7_1_probe_selection_audit.v1``.
Checks C01..C14 each carry a status (pass / skip / fail) and structured
detail (IDs, counts, booleans) — never free text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Pre-registered constants (do not tune without re-registration)
# ---------------------------------------------------------------------------

RUNTIME_WINDOW_SEC: Tuple[float, float] = (3.687, 3.734)
RUNTIME_EDGE_EPSILON_SEC: float = 1e-6
CAP_FAILURE_CLASSES: Tuple[str, ...] = ("generation_cap_with_repetition",)

CATEGORY_CEILING_WORD_LOOP = "ceiling_word_loop"
CATEGORY_CEILING_CHAR_LOOP = "ceiling_char_loop"
CATEGORY_EARLY_LOOP = "early_loop"
CATEGORY_CONTROL = "control_candidate"
CATEGORY_UNCLASSIFIED = "unclassified"
CATEGORIES: Tuple[str, ...] = (
    CATEGORY_CEILING_WORD_LOOP,
    CATEGORY_CEILING_CHAR_LOOP,
    CATEGORY_EARLY_LOOP,
    CATEGORY_CONTROL,
)

EXPECTED_SET_SIZE = 16
EXPECTED_CEILING_CHAR_LOOP_COUNT = 3
EXPECTED_EARLY_LOOP_COUNT = 2
WORD_LOOP_TOP_N = 8

# Mandated early-stopping members (R3): resolved to full clip IDs from data.
MANDATED_EARLY_CLIP_PREFIXES: Tuple[str, ...] = ("304558_", "399287_")
# Named normal controls (R3), as proposed in the Phase 7.0 subagent report.
CONTROL_CLIP_PREFIXES: Tuple[str, ...] = ("111627_", "136706_", "415128_")

# Frozen input hashes pinned at registration time (full sha256).
FROZEN_MANIFEST_SHA256 = (
    "2669cb56ddf2a9c12602b9fcbe406a38724a8e3d1d65be0eeda228ad78955d67"
)
FROZEN_RESERVE_SHA256 = (
    "5dc051dab2e55da6586614676d0b6bfbcb3e7a809b8f28d737e78c399d8b4ff5"
)

DEFAULT_INTELLIGENCE_JSON = REPO_ROOT / "results/hinglish/failure_intelligence.json"
DEFAULT_BASELINE_JSON = REPO_ROOT / "results/hinglish/validation_baseline_metrics.json"
DEFAULT_MANIFEST = Path(
    "/home/ashwin/backups/mucs-validation/frozen_validation_manifest.csv"
)
DEFAULT_AUDIO_DIR = Path("/home/ashwin/backups/mucs-validation/audio")
DEFAULT_RESERVE_JSON = Path("/home/ashwin/backups/mucs-validation/holdout_pool_v0.json")
DEFAULT_REPORT_OUT = Path("/tmp/opencode/phase7.1/audit_report.json")

AUDIT_SCHEMA = "mendspeech.phase7_1_probe_selection_audit.v1"
PLAN_SCHEMA = "mendspeech.phase7_1_probe_selection.v1"

EXIT_OK = 0
EXIT_VIOLATION = 2
EXIT_USAGE = 1


class DataError(RuntimeError):
    """Structured input could not be loaded or lacks required fields."""


class DerivationStop(RuntimeError):
    """A pre-registered derivation mandate failed; the probe set is not derived.

    Attributes:
        violations: structured, token-valued violation records.
    """

    def __init__(self, violations: List[Dict[str, Any]]):
        super().__init__(violations)
        self.violations = violations


# ---------------------------------------------------------------------------
# Loading (IDs, numbers, booleans only; reference columns are NEVER extracted)
# ---------------------------------------------------------------------------


def _required(record: Mapping[str, Any], field: str, source: str, clip_id: str) -> Any:
    if field not in record:
        raise DataError(
            f"{source}: clip {clip_id or '<unknown>'} lacks field {field}"
        )
    return record[field]


def load_intelligence(path: Path) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[str, Any]]:
    """Load per-clip failure-intelligence records (committed artifact).

    Returns (clip_id -> evidence fields, file sha256, provenance block).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    per_clip = raw.get("per_clip")
    if not isinstance(per_clip, list) or not per_clip:
        raise DataError(f"{path.name}: missing non-empty 'per_clip' list")
    clips: Dict[str, Dict[str, Any]] = {}
    for record in per_clip:
        clip_id = str(_required(record, "clip_id", path.name, ""))
        clips[clip_id] = {
            "clip_id": clip_id,
            "speaker_id": str(_required(record, "speaker_id", path.name, clip_id)),
            "failure_class": str(_required(record, "failure_class", path.name, clip_id)),
            "confidence": str(record.get("confidence", "")),
            "insertions": int(_required(record, "insertions", path.name, clip_id)),
            "cap_hit": bool(_required(record, "cap_hit", path.name, clip_id)),
            "heuristic_flag": bool(_required(record, "heuristic_flag", path.name, clip_id)),
            "has_word_loop": bool(_required(record, "has_word_loop", path.name, clip_id)),
            "has_char_loop": bool(_required(record, "has_char_loop", path.name, clip_id)),
            "loop_coverage_words": int(record.get("loop_coverage_words", 0)),
        }
    provenance = raw.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    return clips, file_sha256(path), provenance


def load_baseline_runtimes(path: Path) -> Tuple[Dict[str, float], str]:
    """Load committed per-clip inference runtimes (seconds)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    per_clip = raw.get("per_clip")
    if not isinstance(per_clip, list) or not per_clip:
        raise DataError(f"{path.name}: missing non-empty 'per_clip' list")
    runtimes: Dict[str, float] = {}
    for record in per_clip:
        clip_id = str(_required(record, "clip_id", path.name, ""))
        runtimes[clip_id] = float(
            _required(record, "inference_runtime_sec", path.name, clip_id)
        )
    return runtimes, file_sha256(path)


def load_inspection(path: Path) -> Dict[str, Any]:
    """Load the offline inspection report (regenerable, read-only)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    per_clip = raw.get("per_clip")
    ceiling = raw.get("runtime_ceiling", {})
    if not isinstance(per_clip, list) or not isinstance(ceiling, dict):
        raise DataError(f"{path.name}: missing 'per_clip' list or 'runtime_ceiling'")
    clips: Dict[str, Dict[str, Any]] = {}
    for record in per_clip:
        clip_id = str(_required(record, "clip_id", path.name, ""))
        clips[clip_id] = {
            "runtime_sec": float(_required(record, "runtime_sec", path.name, clip_id)),
            "flagged": bool(_required(record, "flagged", path.name, clip_id)),
            "has_word_loop": bool(_required(record, "has_word_loop", path.name, clip_id)),
            "has_char_loop": bool(_required(record, "has_char_loop", path.name, clip_id)),
            "diagnostic_insertions": int(
                _required(record, "diagnostic_insertions", path.name, clip_id)
            ),
        }
    return {
        "clips": clips,
        "ceiling_members": [str(member) for member in ceiling.get("members", [])],
        "ceiling_min_sec": ceiling.get("min_sec"),
        "ceiling_max_sec": ceiling.get("max_sec"),
        "sha256": file_sha256(path),
    }


def load_manifest(path: Path, sha256: Optional[str] = None) -> Tuple[List[str], Dict[str, str], str]:
    """Read clip IDs and speaker IDs from the frozen manifest.

    Only ``clip_id`` and ``speaker_id`` columns are extracted. The manifest
    contains reference-transcript columns; the CSV reader parses them into
    per-row dicts that are immediately discarded and never copied into any
    output structure (the report is structurally scanned as a backstop).
    """
    clip_ids: List[str] = []
    speakers: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "clip_id" not in fieldnames or "speaker_id" not in fieldnames:
            raise DataError(f"{path.name}: manifest lacks clip_id/speaker_id columns")
        for row in reader:
            clip_id = (row.get("clip_id") or "").strip()
            if not clip_id:
                raise DataError(f"{path.name}: empty clip_id row")
            clip_ids.append(clip_id)
            speakers[clip_id] = (row.get("speaker_id") or "").strip()
    digest = sha256 if sha256 is not None else file_sha256(path)
    return clip_ids, speakers, digest


def load_reserve(path: Path, sha256: Optional[str] = None) -> Tuple[List[str], str]:
    """Load holdout-reserve speaker IDs (never used for any selection choice)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    speakers_raw = raw.get("speakers")
    if not isinstance(speakers_raw, list):
        raise DataError(f"{path.name}: missing 'speakers' list")
    speakers: List[str] = []
    for entry in speakers_raw:
        if isinstance(entry, dict):
            speaker_id = str(entry.get("speaker_id", "")).strip()
        else:
            speaker_id = str(entry).strip()
        if speaker_id:
            speakers.append(speaker_id)
    digest = sha256 if sha256 is not None else file_sha256(path)
    return speakers, digest


def file_sha256(path: Path) -> str:
    """SHA-256 of a file streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Evidence merging and R1/R2 classification
# ---------------------------------------------------------------------------


def build_evidence(
    intelligence: Mapping[str, Mapping[str, Any]],
    runtimes: Mapping[str, float],
) -> Dict[str, Dict[str, Any]]:
    """Merge committed per-clip intelligence with committed runtimes."""
    missing_runtime = sorted(clip for clip in intelligence if clip not in runtimes)
    if missing_runtime:
        raise DataError(
            "no committed runtime for clips: " + ", ".join(missing_runtime[:5])
        )
    merged: Dict[str, Dict[str, Any]] = {}
    for clip_id, record in intelligence.items():
        merged[clip_id] = dict(record)
        merged[clip_id]["runtime_sec"] = float(runtimes[clip_id])
    return merged


def in_runtime_window(
    runtime_sec: Optional[float],
    window: Tuple[float, float] = RUNTIME_WINDOW_SEC,
    epsilon: float = RUNTIME_EDGE_EPSILON_SEC,
) -> bool:
    """R1 runtime-window membership: inclusive edges plus float epsilon only.

    Documented choice: epsilon is 1e-6 s absolute. Substantive decimals
    such as 3.7341 (0.0001 outside the edge) are NOT absorbed.
    """
    if runtime_sec is None:
        return False
    low, high = window
    return (low - epsilon) <= float(runtime_sec) <= (high + epsilon)


def ceiling_membership(
    evidence: Mapping[str, Any],
    runtime_sec: Optional[float],
    window: Tuple[float, float] = RUNTIME_WINDOW_SEC,
    epsilon: float = RUNTIME_EDGE_EPSILON_SEC,
) -> Tuple[bool, bool]:
    """Return (runtime-definition flag, failure-class-definition flag).

    R1 requires the two definitions to agree for every clip; disagreement
    surfaces as check C02.
    """
    runtime_flag = in_runtime_window(runtime_sec, window, epsilon)
    class_flag = (
        evidence["failure_class"] in CAP_FAILURE_CLASSES
        or bool(evidence.get("cap_hit"))
    )
    return runtime_flag, class_flag


def assign_category(evidence: Mapping[str, Any], is_ceiling: bool) -> str:
    """R2 category rules; anything unmatched is ``unclassified``."""
    has_word_loop = bool(evidence.get("has_word_loop"))
    has_char_loop = bool(evidence.get("has_char_loop"))
    flagged = bool(evidence.get("heuristic_flag"))
    if is_ceiling and has_word_loop:
        return CATEGORY_CEILING_WORD_LOOP
    if is_ceiling and has_char_loop and not has_word_loop:
        return CATEGORY_CEILING_CHAR_LOOP
    if not is_ceiling and (has_word_loop or has_char_loop):
        return CATEGORY_EARLY_LOOP
    if not is_ceiling and not has_word_loop and not has_char_loop and not flagged:
        return CATEGORY_CONTROL
    return CATEGORY_UNCLASSIFIED


def derive_ceiling_set(
    evidence: Mapping[str, Mapping[str, Any]],
    window: Tuple[float, float] = RUNTIME_WINDOW_SEC,
    epsilon: float = RUNTIME_EDGE_EPSILON_SEC,
) -> Dict[str, Any]:
    """R1 ceiling derivation plus the category partition per R2."""
    runtime_members: List[str] = []
    class_members: List[str] = []
    disagreements: List[str] = []
    categories: Dict[str, str] = {}
    for clip_id in sorted(evidence):
        runtime_flag, class_flag = ceiling_membership(
            evidence[clip_id], evidence[clip_id]["runtime_sec"], window, epsilon
        )
        if runtime_flag:
            runtime_members.append(clip_id)
        if class_flag:
            class_members.append(clip_id)
        if runtime_flag != class_flag:
            disagreements.append(clip_id)
        categories[clip_id] = assign_category(
            evidence[clip_id], runtime_flag and class_flag
        )
    member_runtimes = [evidence[clip]["runtime_sec"] for clip in runtime_members]
    non_member_runtimes = [
        evidence[clip]["runtime_sec"]
        for clip in sorted(evidence)
        if clip not in set(runtime_members)
    ]
    return {
        "members": sorted(runtime_members),
        "class_definition_members": sorted(class_members),
        "disagreeing_clip_ids": sorted(disagreements),
        "member_min_runtime_sec": min(member_runtimes) if member_runtimes else None,
        "member_max_runtime_sec": max(member_runtimes) if member_runtimes else None,
        "next_highest_non_member_runtime_sec": (
            max(non_member_runtimes) if non_member_runtimes else None
        ),
        "categories": categories,
    }


def _resolve_prefix(prefix: str, candidate_ids: Sequence[str]) -> Optional[str]:
    matches = sorted(clip for clip in candidate_ids if clip.startswith(prefix))
    if len(matches) != 1:
        return None
    return matches[0]


def derive_selection(
    evidence: Mapping[str, Mapping[str, Any]],
    window: Tuple[float, float] = RUNTIME_WINDOW_SEC,
    epsilon: float = RUNTIME_EDGE_EPSILON_SEC,
) -> Dict[str, Any]:
    """Derive the 16-clip set per R1-R3.

    Raises:
        DerivationStop: when any pre-registered mandate (R2 partition,
            char-loop count, early-loop mandate, named controls, top-8
            availability) fails. All violations found are reported at once
            and the probe set is NOT derived.
    """
    ceiling = derive_ceiling_set(evidence, window, epsilon)
    categories = ceiling["categories"]
    violations: List[Dict[str, Any]] = []

    unclassified = sorted(
        clip for clip, category in categories.items() if category == CATEGORY_UNCLASSIFIED
    )
    if unclassified:
        violations.append({"kind": "unclassified_clips", "clip_ids": unclassified})
    if ceiling["disagreeing_clip_ids"]:
        violations.append(
            {
                "kind": "ceiling_definition_disagreement",
                "clip_ids": ceiling["disagreeing_clip_ids"],
            }
        )

    word_loops = sorted(
        (
            clip
            for clip, category in categories.items()
            if category == CATEGORY_CEILING_WORD_LOOP
        ),
        key=lambda clip: (-int(evidence[clip]["insertions"]), clip),
    )
    char_loops = sorted(
        clip for clip, category in categories.items() if category == CATEGORY_CEILING_CHAR_LOOP
    )
    early_loops = sorted(
        clip for clip, category in categories.items() if category == CATEGORY_EARLY_LOOP
    )
    controls = sorted(
        clip for clip, category in categories.items() if category == CATEGORY_CONTROL
    )

    # Insertion-count ties in the word-loop ranking (tie-break: clip_id asc).
    tie_groups: Dict[int, List[str]] = {}
    for clip in word_loops:
        tie_groups.setdefault(int(evidence[clip]["insertions"]), []).append(clip)
    ties = {
        str(count): sorted(clips)
        for count, clips in sorted(tie_groups.items(), reverse=True)
        if len(clips) > 1
    }

    if len(char_loops) != EXPECTED_CEILING_CHAR_LOOP_COUNT:
        violations.append(
            {
                "kind": "char_loop_count_mismatch",
                "found": len(char_loops),
                "expected": EXPECTED_CEILING_CHAR_LOOP_COUNT,
            }
        )
    if len(early_loops) != EXPECTED_EARLY_LOOP_COUNT:
        violations.append(
            {
                "kind": "early_loop_count_mismatch",
                "found": len(early_loops),
                "expected": EXPECTED_EARLY_LOOP_COUNT,
            }
        )
    resolved_early: List[Optional[str]] = [
        _resolve_prefix(prefix, early_loops) for prefix in MANDATED_EARLY_CLIP_PREFIXES
    ]
    if any(resolved is None for resolved in resolved_early) or sorted(
        early_loops
    ) != sorted(res for res in resolved_early if res is not None):
        violations.append(
            {
                "kind": "early_loop_mandate_unresolved",
                "mandated_prefixes": list(MANDATED_EARLY_CLIP_PREFIXES),
                "found_early_loop_clip_ids": early_loops,
            }
        )
    if len(word_loops) < WORD_LOOP_TOP_N:
        violations.append(
            {
                "kind": "word_loop_pool_too_small",
                "found": len(word_loops),
                "expected_top_n": WORD_LOOP_TOP_N,
            }
        )
    resolved_controls: List[Optional[str]] = [
        _resolve_prefix(prefix, controls) for prefix in CONTROL_CLIP_PREFIXES
    ]
    if any(resolved is None for resolved in resolved_controls):
        violations.append(
            {
                "kind": "named_control_unresolved",
                "controlled_prefixes": [
                    prefix
                    for prefix, resolved in zip(CONTROL_CLIP_PREFIXES, resolved_controls)
                    if resolved is None
                ],
            }
        )
    if violations:
        raise DerivationStop(violations)

    selected = (
        word_loops[:WORD_LOOP_TOP_N]
        + [res for res in resolved_early if res is not None]
        + [res for res in resolved_controls if res is not None]
        + char_loops
    )
    selected = sorted(selected)
    return {
        "selected": selected,
        "categories": {clip: categories[clip] for clip in selected},
        "ordered_word_loops": word_loops,
        "word_loop_insertion_tie_groups": ties,
        "char_loops": char_loops,
        "early_loops": early_loops,
        "named_controls": sorted(res for res in resolved_controls if res is not None),
        "ceiling": ceiling,
    }


# ---------------------------------------------------------------------------
# Checks, content safety, and the audit report
# ---------------------------------------------------------------------------


def _check(check_id: str, kind: str, **detail: Any) -> Dict[str, Any]:
    return {"id": check_id, "kind": kind, "status": "pass", **detail}


def _fail(check_id: str, kind: str, **detail: Any) -> Dict[str, Any]:
    return {"id": check_id, "kind": kind, "status": "fail", **detail}


def _skip(check_id: str, kind: str, **detail: Any) -> Dict[str, Any]:
    return {"id": check_id, "kind": kind, "status": "skip", **detail}


def compute_audio_hashes(
    selected: Sequence[str], audio_dir: Path
) -> Tuple[Dict[str, str], List[str]]:
    """SHA-256 per selected clip's audio file; report missing files."""
    hashes: Dict[str, str] = {}
    missing: List[str] = []
    for clip_id in selected:
        audio_path = audio_dir / f"{clip_id}.wav"
        if audio_path.is_file():
            hashes[clip_id] = file_sha256(audio_path)
        else:
            missing.append(clip_id)
    return hashes, sorted(missing)


def selection_records(
    selected: Sequence[str],
    categories: Mapping[str, str],
    evidence: Mapping[str, Mapping[str, Any]],
    manifest_speakers: Mapping[str, str],
    reserve_speakers: Sequence[str],
    audio_hashes: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """Per-clip audit records (IDs, numbers, booleans, hashes only)."""
    reserve = set(reserve_speakers)
    records: List[Dict[str, Any]] = []
    for clip_id in selected:
        clip_evidence = evidence[clip_id]
        speaker_id = str(clip_evidence["speaker_id"])
        records.append(
            {
                "clip_id": clip_id,
                "category": categories[clip_id],
                "speaker_id": speaker_id,
                "runtime_sec": float(clip_evidence["runtime_sec"]),
                "insertions": int(clip_evidence["insertions"]),
                "cap_hit": bool(clip_evidence["cap_hit"]),
                "heuristic_flag": bool(clip_evidence["heuristic_flag"]),
                "has_word_loop": bool(clip_evidence["has_word_loop"]),
                "has_char_loop": bool(clip_evidence["has_char_loop"]),
                "loop_coverage_words": int(clip_evidence["loop_coverage_words"]),
                "in_dev_cohort": clip_id in manifest_speakers,
                "speaker_not_in_reserve": speaker_id not in reserve,
                "audio_sha256": audio_hashes.get(clip_id),
            }
        )
    return records


def content_safety_violations(value: Any) -> List[str]:
    """Structural scan: only safe string shapes may appear anywhere.

    Allowed string values: clip IDs (``^\\d{6}_[A-Za-z0-9]+_\\d{4}$``),
    6-digit speaker IDs, category/flag/kind/status/schema tokens, file
    names, and 64-hex hashes. Keys must be plain tokens. Devanagari
    codepoints are always forbidden. Whitespace-containing strings
    (sentences, transcript text) always fail the token whitelist.
    """
    clip_re = re.compile(r"^\d{6}_[A-Za-z0-9]+_\d{4}$")
    speaker_re = re.compile(r"^\d{6}$")
    hash_re = re.compile(r"^[0-9a-f]{64}$")
    token_re = re.compile(r"^[A-Za-z0-9_.\-]+$")
    devanagari = re.compile(r"[\u0900-\u097F]")
    violations: List[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            if devanagari.search(node):
                violations.append(f"{path}:devanagari")
            elif not (
                clip_re.match(node)
                or speaker_re.match(node)
                or hash_re.match(node)
                or token_re.match(node)
            ):
                violations.append(f"{path}:unsafe_string")
        elif isinstance(node, dict):
            for key, child in node.items():
                if not isinstance(key, str) or not token_re.match(key):
                    violations.append(f"{path}:unsafe_key")
                else:
                    walk(child, f"{path}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}[{index}]")
        return

    walk(value, "$")
    return violations


#: Plan schema variants accepted by C13. The committed probe plan (built by
#: ``scripts/plan_hinglish_decoder_probe.py``) names the runtime
#: ``baseline_runtime_sec``, the flag ``heuristic_flagged``, and calls the
#: selected control clips ``control``; earlier drafts used ``runtime_sec``,
#: ``heuristic_flag`` and ``control_candidate``. Values are always compared
#: strictly; only these key/category spellings are aliased.
_SELECTION_CATEGORY_ALIASES = {"control_candidate": "control"}


def _canonical_selection_category(name: Any) -> str:
    """Normalize the selected-clip category spelling for C13 comparisons."""
    return _SELECTION_CATEGORY_ALIASES.get(str(name), str(name))


def _plan_clip_issues(
    plan_clips: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
    audio_hashes: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """Compare a built plan against the recomputed expectation (C13)."""
    issues: List[Dict[str, Any]] = []
    ids = [str(entry.get("clip_id", "")) for entry in plan_clips]
    if sorted(ids) != sorted(expected):
        issues.append(
            {
                "issue": "clip_id_set_mismatch",
                "plan_only_clip_ids": sorted(set(ids) - set(expected)),
                "expected_only_clip_ids": sorted(set(expected) - set(ids)),
            }
        )
    for entry in plan_clips:
        clip_id = str(entry.get("clip_id", ""))
        if clip_id not in expected:
            continue
        want = expected[clip_id]
        got_category = _canonical_selection_category(entry.get("category", ""))
        if got_category != _canonical_selection_category(want["category"]):
            issues.append(
                {
                    "issue": "category_mismatch",
                    "clip_id": clip_id,
                    "plan_category": got_category,
                }
            )
        got_runtime = entry.get("runtime_sec")
        if got_runtime is None:
            got_runtime = entry.get("baseline_runtime_sec")
        if got_runtime is None or (
            abs(float(got_runtime) - float(want["runtime_sec"])) > RUNTIME_EDGE_EPSILON_SEC
        ):
            issues.append({"issue": "runtime_mismatch", "clip_id": clip_id})
        if entry.get("insertions") != want["insertions"]:
            issues.append({"issue": "insertions_mismatch", "clip_id": clip_id})
        for field, alternate in (
            ("heuristic_flag", "heuristic_flagged"),
            ("has_word_loop", None),
            ("has_char_loop", None),
        ):
            got_flag = entry.get(field)
            if got_flag is None and alternate is not None:
                got_flag = entry.get(alternate)
            if bool(got_flag) != bool(want[field]):
                issues.append(
                    {"issue": "flag_mismatch", "clip_id": clip_id, "field": field}
                )
        plan_hash = entry.get("audio_sha256")
        if plan_hash is not None and str(plan_hash) != audio_hashes.get(clip_id):
            issues.append({"issue": "audio_sha256_mismatch", "clip_id": clip_id})
    return issues


def run_audit(
    *,
    evidence: Mapping[str, Mapping[str, Any]],
    evidence_sha256: str,
    evidence_provenance: Mapping[str, Any],
    manifest_ids: Sequence[str],
    manifest_speakers: Mapping[str, str],
    manifest_sha256: str,
    reserve_speakers: Sequence[str],
    reserve_sha256: str,
    audio_dir: Path,
    baseline_sha256: Optional[str] = None,
    inspection: Optional[Mapping[str, Any]] = None,
    expected_manifest_sha256: Optional[str] = None,
    expected_reserve_sha256: Optional[str] = None,
    plan: Optional[Mapping[str, Any]] = None,
    window: Tuple[float, float] = RUNTIME_WINDOW_SEC,
    epsilon: float = RUNTIME_EDGE_EPSILON_SEC,
) -> Tuple[Dict[str, Any], bool]:
    """Execute every pre-registered check and build the audit report.

    Returns ``(report, passed)``. Any failing check makes ``passed`` false.
    Ordered rules on failure: R3 mandates stop the derivation (no probe set
    is emitted); R4/R5 and the remaining checks still report whatever is
    decidable so the report documents all evidence available.
    """
    checks: List[Dict[str, Any]] = []
    mode = "plan_audit" if plan is not None else "derive"
    cohort_ids = sorted(evidence)

    # C01 cohort completeness ---------------------------------------------------
    duplicates = sorted(
        {clip for clip in evidence if list(cohort_ids).count(clip) > 1}
    )
    if cohort_ids and not duplicates:
        checks.append(_check("C01", "cohort_record_completeness", clip_count=len(cohort_ids)))
    else:
        checks.append(
            _fail("C01", "cohort_record_completeness",
                  duplicate_clip_ids=duplicates, clip_count=len(cohort_ids))
        )

    # C02 ceiling definition agreement (R1) --------------------------------------
    ceiling = derive_ceiling_set(evidence, window, epsilon)
    categories = ceiling["categories"]
    edges_match = all(
        edge is not None and abs(edge - want) <= epsilon
        for edge, want in zip(
            (ceiling["member_min_runtime_sec"], ceiling["member_max_runtime_sec"]), window
        )
    )
    if not ceiling["disagreeing_clip_ids"] and edges_match:
        checks.append(
            _check(
                "C02",
                "ceiling_definition_agreement",
                clips_checked=len(cohort_ids),
                ceiling_member_count=len(ceiling["members"]),
                window_min_sec=window[0],
                window_max_sec=window[1],
                member_min_runtime_sec=ceiling["member_min_runtime_sec"],
                member_max_runtime_sec=ceiling["member_max_runtime_sec"],
                next_highest_non_member_runtime_sec=ceiling[
                    "next_highest_non_member_runtime_sec"
                ],
            )
        )
    else:
        checks.append(
            _fail(
                "C02",
                "ceiling_definition_agreement",
                disagreeing_clip_ids=ceiling["disagreeing_clip_ids"],
                member_min_runtime_sec=ceiling["member_min_runtime_sec"],
                member_max_runtime_sec=ceiling["member_max_runtime_sec"],
                window_min_sec=window[0],
                window_max_sec=window[1],
            )
        )

    # R3 derivation --------------------------------------------------------------
    try:
        selection = derive_selection(evidence, window, epsilon)
        selected = selection["selected"]
        derivation_failed = False
        checks.append(
            _check(
                "C03",
                "derivation_mandates",
                early_loop_clip_ids=selection["early_loops"],
                named_control_clip_ids=selection["named_controls"],
                char_loop_clip_ids=selection["char_loops"],
                word_loop_insertion_tie_groups=selection[
                    "word_loop_insertion_tie_groups"
                ],
            )
        )
    except DerivationStop as stop:
        derivation_failed = True
        selected = []
        checks.append(_fail("C03", "derivation_mandates", violations=stop.violations))

    # C04 category partition (R2) --------------------------------------------------
    unclassified_count = sum(
        1 for category in categories.values() if category == CATEGORY_UNCLASSIFIED
    )
    if unclassified_count == 0 and not derivation_failed:
        counts = {
            category: sum(1 for c in categories.values() if c == category)
            for category in CATEGORIES
        }
        checks.append(
            _check("C04", "category_partition", counts=counts, clip_count=len(cohort_ids))
        )
    else:
        checks.append(
            _fail(
                "C04",
                "category_partition",
                counts={
                    category: sum(1 for c in categories.values() if c == category)
                    for category in CATEGORIES
                },
                unclassified_count=unclassified_count,
            )
        )

    records: List[Dict[str, Any]] = []
    audio_hashes: Dict[str, str] = {}

    if not derivation_failed:
        # C05 ordering + tie-break documentation -------------------------------
        checks.append(
            _check(
                "C05",
                "word_loop_top_selection",
                ordered_word_loop_clip_ids=selection["ordered_word_loops"],
                top_n=WORD_LOOP_TOP_N,
                insertion_tie_groups=selection["word_loop_insertion_tie_groups"],
            )
        )
        # C06 set composition ---------------------------------------------------
        audio_hashes, missing_audio = compute_audio_hashes(selected, audio_dir)
        composition_ok = (
            len(selected) == EXPECTED_SET_SIZE
            and len(set(selected)) == len(selected)
            and sorted(selection["char_loops"])
            == sorted(clip for clip in selected if categories[clip] == CATEGORY_CEILING_CHAR_LOOP)
            and sorted(selection["early_loops"])
            == sorted(clip for clip in selected if categories[clip] == CATEGORY_EARLY_LOOP)
            and sorted(selection["named_controls"])
            == sorted(clip for clip in selected if categories[clip] == CATEGORY_CONTROL)
        )
        checks.append(
            _check(
                "C06",
                "set_composition",
                set_size=len(selected),
                distinct_clip_count=len(set(selected)),
                char_loop_count=len(selection["char_loops"]),
                early_loop_count=len(selection["early_loops"]),
                control_count=len(selection["named_controls"]),
            )
            if composition_ok
            else _fail(
                "C06",
                "set_composition",
                set_size=len(selected),
                distinct_clip_count=len(set(selected)),
            )
        )

        # C07 R4 membership -----------------------------------------------------
        manifest_id_set = set(manifest_ids)
        missing = sorted(clip for clip in selected if clip not in manifest_id_set)
        cohort_equals_manifest = set(cohort_ids) == manifest_id_set
        if not missing and cohort_equals_manifest:
            checks.append(
                _check(
                    "C07",
                    "dev_cohort_membership",
                    manifest_clip_count=len(manifest_id_set),
                )
            )
        else:
            checks.append(
                _fail(
                    "C07",
                    "dev_cohort_membership",
                    missing_clip_ids=missing,
                    cohort_equals_manifest=cohort_equals_manifest,
                )
            )

        # C08 R5 reserve exclusion ------------------------------------------------
        reserve = set(reserve_speakers)
        colliding = sorted(
            {str(manifest_speakers.get(clip, "")) for clip in selected} & reserve
        )
        dev_colliding = sorted(
            {str(manifest_speakers.get(clip, "")) for clip in cohort_ids} & reserve
        )
        if not colliding and not dev_colliding:
            checks.append(
                _check(
                    "C08",
                    "reserve_exclusion",
                    reserve_speaker_count=len(reserve),
                    dev_speaker_colliding_count=0,
                )
            )
        else:
            checks.append(
                _fail(
                    "C08",
                    "reserve_exclusion",
                    colliding_speaker_ids=colliding,
                    dev_cohort_colliding_speaker_ids=dev_colliding,
                )
            )

        records = selection_records(
            selected,
            categories,
            evidence,
            manifest_speakers,
            reserve_speakers,
            audio_hashes,
        )
        audio_check = (
            _fail("C10", "audio_files", missing_clip_ids=missing_audio)
            if missing_audio
            else _check("C10", "audio_files", hashed_clip_count=len(audio_hashes))
        )
    else:
        audio_check = _skip("C10", "audio_files", reason="selection_not_derived")

    # C09 frozen input hashes -----------------------------------------------------
    pin_checks: List[Tuple[str, Optional[str], str]] = [
        ("manifest", expected_manifest_sha256, manifest_sha256),
        ("reserve", expected_reserve_sha256, reserve_sha256),
    ]
    active_pins = [(name, expected, actual) for name, expected, actual in pin_checks if expected]
    if not active_pins:
        checks.append(
            _skip("C09", "frozen_input_hashes", reason="expected_hashes_not_pinned")
        )
    elif all(actual == expected for _, expected, actual in active_pins):
        checks.append(
            _check(
                "C09",
                "frozen_input_hashes",
                manifest_sha256=manifest_sha256,
                reserve_sha256=reserve_sha256,
            )
        )
    else:
        checks.append(
            _fail(
                "C09",
                "frozen_input_hashes",
                manifest_sha256=manifest_sha256,
                reserve_sha256=reserve_sha256,
            )
        )
    checks.append(audio_check)

    # C11 inspection cross-check ----------------------------------------------------
    if inspection is not None:
        issues: List[Dict[str, Any]] = []
        for clip_id in cohort_ids:
            if clip_id not in inspection["clips"]:
                issues.append({"issue": "clip_missing", "clip_id": clip_id})
                continue
            insp = inspection["clips"][clip_id]
            cls = evidence[clip_id]
            if abs(insp["runtime_sec"] - cls["runtime_sec"]) > epsilon:
                issues.append({"issue": "runtime_mismatch", "clip_id": clip_id})
            if insp["flagged"] != bool(cls["heuristic_flag"]):
                issues.append({"issue": "flag_mismatch", "clip_id": clip_id})
            if insp["has_word_loop"] != bool(cls["has_word_loop"]):
                issues.append({"issue": "word_loop_mismatch", "clip_id": clip_id})
            if insp["has_char_loop"] != bool(cls["has_char_loop"]):
                issues.append({"issue": "char_loop_mismatch", "clip_id": clip_id})
            if insp["diagnostic_insertions"] != int(cls["insertions"]):
                issues.append({"issue": "insertions_mismatch", "clip_id": clip_id})
        if sorted(inspection["ceiling_members"]) != sorted(ceiling["members"]):
            issues.append({"issue": "ceiling_member_mismatch"})
        if inspection["ceiling_min_sec"] is not None and (
            abs(float(inspection["ceiling_min_sec"]) - window[0]) > epsilon
        ):
            issues.append({"issue": "ceiling_min_mismatch"})
        if inspection["ceiling_max_sec"] is not None and (
            abs(float(inspection["ceiling_max_sec"]) - window[1]) > epsilon
        ):
            issues.append({"issue": "ceiling_max_mismatch"})
        checks.append(
            _fail("C11", "inspection_crosscheck", issues=issues)
            if issues
            else _check(
                "C11",
                "inspection_crosscheck",
                clips_checked=len(cohort_ids),
                inspection_sha256=inspection["sha256"],
            )
        )
    else:
        checks.append(_skip("C11", "inspection_crosscheck", reason="inspection_not_provided"))

    # C12 committed provenance hashes -------------------------------------------------
    provenance_issues: List[Dict[str, Any]] = []
    declared_metrics = str(evidence_provenance.get("metrics_sha256", ""))
    if declared_metrics and baseline_sha256 and declared_metrics != baseline_sha256:
        provenance_issues.append({"issue": "baseline_metrics_sha256"})
    if inspection is not None:
        declared_inspection = str(evidence_provenance.get("inspection_sha256", ""))
        if declared_inspection and declared_inspection != inspection["sha256"]:
            provenance_issues.append({"issue": "inspection_sha256"})
    declared_manifest = str(evidence_provenance.get("manifest_sha256", ""))
    if declared_manifest and declared_manifest != manifest_sha256:
        provenance_issues.append({"issue": "manifest_sha256"})
    checks.append(
        _fail("C12", "committed_provenance_hashes", issues=provenance_issues)
        if provenance_issues
        else _check("C12", "committed_provenance_hashes")
    )

    # C13 plan match --------------------------------------------------------------------
    if plan is not None:
        if derivation_failed:
            checks.append(
                _fail("C13", "plan_match", issue="derivation_failed_plan_unverifiable")
            )
        else:
            plan_clips = plan.get("clips", [])
            if not isinstance(plan_clips, list):
                checks.append(_fail("C13", "plan_match", issue="plan_clips_not_a_list"))
            else:
                expected_records = {
                    clip: {
                        "category": categories[clip],
                        "runtime_sec": float(evidence[clip]["runtime_sec"]),
                        "insertions": int(evidence[clip]["insertions"]),
                        "heuristic_flag": bool(evidence[clip]["heuristic_flag"]),
                        "has_word_loop": bool(evidence[clip]["has_word_loop"]),
                        "has_char_loop": bool(evidence[clip]["has_char_loop"]),
                    }
                    for clip in selected
                }
                issues = _plan_clip_issues(plan_clips, expected_records, audio_hashes)
                if issues:
                    checks.append(_fail("C13", "plan_match", issues=issues))
                else:
                    hashed = sum(1 for entry in plan_clips if entry.get("audio_sha256"))
                    checks.append(
                        _check(
                            "C13",
                            "plan_match",
                            plan_clip_count=len(plan_clips),
                            audio_sha256_present_count=hashed,
                        )
                    )
    else:
        checks.append(_skip("C13", "plan_match", reason="derive_mode"))

    # Assemble the report, then run the structural content-safety scan (C14).
    report: Dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "mode": mode,
        "passed": False,
        "rules": {
            "runtime_window_sec": list(window),
            "runtime_edge_epsilon_sec": epsilon,
            "ceiling_char_loop_expected": EXPECTED_CEILING_CHAR_LOOP_COUNT,
            "early_loop_expected": EXPECTED_EARLY_LOOP_COUNT,
            "word_loop_top_n": WORD_LOOP_TOP_N,
            "set_size_expected": EXPECTED_SET_SIZE,
        },
        "provenance": {
            "intelligence_json_sha256": evidence_sha256,
            "manifest_sha256": manifest_sha256,
            "reserve_json_sha256": reserve_sha256,
        },
        "checks": checks,
        "selection": records,
        "derivation_failed": derivation_failed,
    }
    safety = content_safety_violations(report)
    checks.append(
        _fail("C14", "content_safety_scan", violations=safety)
        if safety
        else _check("C14", "content_safety_scan")
    )
    report["passed"] = all(
        check.get("status") in ("pass", "skip") for check in report["checks"]
    )
    return report, bool(report["passed"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(report: Mapping[str, Any]) -> None:
    print(
        f"probe selection audit — mode {report.get('mode', 'unknown')}: "
        f"{'PASS' if report.get('passed') else 'VIOLATIONS'}"
    )
    for check in report.get("checks", []):
        marker = {"pass": "ok  ", "skip": "skip", "fail": "FAIL"}[check["status"]]
        print(f"  [{marker}] {check['id']} {check['kind']}")
    for record in report.get("selection", []):
        sha = str(record.get("audio_sha256") or "")
        print(
            f"  {record['clip_id']} cat={record['category']} "
            f"rt={record['runtime_sec']:.3f}s ins={record['insertions']} "
            f"word_loop={record['has_word_loop']} char_loop={record['has_char_loop']} "
            f"flag={record['heuristic_flag']} dev={record['in_dev_cohort']} "
            f"reserve_free={record['speaker_not_in_reserve']} sha={sha[:12]}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Phase 7.1 J1 probe 16-clip selection against the "
            "pre-registered rules (clip IDs, numbers, flags, hashes only)."
        )
    )
    parser.add_argument(
        "--intelligence-json",
        type=Path,
        default=DEFAULT_INTELLIGENCE_JSON,
        help="committed failure-intelligence JSON (insertions, loop flags)",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=DEFAULT_BASELINE_JSON,
        help=(
            "committed baseline metrics JSON providing per-clip "
            "inference_runtime_sec (R4 runtime provenance)"
        ),
    )
    parser.add_argument(
        "--inspection-json",
        type=Path,
        default=None,
        help="optional offline inspection report JSON for cross-checks",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--reserve-json", type=Path, default=DEFAULT_RESERVE_JSON)
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=None,
        help="optional built plan JSON; audits it instead of deriving candidates",
    )
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument(
        "--expected-manifest-sha256",
        default=FROZEN_MANIFEST_SHA256,
        help="pinned sha256 of the frozen manifest (pass '' to disable the pin)",
    )
    parser.add_argument(
        "--expected-reserve-sha256",
        default=FROZEN_RESERVE_SHA256,
        help="pinned sha256 of the holdout reserve JSON (pass '' to disable the pin)",
    )
    args = parser.parse_args(argv)

    try:
        intelligence, evidence_sha, provenance = load_intelligence(args.intelligence_json)
        runtimes, baseline_sha = load_baseline_runtimes(args.baseline_json)
        evidence = build_evidence(intelligence, runtimes)
        manifest_ids, manifest_speakers, manifest_sha = load_manifest(args.manifest)
        reserve_speakers, reserve_sha = load_reserve(args.reserve_json)
        inspection = (
            load_inspection(args.inspection_json) if args.inspection_json else None
        )
        plan = json.loads(args.plan_json.read_text(encoding="utf-8")) if args.plan_json else None
    except (OSError, json.JSONDecodeError, DataError, KeyError) as error:
        print(f"usage error: {error}", file=sys.stderr)
        return EXIT_USAGE

    report, passed = run_audit(
        evidence=evidence,
        evidence_sha256=evidence_sha,
        evidence_provenance=provenance,
        manifest_ids=manifest_ids,
        manifest_speakers=manifest_speakers,
        manifest_sha256=manifest_sha,
        reserve_speakers=reserve_speakers,
        reserve_sha256=reserve_sha,
        audio_dir=args.audio_dir,
        baseline_sha256=baseline_sha,
        inspection=inspection,
        expected_manifest_sha256=args.expected_manifest_sha256 or None,
        expected_reserve_sha256=args.expected_reserve_sha256 or None,
        plan=plan,
    )

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _print_summary(report)
    return EXIT_OK if passed else EXIT_VIOLATION


if __name__ == "__main__":
    sys.exit(main())

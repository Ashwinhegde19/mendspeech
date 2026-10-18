"""Synthetic, offline tests for the Phase 7.1 probe selection audit.

All clip IDs, speakers, runtimes, and counts used here are invented for the
tests. No private transcript, reference text, or model output is read, and no
network, GPU, or model download is involved. The audit's structural
content-safety scan is exercised directly, including the guarantee that the
audit output can only contain clip IDs, numbers, booleans, flag names, and
hashes in string positions (a marker token planted in the fixture manifest's
reference column must never appear in any audit output; the Devanagari
probes use isolated Unicode codepoints, never real language words).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_hinglish_probe_selection import (  # noqa: E402
    assign_category,
    content_safety_violations,
    in_runtime_window,
    main,
)

# Marker string planted in the fixture manifest reference column. The audit
# must never extract or echo reference-column values, so this must never reach
# the audit output.
LEAK_MARKER = "zqxleakmarker"

WORD_TOKEN = "w" * 16
CHAR_TOKEN = "c" * 16
NOISE_TOKEN = "n" * 16

# Word-loop ceiling runtimes spanning the full pre-registered window, 3
# decimals, so that window-edge checks (member min/max == 3.687/3.734) hold.
CEILING_RUNTIMES = (3.687, 3.693, 3.699, 3.704, 3.710, 3.716, 3.722, 3.728, 3.734)


def clip(
    prefix: str,
    *,
    speaker: Optional[str] = None,
    ceiling: bool = False,
    word_loop: bool = False,
    char_loop: bool = False,
    flagged: bool = False,
    insertions: int = 0,
    runtime: Optional[float] = None,
    failure_class: Optional[str] = None,
    loop_coverage: int = 0,
) -> Dict[str, Any]:
    """Build one synthetic evidence record shaped like committed artifacts."""
    token = WORD_TOKEN if word_loop else (CHAR_TOKEN if char_loop else NOISE_TOKEN)
    clip_id = f"{prefix}_{token}_0000"
    if ceiling and failure_class is None:
        failure_class = "generation_cap_with_repetition"
    elif failure_class is None:
        failure_class = "script_mismatch_dominant"
    if ceiling and runtime is None:
        runtime = 3.7
    elif runtime is None:
        runtime = 1.0
    return {
        "clip_id": clip_id,
        "speaker_id": speaker or prefix,
        "failure_class": failure_class,
        "confidence": "structural_evidence" if ceiling else "heuristic_estimate",
        "insertions": insertions,
        "cap_hit": ceiling,
        "heuristic_flag": flagged,
        "has_word_loop": word_loop,
        "has_char_loop": char_loop,
        "loop_coverage_words": loop_coverage,
        "runtime_sec": float(runtime),
    }


def default_cohort(tie: bool = False) -> List[Dict[str, Any]]:
    """Synthetic cohort: 9 word-loop ceilings, 3 char-loop ceilings, 2 early
    loops (with the mandated prefixes), 5 controls.

    Word-loop insertions descend; with ``tie`` the 8th and 9th ranked loops
    both carry 45 insertions and clip-ID-ascending must resolve the 8th slot
    toward the smaller prefix.
    """
    records: List[Dict[str, Any]] = []
    insertions = [130, 120, 110, 100, 90, 80, 70, 45, 45] if tie else [
        130, 120, 110, 100, 90, 80, 70, 60, 40,
    ]
    for index in range(9):
        prefix = f"1000{index + 1:02d}"
        records.append(
            clip(
                prefix,
                ceiling=True,
                word_loop=True,
                flagged=True,
                insertions=insertions[index],
                runtime=CEILING_RUNTIMES[index],
                loop_coverage=insertions[index] + 2,
            )
        )
    for index, prefix in enumerate(("100010", "100011", "100012")):
        records.append(
            clip(
                prefix,
                ceiling=True,
                char_loop=True,
                flagged=False,
                insertions=[3, 2, 1][index],
                runtime=3.69 + index * 0.01,
            )
        )
    records.extend(
        [
            clip("304558", word_loop=True, flagged=True, insertions=16, runtime=1.054,
                 loop_coverage=23),
            clip("399287", word_loop=True, flagged=False, insertions=9, runtime=1.2,
                 loop_coverage=12),
            clip("111627", insertions=0, runtime=1.562),
            clip("136706", insertions=0, runtime=0.667),
            clip("415128", insertions=1, runtime=0.170),
            clip("200000", insertions=2, runtime=0.9),
            clip("200001", insertions=0, runtime=0.4),
        ]
    )
    return records


def write_intelligence(
    path: Path,
    cohort: Sequence[Dict[str, Any]],
    provenance: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "report_version": "mendspeech.hinglish_failure_intelligence.v1",
        "provenance": provenance or {},
        "per_clip": list(cohort),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_baseline(path: Path, cohort: Sequence[Dict[str, Any]]) -> None:
    payload = {
        "schema_version": "mendspeech.hinglish_validation_baseline.v2",
        "per_clip": [
            {
                "clip_id": entry["clip_id"],
                "speaker_id": entry["speaker_id"],
                "inference_runtime_sec": entry["runtime_sec"],
            }
            for entry in cohort
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_inspection(path: Path, cohort: Sequence[Dict[str, Any]]) -> None:
    members = [entry["clip_id"] for entry in cohort if entry["cap_hit"]]
    member_runtimes = [entry["runtime_sec"] for entry in cohort if entry["cap_hit"]]
    payload = {
        "clip_count": len(cohort),
        "per_clip": [
            {
                "clip_id": entry["clip_id"],
                "flagged": entry["heuristic_flag"],
                "runtime_sec": entry["runtime_sec"],
                "has_word_loop": entry["has_word_loop"],
                "has_char_loop": entry["has_char_loop"],
                "diagnostic_insertions": entry["insertions"],
            }
            for entry in cohort
        ],
        "runtime_ceiling": {
            "members": members,
            "min_sec": min(member_runtimes),
            "max_sec": max(member_runtimes),
            "found": True,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_manifest(path: Path, cohort: Sequence[Dict[str, Any]]) -> str:
    """Write a tiny manifest including a reference column that must be ignored."""
    lines = ["clip_id,speaker_id,reference_transcript"]
    for entry in cohort:
        lines.append(f"{entry['clip_id']},{entry['speaker_id']},{LEAK_MARKER}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_audio(directory: Path, cohort: Sequence[Dict[str, Any]]) -> None:
    for entry in cohort:
        (directory / f"{entry['clip_id']}.wav").write_bytes(
            (f"RIFF{entry['clip_id']}" + "\x00" * 16).encode("ascii")
        )


def write_reserve(path: Path, speakers: Sequence[str]) -> str:
    payload = {
        "pool_version": "mendspeech.mucs_holdout_pool.v0",
        "speakers": [{"speaker_id": speaker} for speaker in speakers],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audio_sha256(audio_dir: Path, clip_id: str) -> str:
    return hashlib.sha256((audio_dir / f"{clip_id}.wav").read_bytes()).hexdigest()


@pytest.fixture()
def fixtures(tmp_path: Path) -> Dict[str, Any]:
    cohort = default_cohort()
    baseline_path = tmp_path / "baseline.json"
    manifest_path = tmp_path / "manifest.csv"
    audio_dir = tmp_path / "audio"
    reserve_path = tmp_path / "reserve.json"
    inspection_path = tmp_path / "inspection.json"
    audio_dir.mkdir()
    write_baseline(baseline_path, cohort)
    manifest_sha = write_manifest(manifest_path, cohort)
    write_audio(audio_dir, cohort)
    reserve_sha = write_reserve(reserve_path, speakers=["999000", "999001"])
    write_inspection(inspection_path, cohort)
    inspection_sha = hashlib.sha256(inspection_path.read_bytes()).hexdigest()
    baseline_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    intelligence_path = tmp_path / "intelligence.json"
    write_intelligence(
        intelligence_path,
        cohort,
        provenance={
            "metrics_sha256": baseline_sha,
            "inspection_sha256": inspection_sha,
            "manifest_sha256": manifest_sha,
        },
    )
    return {
        "cohort": cohort,
        "intelligence": intelligence_path,
        "baseline": baseline_path,
        "manifest": manifest_path,
        "manifest_sha": manifest_sha,
        "audio": audio_dir,
        "reserve": reserve_path,
        "reserve_sha": reserve_sha,
        "inspection": inspection_path,
    }


def base_argv(fixture: Dict[str, Any], report_out: Path) -> List[str]:
    return [
        "--intelligence-json", str(fixture["intelligence"]),
        "--baseline-json", str(fixture["baseline"]),
        "--manifest", str(fixture["manifest"]),
        "--audio-dir", str(fixture["audio"]),
        "--reserve-json", str(fixture["reserve"]),
        "--report-out", str(report_out),
        "--expected-manifest-sha256", fixture["manifest_sha"],
        "--expected-reserve-sha256", fixture["reserve_sha"],
    ]


def run_cli(
    fixture: Dict[str, Any],
    tmp_path: Path,
    extra_args: Optional[List[str]] = None,
) -> Tuple[int, Dict[str, Any]]:
    report_out = tmp_path / "audit_report.json"
    code = main(base_argv(fixture, report_out) + (extra_args or []))
    report = json.loads(report_out.read_text(encoding="utf-8"))
    return code, report


def check_by_id(report: Dict[str, Any], check_id: str) -> Dict[str, Any]:
    return next(check for check in report["checks"] if check["id"] == check_id)


def rewritten_fixtures(
    tmp_path: Path, cohort: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Write a full fixture set for a mutated cohort (no pinned hashes)."""
    baseline_path = tmp_path / "baseline2.json"
    manifest_path = tmp_path / "manifest2.csv"
    audio_dir = tmp_path / "audio2"
    reserve_path = tmp_path / "reserve2.json"
    audio_dir.mkdir()
    write_baseline(baseline_path, cohort)
    manifest_sha = write_manifest(manifest_path, cohort)
    write_audio(audio_dir, cohort)
    reserve_sha = write_reserve(reserve_path, speakers=["999000", "999001"])
    intelligence_path = tmp_path / "intelligence2.json"
    write_intelligence(intelligence_path, cohort)
    return {
        "cohort": cohort,
        "intelligence": intelligence_path,
        "baseline": baseline_path,
        "manifest": manifest_path,
        "manifest_sha": manifest_sha,
        "audio": audio_dir,
        "reserve": reserve_path,
        "reserve_sha": reserve_sha,
        "inspection": None,
    }


# ---------------------------------------------------------------------------
# R1 runtime-window boundary behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("runtime", "expected"),
    [
        (3.687, True),          # exact lower edge (inclusive)
        (3.734, True),          # exact upper edge (inclusive)
        (3.7339, True),         # inside the window
        (3.7341, False),        # 0.0001 above the edge: NOT absorbed
        (3.6869999, True),      # float noise below the edge: absorbed (epsilon)
        (3.6869, False),        # 1e-4 below the edge: NOT absorbed
        (1.562, False),
        (None, False),
    ],
)
def test_runtime_window_boundaries(runtime: Optional[float], expected: bool) -> None:
    """Documented tolerance: inclusive edges plus a 1e-6 float-noise epsilon."""
    assert in_runtime_window(runtime) is expected


def test_assign_category_partition() -> None:
    base = dict(
        failure_class="script_mismatch_dominant",
        confidence="heuristic_estimate",
        insertions=0,
        cap_hit=False,
        heuristic_flag=False,
        has_word_loop=False,
        has_char_loop=False,
        loop_coverage_words=0,
    )
    assert assign_category({**base, "has_word_loop": True}, True) == "ceiling_word_loop"
    assert assign_category({**base, "has_char_loop": True}, True) == "ceiling_char_loop"
    assert (
        assign_category({**base, "has_word_loop": True, "heuristic_flag": True}, False)
        == "early_loop"
    )
    assert assign_category(base, False) == "control_candidate"
    assert assign_category({**base, "heuristic_flag": True}, False) == "unclassified"


# ---------------------------------------------------------------------------
# Full derive-mode pipeline
# ---------------------------------------------------------------------------


def test_derive_mode_passes_and_selects_expected_16(
    tmp_path: Path, fixtures: Dict[str, Any]
) -> None:
    code, report = run_cli(fixtures, tmp_path)
    assert code == 0
    assert report["passed"] is True
    assert not [check for check in report["checks"] if check["status"] == "fail"]
    audio_dir: Path = fixtures["audio"]
    expected = {
        f"1000{index + 1:02d}_{WORD_TOKEN}_0000" for index in range(8)
    } | {
        f"10001{index}_{CHAR_TOKEN}_0000" for index in range(0, 3)
    } | {
        "304558_wwwwwwwwwwwwwwww_0000",
        "399287_wwwwwwwwwwwwwwww_0000",
        "111627_nnnnnnnnnnnnnnnn_0000",
        "136706_nnnnnnnnnnnnnnnn_0000",
        "415128_nnnnnnnnnnnnnnnn_0000",
    }
    selected = {record["clip_id"] for record in report["selection"]}
    assert selected == expected
    assert len(report["selection"]) == 16
    for record in report["selection"]:
        assert record["in_dev_cohort"] is True
        assert record["speaker_not_in_reserve"] is True
        assert record["audio_sha256"] == audio_sha256(audio_dir, record["clip_id"])
    check = check_by_id(report, "C14")
    assert check["status"] == "pass"


def test_derive_is_deterministic(tmp_path: Path, fixtures: Dict[str, Any]) -> None:
    first_out = tmp_path / "first.json"
    code = main(base_argv(fixtures, first_out))
    assert code == 0
    second_out = tmp_path / "second.json"
    code = main(base_argv(fixtures, second_out))
    assert code == 0
    assert first_out.read_bytes() == second_out.read_bytes()


def test_top8_tie_break_prefers_smaller_clip_id(tmp_path: Path) -> None:
    """8th/9th word loops tie at 45 insertions; clip-ID ascending wins slot 8."""
    fixture = rewritten_fixtures(tmp_path, default_cohort(tie=True))
    code, report = run_cli(fixture, tmp_path)
    assert code == 0
    ids = [record["clip_id"] for record in report["selection"]]
    small = f"100008_{WORD_TOKEN}_0000"
    large = f"100009_{WORD_TOKEN}_0000"
    assert small in ids
    assert large not in ids
    ties = check_by_id(report, "C05")["insertion_tie_groups"]
    assert ties == {"45": [small, large]}


# ---------------------------------------------------------------------------
# R3 mandate enforcement (STOP rules)
# ---------------------------------------------------------------------------


def test_missing_mandated_early_clip_stops(tmp_path: Path, fixtures: Dict[str, Any]) -> None:
    cohort = [e for e in fixtures["cohort"] if not e["clip_id"].startswith("399287_")]
    fixture = rewritten_fixtures(tmp_path, cohort)
    code, report = run_cli(fixture, tmp_path)
    assert code == 2
    assert report["passed"] is False
    failed = check_by_id(report, "C03")
    kinds = {violation["kind"] for violation in failed["violations"]}
    assert "early_loop_count_mismatch" in kinds
    assert "early_loop_mandate_unresolved" in kinds
    assert report["selection"] == []
    assert check_by_id(report, "C10")["status"] == "skip"


def test_extra_early_loop_clip_stops(tmp_path: Path, fixtures: Dict[str, Any]) -> None:
    cohort = fixtures["cohort"] + [
        clip("333333", word_loop=True, flagged=True, insertions=5, runtime=0.8)
    ]
    fixture = rewritten_fixtures(tmp_path, cohort)
    code, report = run_cli(fixture, tmp_path)
    assert code == 2
    kinds = {
        violation["kind"] for violation in check_by_id(report, "C03")["violations"]
    }
    assert "early_loop_count_mismatch" in kinds


def test_control_criteria_violation_stops(tmp_path: Path, fixtures: Dict[str, Any]) -> None:
    cohort = [dict(entry) for entry in fixtures["cohort"]]
    for entry in cohort:
        if entry["clip_id"].startswith("111627_"):
            entry["heuristic_flag"] = True
    fixture = rewritten_fixtures(tmp_path, cohort)
    code, report = run_cli(fixture, tmp_path)
    assert code == 2
    kinds = {
        violation["kind"] for violation in check_by_id(report, "C03")["violations"]
    }
    assert "unclassified_clips" in kinds
    assert "named_control_unresolved" in kinds
    assert check_by_id(report, "C04")["status"] == "fail"


def test_char_loop_count_violation_stops(tmp_path: Path, fixtures: Dict[str, Any]) -> None:
    cohort = [e for e in fixtures["cohort"] if not e["clip_id"].startswith("100012_")]
    fixture = rewritten_fixtures(tmp_path, cohort)
    code, report = run_cli(fixture, tmp_path)
    assert code == 2
    kinds = {
        violation["kind"] for violation in check_by_id(report, "C03")["violations"]
    }
    assert "char_loop_count_mismatch" in kinds


def test_ceiling_definition_disagreement_is_a_violation(
    tmp_path: Path, fixtures: Dict[str, Any]
) -> None:
    """A cap-class clip with a low runtime breaks R1; the audit must fail."""

    def disagree(entry: Dict[str, Any]) -> bool:
        return entry["clip_id"].startswith("100005_")

    cohort = [dict(entry) for entry in fixtures["cohort"]]
    for entry in cohort:
        if disagree(entry):
            entry["runtime_sec"] = 0.5
            entry["cap_hit"] = False
            entry["failure_class"] = "generation_cap_with_repetition"
    # cap_hit must stay truthy for class_flag True while runtime flag is False.
    for entry in cohort:
        if disagree(entry):
            entry["cap_hit"] = True
    fixture = rewritten_fixtures(tmp_path, cohort)
    code, report = run_cli(fixture, tmp_path)
    assert code == 2
    assert check_by_id(report, "C02")["status"] == "fail"
    kinds = {
        violation["kind"] for violation in check_by_id(report, "C03")["violations"]
    }
    assert "ceiling_definition_disagreement" in kinds


# ---------------------------------------------------------------------------
# R4/R5 enforcement
# ---------------------------------------------------------------------------


def test_r4_membership_violation(tmp_path: Path, fixtures: Dict[str, Any]) -> None:
    """A derived clip missing from the manifest is a violation."""
    outcast = f"100003_{WORD_TOKEN}_0000"
    manifest = tmp_path / "manifest_missing.csv"
    lines = ["clip_id,speaker_id,reference_transcript"]
    for entry in fixtures["cohort"]:
        if entry["clip_id"] != outcast:
            lines.append(f"{entry['clip_id']},{entry['speaker_id']},{LEAK_MARKER}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fixture = dict(fixtures)
    fixture["manifest"] = manifest
    fixture["manifest_sha"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    code, report = run_cli(fixture, tmp_path)
    assert code == 2
    missing = check_by_id(report, "C07")["missing_clip_ids"]
    assert missing == [outcast]


def test_r5_reserve_collision_violation(tmp_path: Path, fixtures: Dict[str, Any]) -> None:
    reserve = tmp_path / "reserve_collision.json"
    write_reserve(reserve, speakers=["100001"])  # speaker of a derived ceiling clip
    fixture = dict(fixtures)
    fixture["reserve"] = reserve
    fixture["reserve_sha"] = hashlib.sha256(reserve.read_bytes()).hexdigest()
    code, report = run_cli(fixture, tmp_path)
    assert code == 2
    assert check_by_id(report, "C08")["colliding_speaker_ids"] == ["100001"]


def test_frozen_input_hash_mismatch_fails(
    tmp_path: Path, fixtures: Dict[str, Any]
) -> None:
    code, report = run_cli(
        fixtures, tmp_path, extra_args=["--expected-manifest-sha256", "0" * 64]
    )
    assert code == 2
    assert check_by_id(report, "C09")["status"] == "fail"


def test_unpinned_hashes_are_a_documented_skip(
    tmp_path: Path, fixtures: Dict[str, Any]
) -> None:
    code, report = run_cli(
        fixtures,
        tmp_path,
        extra_args=[
            "--expected-manifest-sha256", "",
            "--expected-reserve-sha256", "",
        ],
    )
    assert code == 0
    skip = check_by_id(report, "C09")
    assert skip["status"] == "skip"


# ---------------------------------------------------------------------------
# Inspection cross-check and provenance
# ---------------------------------------------------------------------------


def test_inspection_crosscheck_catches_tampered_runtime(
    tmp_path: Path, fixtures: Dict[str, Any]
) -> None:
    tampered = tmp_path / "inspection_tampered.json"
    payload = json.loads(fixtures["inspection"].read_text(encoding="utf-8"))
    for record in payload["per_clip"]:
        if record["clip_id"].startswith(f"100002_{WORD_TOKEN}"):
            record["runtime_sec"] = 0.1
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    code, report = run_cli(
        fixtures, tmp_path, extra_args=["--inspection-json", str(tampered)]
    )
    assert code == 2
    kinds = {issue["issue"] for issue in check_by_id(report, "C11")["issues"]}
    assert "runtime_mismatch" in kinds


def test_inspection_crosscheck_catches_ceiling_member_mismatch(
    tmp_path: Path, fixtures: Dict[str, Any]
) -> None:
    tampered = tmp_path / "inspection_members.json"
    payload = json.loads(fixtures["inspection"].read_text(encoding="utf-8"))
    payload["runtime_ceiling"]["members"] = payload["runtime_ceiling"]["members"][:2]
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    code, report = run_cli(
        fixtures, tmp_path, extra_args=["--inspection-json", str(tampered)]
    )
    assert code == 2
    kinds = {issue["issue"] for issue in check_by_id(report, "C11")["issues"]}
    assert "ceiling_member_mismatch" in kinds


def test_committed_provenance_hash_mismatch_fails(
    tmp_path: Path, fixtures: Dict[str, Any]
) -> None:
    intelligence_path = tmp_path / "intelligence_badprov.json"
    write_intelligence(
        intelligence_path,
        fixtures["cohort"],
        provenance={"manifest_sha256": "e" * 64},
    )
    fixture = dict(fixtures)
    fixture["intelligence"] = intelligence_path
    code, report = run_cli(
        fixture,
        tmp_path,
        extra_args=["--inspection-json", str(fixtures["inspection"])],
    )
    assert code == 2
    issues = check_by_id(report, "C12")["issues"]
    assert any("manifest_sha256" in entry["issue"] for entry in issues)


# ---------------------------------------------------------------------------
# Plan-audit mode
# ---------------------------------------------------------------------------


def plan_clips_from_report(report_out: Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(report_out).read_text(encoding="utf-8"))
    return [
        {
            "clip_id": record["clip_id"],
            "category": record["category"],
            "runtime_sec": record["runtime_sec"],
            "insertions": record["insertions"],
            "heuristic_flag": record["heuristic_flag"],
            "has_word_loop": record["has_word_loop"],
            "has_char_loop": record["has_char_loop"],
            "audio_sha256": record["audio_sha256"],
        }
        for record in payload["selection"]
    ]


def derive_report_out(tmp_path: Path, fixtures: Dict[str, Any]) -> Path:
    code, report = run_cli(fixtures, tmp_path)
    assert code == 0
    return tmp_path / "audit_report.json"


def run_plan_cli(
    fixture: Dict[str, Any],
    tmp_path: Path,
    plan_path: Path,
    inspection: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    extra = ["--plan-json", str(plan_path)]
    if inspection:
        extra += ["--inspection-json", str(fixture["inspection"])]
    return run_cli(fixture, tmp_path, extra_args=extra)


def test_plan_audit_passes_on_correct_plan(
    tmp_path: Path, fixtures: Dict[str, Any]
) -> None:
    derive_out = derive_report_out(tmp_path, fixtures)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "plan_schema": "mendspeech.phase7_1_probe_selection.v1",
                "clips": plan_clips_from_report(derive_out),
            }
        ),
        encoding="utf-8",
    )
    code, report = run_plan_cli(fixtures, tmp_path, plan_path, inspection=True)
    assert code == 0
    plan_check = check_by_id(report, "C13")
    assert plan_check["status"] == "pass"
    assert plan_check["plan_clip_count"] == 16
    assert plan_check["audio_sha256_present_count"] == 16


@pytest.mark.parametrize(
    ("corrupt", "expected_issue"),
    [
        ("category", "category_mismatch"),
        ("hash", "audio_sha256_mismatch"),
        ("insertions", "insertions_mismatch"),
        ("runtime", "runtime_mismatch"),
        ("flag", "flag_mismatch"),
        ("drop_clip", "clip_id_set_mismatch"),
        ("wrong_clip", "clip_id_set_mismatch"),
    ],
)
def test_plan_audit_catches_corruption(
    tmp_path: Path, fixtures: Dict[str, Any], corrupt: str, expected_issue: str
) -> None:
    derive_out = derive_report_out(tmp_path, fixtures)
    clips = plan_clips_from_report(derive_out)
    if corrupt == "category":
        clips[0]["category"] = "early_loop"
    elif corrupt == "hash":
        clips[0]["audio_sha256"] = "f" * 64
    elif corrupt == "insertions":
        clips[0]["insertions"] = clips[0]["insertions"] + 1
    elif corrupt == "runtime":
        clips[0]["runtime_sec"] = float(clips[0]["runtime_sec"]) + 0.5
    elif corrupt == "flag":
        clips[0]["has_word_loop"] = not clips[0]["has_word_loop"]
    elif corrupt == "drop_clip":
        clips = clips[1:]
    elif corrupt == "wrong_clip":
        clips.append({**clips[0], "clip_id": f"555555_{WORD_TOKEN}_0000"})
    plan_path = tmp_path / f"plan_{corrupt}.json"
    plan_path.write_text(
        json.dumps(
            {"plan_schema": "mendspeech.phase7_1_probe_selection.v1", "clips": clips}
        ),
        encoding="utf-8",
    )
    code, report = run_plan_cli(fixtures, tmp_path, plan_path)
    assert code == 2
    kinds = {issue["issue"] for issue in check_by_id(report, "C13")["issues"]}
    assert expected_issue in kinds


# ---------------------------------------------------------------------------
# Content-safety structure check
# ---------------------------------------------------------------------------


def test_content_safety_scan_rejects_transcript_like_values() -> None:
    # Whitespace-containing strings (sentences, transcripts) always fail.
    assert content_safety_violations({"note": "hello world this is a transcript"})
    # Isolated Devanagari codepoints (synthetic, not language words) always fail.
    assert content_safety_violations({"note": "\u0900abcd"})
    assert content_safety_violations({"note": "abc\u0915"})
    # Unsafe keys are rejected.
    assert content_safety_violations({"unsafe key with space": "x"})
    assert content_safety_violations({"clip_id": "111627_bYPRSx2sjXooUdv8_0000, गreeting"}) != []


def test_content_safety_scan_accepts_safe_values() -> None:
    report = {
        "schema": "mendspeech.phase7_1_probe_selection_audit.v1",
        "mode": "derive",
        "clip_id": "111627_bYPRSx2sjXooUdv8_0000",
        "speaker_id": "111627",
        "audio_sha256": "ab" * 32,
        "category": "ceiling_word_loop",
        "runtime_sec": 3.687,
        "insertions": 132,
        "has_word_loop": True,
        "checks": [{"id": "C01", "kind": "cohort_record_completeness", "status": "pass"}],
    }
    assert content_safety_violations(report) == []


def test_audit_output_never_carries_transcript_text(
    tmp_path: Path, fixtures: Dict[str, Any]
) -> None:
    code, report = run_cli(fixtures, tmp_path)
    assert code == 0
    report_out = tmp_path / "audit_report.json"
    text = report_out.read_text(encoding="utf-8")
    assert LEAK_MARKER not in text
    assert "\u0900" not in text  # no Devanagari block characters anywhere
    assert report["schema"] == "mendspeech.phase7_1_probe_selection_audit.v1"
    assert content_safety_violations(json.loads(text)) == []


# ---------------------------------------------------------------------------
# CLI robustness
# ---------------------------------------------------------------------------


def test_missing_input_file_is_a_usage_error(
    tmp_path: Path, fixtures: Dict[str, Any]
) -> None:
    argv = [
        "--intelligence-json", str(tmp_path / "missing.json"),
        "--baseline-json", str(fixtures["baseline"]),
        "--manifest", str(fixtures["manifest"]),
        "--audio-dir", str(fixtures["audio"]),
        "--reserve-json", str(fixtures["reserve"]),
        "--report-out", str(tmp_path / "unused.json"),
        "--expected-manifest-sha256", "",
        "--expected-reserve-sha256", "",
    ]
    assert main(argv) == 1

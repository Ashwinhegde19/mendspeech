"""Guards the plan documents against drift, contradictions, and stale claims.

``tests/test_docs_links.py`` proves that links resolve. This suite proves that
the plan is internally consistent: the manifest matches the day specs, statuses
and prerequisites are valid, derived documents are regenerated, and no public
document still advertises scope that was removed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
DAYS_DIR = DOCS / "days"
PLAN = DOCS / "plan_manifest.json"
DAY_RE = re.compile(r"^# Day (\d+): (.+)$", re.M)
STATUS_RE = re.compile(r"STATUS: (CORE|LEARN-ONLY|MERGED|DROPPED)")
VALID_STATUS = {"CORE", "LEARN-ONLY", "MERGED", "DROPPED"}
SESSION_STATUS = {"CORE", "LEARN-ONLY", "MERGED", "DROPPED"}

PUBLIC_DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "pyproject.toml",
    DOCS / "REVISED_EXECUTION_PLAN.md",
    DOCS / "MendSpeech_Project_Blueprint.md",
    DOCS / "MendSpeech_8_Week_Master_Roadmap.md",
    DOCS / "LATENCY_AND_QUALITY_CONTRACT.md",
    DOCS / "EDITOR_AND_RL_CONTRACT.md",
    DOCS / "SPEECH_ML_SYSTEMS_DRILLS.md",
    DOCS / "MendSpeech_PDF_Set_Index.md",
]

# Terms from the removed restoration scope, and motivation that must never
# appear in a public repository.
STALE_SCOPE = re.compile(
    r"repair_modes_calibrated|app/mendspeech_|src/tts/|src/repair/|"
    r"direct_audio_inpaint|vocoder|cascaded repair|selective semantic speech restoration|"
    r"boundary matched|seam diagnostics|abstain\b|abstention",
    re.I,
)
FORBIDDEN_MOTIVATION = re.compile(
    r"recruiter|job search|job description|job posting|hiring|hired|"
    r"resume|cover letter|interview prep|portfolio for|apply(ing)? (to|for a)|"
    r"applicant|career change|wispr|sarvam|level\.ai|bolna",
    re.I,
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def day_texts() -> dict[int, str]:
    return {
        number: (DAYS_DIR / f"day_{number:02d}.md").read_text(encoding="utf-8")
        for number in range(1, 57)
    }


def test_manifest_lists_every_day(manifest: dict) -> None:
    numbers = [entry["day"] for entry in manifest["days"]]
    assert numbers == list(range(1, 57))


def test_manifest_matches_day_specs(manifest: dict, day_texts: dict[int, str]) -> None:
    for entry in manifest["days"]:
        number = entry["day"]
        text = day_texts[number]
        assert f"# Day {number:02d}: {entry['title']}" in text, number
        status = STATUS_RE.search(text)
        if number <= 9:
            assert status is None or status.group(1) == entry["status"], number
            continue
        assert status is not None, f"day {number:02d} has no status banner"
        assert status.group(1) == entry["status"], number


def test_every_status_is_known(manifest: dict) -> None:
    for entry in manifest["days"]:
        assert entry["status"] in VALID_STATUS, entry


def test_prerequisites_precede_their_session(manifest: dict) -> None:
    for entry in manifest["days"]:
        for prereq in entry["prerequisites"]:
            assert prereq < entry["day"], f"day {entry['day']:02d} depends on {prereq}"


def test_prerequisites_resolve_to_real_days(manifest: dict) -> None:
    known = {entry["day"] for entry in manifest["days"]}
    for entry in manifest["days"]:
        for prereq in entry["prerequisites"]:
            assert prereq in known, f"day {entry['day']:02d} -> unknown day {prereq}"


def test_non_core_sessions_declare_no_required_artifacts(
    day_texts: dict[int, str],
) -> None:
    for number, text in day_texts.items():
        status = STATUS_RE.search(text)
        if status is None or status.group(1) in SESSION_STATUS - {"CORE"}:
            continue
        if status.group(1) == "DROPPED":
            continue
        assert "2. Build in MendSpeech" in text, number


def test_effort_is_declared_for_new_sessions(manifest: dict, day_texts: dict[int, str]) -> None:
    for entry in manifest["days"]:
        if entry["day"] < 10:
            continue
        low, high = entry["effort_hours"]
        assert 0 <= low <= high, entry["day"]
        if entry["status"] in {"LEARN-ONLY", "MERGED"}:
            assert low == 0, entry["day"]
        assert "**Effort:**" in day_texts[entry["day"]], entry["day"]


def test_phase_ranges_partition_the_sequence(manifest: dict) -> None:
    covered: list[int] = []
    for phase in manifest["phases"]:
        low, high = phase["days"]
        covered.extend(range(low, high + 1))
        core = [
            entry["day"]
            for entry in manifest["days"]
            if low <= entry["day"] <= high and entry["status"] == "CORE"
        ]
        assert phase["core_sessions"] == len(core), phase["id"]
    missing = sorted(set(range(1, 57)) - set(covered))
    assert missing == [17], missing


def test_execution_plan_session_count_matches_manifest(manifest: dict) -> None:
    plan = (DOCS / "REVISED_EXECUTION_PLAN.md").read_text(encoding="utf-8")
    core_total = sum(p["core_sessions"] for p in manifest["phases"])
    remaining = core_total - 9
    assert re.search(rf"\*\*{remaining}\b", plan), "execution plan must state the session total"
    low = sum(p["effort_hours"][0] for p in manifest["phases"])
    high = sum(p["effort_hours"][1] for p in manifest["phases"])
    assert str(low) in plan and str(high) in plan, "execution plan must state the effort range"


def test_calibration_is_an_explicit_deliverable(manifest: dict) -> None:
    day = next(e for e in manifest["days"] if e["day"] == 25)
    assert day["status"] == "CORE"
    text = (DAYS_DIR / "day_25.md").read_text(encoding="utf-8")
    assert "calibration.py" in text
    assert "reliability" in text.lower()


def test_rl_sessions_are_gated_by_feasibility(manifest: dict) -> None:
    by_day = {e["day"]: e for e in manifest["days"]}
    pilot = by_day[16]
    assert by_day[13]["day"] in pilot["prerequisites"]
    assert 15 in pilot["prerequisites"]
    run = by_day[40]
    assert 16 in run["prerequisites"], "the RL run depends on the feasibility pilot"


def test_serving_precedes_the_latency_budget(manifest: dict) -> None:
    by_day = {e["day"]: e for e in manifest["days"]}
    for prereq in by_day[47]["prerequisites"]:
        assert prereq < 47
    assert 44 in by_day[47]["prerequisites"]


def test_end_to_end_baseline_precedes_optimization(manifest: dict) -> None:
    """Profiling and every optimization must trace back to the Day 26 baseline.

    Leaves may depend on the profile (Day 27) instead of restating Day 26, so
    this walks the prerequisite graph instead of matching a single edge.
    """
    by_day = {e["day"]: e for e in manifest["days"]}

    def reaches_baseline(day: int, seen: set[int] | None = None) -> bool:
        seen = seen or set()
        if day in seen:
            return False
        seen.add(day)
        if day == 26:
            return True
        return any(reaches_baseline(p, seen) for p in by_day[day]["prerequisites"])

    for optimized in (27, 28, 29, 30, 31, 32, 33):
        assert reaches_baseline(optimized), optimized


def test_derived_documents_are_up_to_date() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "plan_docs.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("doc", PUBLIC_DOCS, ids=lambda p: p.name)
def test_public_documents_exclude_removed_scope(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    hits = {m.group(0) for m in STALE_SCOPE.finditer(text)}
    assert not hits, f"{doc.name} still advertises removed scope: {sorted(hits)}"


@pytest.mark.parametrize("doc", PUBLIC_DOCS, ids=lambda p: p.name)
def test_public_documents_exclude_motivation_language(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    hits = {m.group(0) for m in FORBIDDEN_MOTIVATION.finditer(text)}
    assert not hits, f"{doc.name} contains job-seeking language: {sorted(hits)}"


def test_package_description_matches_the_thesis() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "restoration" not in text.lower()

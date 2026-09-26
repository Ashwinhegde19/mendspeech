#!/usr/bin/env python3
"""Generate and verify the derived plan documents from the day specs.

The day files in ``docs/days/`` are the source of truth. This script writes
``docs/plan_manifest.json``, the eight week guides, the compiled 56-day
reference, and the two navigation indexes, then verifies the result.

Usage::

    python scripts/plan_docs.py --write    # regenerate derived documents
    python scripts/plan_docs.py --check    # verify without writing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
DAYS_DIR = DOCS / "days"
PLAN_PATH = DOCS / "plan_manifest.json"

LABELS = (
    "Learn",
    "Build in MendSpeech",
    "Experiment and Measure",
    "Required Output Artifacts",
)
STATUS_RE = re.compile(r"STATUS: (CORE|LEARN-ONLY|MERGED|DROPPED)")
TITLE_RE = re.compile(r"^# Day (\d+): (.+)$", re.M)
COMPUTE_RE = re.compile(r"### Compute Target\n`(.*?)`", re.S)
EFFORT_RE = re.compile(r"\*\*Effort:\*\* (\d+)\D(\d+)")
PREREQ_RE = re.compile(r"\*\*Prerequisites:\*\* (.+)")
PHASE_RE = re.compile(r"\*\*Phase:\*\* (P\d)", re.M)

PHASE_GOALS = {
    "P1": "Audio lab, deterministic damage suite, frozen labeled benchmark, CTC and ASR baseline.",
    "P2": "Data roles and protocol, confidence, timestamps, conservative editor contract, decoder comparison, early ASR-to-editor baseline, and SFT/GRPO feasibility.",
    "P3": "Streaming capability gate, chunk/session loop, endpointing, fixed-context frontier, calibration and triage, and an early end-to-end pipeline with a latency baseline.",
    "P4": "Profiling, compile/graph capture, batching, precision parity, cache-failure evidence, scorecard, and pipeline revalidation.",
    "P5": "Editor reward, SFT and compute-matched control, GRPO run, failure casebook, and a personalization feasibility decision.",
    "P6": "Serving contract, async service, load to saturation, editor selection under load, correlated latency budget, one optimization round, and a progress review.",
    "P7": "Frozen protocol, evaluation freeze, robustness matrix, ablations, final condition comparison, technical report, and reproduction guide.",
    "P8": "Final demo, clean reproduction, and tagged release.",
}
PHASE_RANGES = {"P1": (1, 9), "P2": (10, 16), "P3": (18, 26), "P4": (27, 33),
                "P5": (34, 42), "P6": (43, 49), "P7": (50, 55), "P8": (56, 56)}

WEEK_THEMES = {
    1: "Audio DSP and the deterministic damage suite",
    2: "Data protocol, confidence, editor contract, and the early end-to-end baseline",
    3: "Streaming capability, session loop, endpointing, and calibration",
    4: "Context trade-off, calibration, triage, and the end-to-end latency baseline",
    5: "Profiling, compile/graph capture, batching, precision, and the scorecard",
    6: "Editor reward, SFT and compute-matched control, GRPO, and ASR robustness adaptation",
    7: "Serving, load, editor selection, and the correlated latency budget",
    8: "Frozen evaluation, report, and release",
}

WEEK_BLURB = {
    1: "Audio foundations and the standalone damage suite are complete; treat this week as frozen evidence.",
    2: "Establish scoring, data roles, the conservative editor contract, and a measured end-to-end baseline before any optimization.",
    3: "Build and verify the streaming chunk loop, endpointing, and calibrated triage.",
    4: "Measure fixed context and produce a calibrated, end-to-end latency baseline.",
    5: "Optimize only what the profile shows, and record what does not help.",
    6: "Post-train the text editor with explicit controls, and adapt the recognizer for acoustic robustness.",
    7: "Ship one serving endpoint, measure the correlated latency budget, and make one evidence-driven optimization.",
    8: "Freeze the protocol, run the matrix and ablations, and write the report and reproduction guide.",
}

COMPILING_HEADER = """# MendSpeech Complete 56-Day Plan

> **Generated reference.** Rebuilt from the day specs by `scripts/plan_docs.py`.
> Read the [execution plan](REVISED_EXECUTION_PLAN.md) for scope, phases, and gates.
> Status banners override bodies: merged and dropped days do not create sessions
> or artifact obligations. Day numbers are specification identifiers, not
> calendar deadlines. Archived PDFs are unchanged.
"""

INDEX_NOTE = """> **Generated navigation.** Day numbers are specification identifiers, not
> calendar promises. Status banners override day bodies; optional drills are
> not release gates. Regenerate with `python scripts/plan_docs.py --write`.
"""


def parse_day(number: int) -> dict:
    path = DAYS_DIR / f"day_{number:02d}.md"
    text = path.read_text(encoding="utf-8")
    title = TITLE_RE.search(text)
    status = STATUS_RE.search(text)
    compute = COMPUTE_RE.search(text)
    effort = EFFORT_RE.search(text)
    prereq = PREREQ_RE.search(text)
    phase = PHASE_RE.search(text)
    entry = {
        "day": number,
        "title": title.group(2).strip() if title else path.stem,
        "status": status.group(1) if status else "CORE",
        "compute": compute.group(1).strip() if compute else "Local CPU",
    }
    if effort:
        entry["effort_hours"] = [int(effort.group(1)), int(effort.group(2))]
    if prereq:
        entry["prerequisites"] = [int(n) for n in re.findall(r"Day (\d+)", prereq.group(1))]
    else:
        entry["prerequisites"] = []
    if phase:
        entry["phase"] = phase.group(1)
    return entry


def phase_for(number: int) -> str:
    for name, (low, high) in PHASE_RANGES.items():
        if low <= number <= high:
            return name
    return "P1"


def blockquote(text: str) -> str:
    """Render text as a blockquote with no trailing whitespace per line."""
    lines = [f"> {line}".rstrip() for line in text.splitlines()]
    return "\n".join(lines)


def extract_sections(text: str) -> dict:
    out = {}
    for index, label in enumerate(LABELS, start=1):
        match = re.search(
            rf"^### {index}\. {re.escape(label)}\n(.*?)(?=^### |\Z)", text, re.M | re.S
        )
        if match:
            body = match.group(1).strip()
            body = re.sub(r"\n---\s*$", "", body).rstrip()
            out[label] = body
    done = re.search(r"^### 5\. Completion Check\n(.*?)(?=^### |\Z)", text, re.M | re.S)
    if done:
        body = done.group(1).strip()
        body = re.sub(r"\n---\s*$", "", body).rstrip()
        # Drop the "Definition of Done" label, then unwrap the blockquote so the
        # caller can re-render it uniformly for both the legacy day 01-09 format
        # and the generated format.
        body = re.sub(r"^>\s*\*\*Definition of Done[^*]*\*\*\s*\n", "", body, flags=re.M)
        body = "\n".join(
            re.sub(r"^>\s?", "", line).rstrip() for line in body.splitlines()
        ).strip()
        out["Completion Check"] = body
    return out


def build_manifest() -> dict:
    days = []
    for number in range(1, 57):
        entry = parse_day(number)
        if number >= 10:
            entry["phase"] = phase_for(number)
        days.append(entry)
    phases = []
    for name in sorted(PHASE_RANGES):
        low, high = PHASE_RANGES[name]
        members = [d for d in days if low <= d["day"] <= high and d["status"] == "CORE"]
        low_hours = sum(d.get("effort_hours", [0, 0])[0] for d in members)
        high_hours = sum(d.get("effort_hours", [0, 0])[1] for d in members)
        phases.append({
            "id": name,
            "days": [low, high],
            "goal": PHASE_GOALS[name],
            "core_sessions": len(members),
            "effort_hours": [low_hours, high_hours],
        })
    return {
        "version": 4,
        "description": (
            "Meaning-preserving dictation: streaming ASR, conservative transcript "
            "editing, robustness adaptation, bounded editor post-training, and a "
            "measured latency budget."
        ),
        "controlling_documents": {
            "execution_plan": "docs/REVISED_EXECUTION_PLAN.md",
            "blueprint": "docs/MendSpeech_Project_Blueprint.md",
            "timing_contract": "docs/LATENCY_AND_QUALITY_CONTRACT.md",
            "editor_rl_contract": "docs/EDITOR_AND_RL_CONTRACT.md",
        },
        "phases": phases,
        "days": days,
        "notes": [
            "Day numbers are specification identifiers, not calendar days.",
            "Effort ranges are focused hours including learning and tests, not deadlines.",
            "Day 16 gates whether editor post-training is feasible; it is not a claim of completed RL.",
            "A blocked required target is incomplete and requires an explicit scope decision.",
        ],
    }


def week_document(week: int, days: list[dict]) -> str:
    low = (week - 1) * 7 + 1
    high = low + 6
    rows = []
    for entry in days:
        if not low <= entry["day"] <= high:
            continue
        rows.append(
            f"| **Day {entry['day']:02d}** | {entry['title']} | "
            f"`{entry['compute']}` | {entry['status']} | "
            f"[Open Day {entry['day']:02d}](days/day_{entry['day']:02d}.md) |"
        )
    table = "\n".join(rows)
    body = [
        f"# Week {week}",
        "",
        f"> **Days {low:02d}–{high:02d}**",
        f"> **Navigation:** [← Index](INDEX.md) | [Master Index](INDEX.md) | "
        f"[Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | "
        f"[Executive Plan](REVISED_EXECUTION_PLAN.md)",
        "",
        "---",
        "",
        "> [!IMPORTANT]",
        f"> **Week theme:** {WEEK_THEMES[week]}",
        f"> {WEEK_BLURB[week]}",
        "",
        "---",
        "",
        "## Week Map",
        "",
        "| Day | Focus | Compute | Status | Daily Link |",
        "| :--- | :--- | :--- | :--- | :--- |",
        table,
        "",
        "---",
        "",
        "## Daily Detailed Operating Plans",
        "",
    ]
    for entry in days:
        if not low <= entry["day"] <= high:
            continue
        number = entry["day"]
        text = (DAYS_DIR / f"day_{number:02d}.md").read_text(encoding="utf-8")
        sections = extract_sections(text)
        block = [
            f"### DAY {number:02d}: {entry['title']}",
            f"- **Compute:** {entry['compute']}",
            f"- **Dedicated Daily File:** [`docs/days/day_{number:02d}.md`](days/day_{number:02d}.md)",
            "",
        ]
        status_line = STATUS_RE.search(text)
        if status_line:
            block.append(f"> **{status_line.group(0)}**")
        if entry["prerequisites"]:
            links = ", ".join(
                f"[Day {n:02d}](days/day_{n:02d}.md)" for n in entry["prerequisites"]
            )
            block.append(f"> **Prerequisites:** {links}")
        if entry.get("effort_hours"):
            low_h, high_h = entry["effort_hours"]
            block.append(f"> **Effort:** {low_h}–{high_h} focused hours.")
        block.append("")
        for label in LABELS:
            if label in sections:
                block.append(f"#### {label}")
                block.append(sections[label])
                block.append("")
        if "Completion Check" in sections:
            block.append("#### Completion Check")
            block.append(blockquote(sections["Completion Check"]))
            block.append("")
        block.append("---")
        block.append("")
        body.append("\n".join(block))
    return "\n".join(body).rstrip() + "\n"


def compiled_document(days: list[dict]) -> str:
    parts = [COMPILING_HEADER, "---"]
    for entry in days:
        number = entry["day"]
        if (number - 1) % 7 == 0:
            week = (number - 1) // 7 + 1
            name = f"Week_{week}_MendSpeech_Daily_Plan.md"
            parts.append(f"\n## Week {week}: {WEEK_THEMES[week]}\n")
            parts.append(f"[Week {week} guide]({name})\n")
        text = (DAYS_DIR / f"day_{number:02d}.md").read_text(encoding="utf-8")
        sections = extract_sections(text)
        parts.append(f"\n### Day {number:02d}: {entry['title']}\n")
        parts.append(f"[Full Day {number:02d} spec](days/day_{number:02d}.md)\n")
        parts.append(f"**Compute:** {entry['compute']}\n")
        status_line = STATUS_RE.search(text)
        if status_line:
            parts.append(f"\n> **{status_line.group(0)}**\n")
        for label in LABELS:
            if label in sections:
                parts.append(f"\n#### {label}\n")
                parts.append(sections[label] + "\n")
        if "Completion Check" in sections:
            parts.append("\n#### Completion Check\n")
            parts.append(blockquote(sections["Completion Check"]) + "\n")
    return "".join(parts).rstrip() + "\n"


def index_document() -> str:
    manifest = build_manifest()
    weeks = [
        ("Audio, Degradation, & Measurement Foundations",
         "Build the audio laboratory and release `SpeechDamageBench` as a standalone package."),
        ("Recognition, Editor Contract, & End-to-End Baseline",
         "Scoring, data roles, confidence, the conservative editor contract, decoder comparison, and the first ASR-to-editor baseline."),
        ("Streaming Capability, Session Loop, & Endpointing",
         "Verify streaming and cache support, build the chunk loop, and measure endpointing."),
        ("Context, Calibration, Triage, & Latency Baseline",
         "Fixed-lookahead trade-off, fitted calibration, triage policy, and a full pipeline latency baseline."),
        ("Inference Optimization",
         "Profile first, then compile/graph capture, batching, precision parity, cache-failure evidence, and a scorecard."),
        ("Editor Post-Training & ASR Robustness",
         "Reward design, SFT and compute-matched control, bounded GRPO, and acoustic robustness adaptation."),
        ("Serving, Load, & Correlated Latency Budget",
         "One endpoint, load to saturation, editor selection under load, and the per-stage budget."),
        ("Frozen Evaluation, Report, & Release",
         "Freeze the protocol, run the matrix and ablations, and publish the report and reproduction guide."),
    ]
    rows = []
    for week, (focus, milestone) in enumerate(weeks, start=1):
        low = (week - 1) * 7 + 1
        high = low + 6
        links = " • ".join(
            f"[Day {d:02d}](days/day_{d:02d}.md)" for d in range(low, high + 1)
        )
        rows.append(
            f"| **Week {week}** | {focus} | {milestone} | "
            f"[Week {week} Guide](Week_{week}_MendSpeech_Daily_Plan.md) | {links} |"
        )
    return "\n".join([
        "# MendSpeech Documentation Index",
        "",
        "Session specifications and engineering rules for the MendSpeech project.",
        "",
        INDEX_NOTE,
        "---",
        "",
        "## Controlling documents",
        "",
        "- [Revised Execution Plan](REVISED_EXECUTION_PLAN.md) — phases, gates, scope, and compute rules.",
        "- [Project Blueprint](MendSpeech_Project_Blueprint.md) — architecture, metrics, and definition of done.",
        "- [Latency and Quality Contract](LATENCY_AND_QUALITY_CONTRACT.md) — timing boundaries, workloads, and fair comparison rules.",
        "- [Editor and RL Contract](EDITOR_AND_RL_CONTRACT.md) — conservative editing limits, data, model route, and reward design.",
        "- [Plan manifest](plan_manifest.json) — machine-readable statuses, prerequisites, and effort ranges.",
        "- [Optional systems drills](SPEECH_ML_SYSTEMS_DRILLS.md) — study material, not release gates.",
        "",
        "---",
        "",
        "## 8-week / 56-specification progression",
        "",
        "| Week | Focus | Milestone | Weekly Plan | Daily Files |",
        "| :--- | :--- | :--- | :--- | :--- |",
        *rows,
        "",
        "---",
        "",
        "## Session totals",
        "",
    ] + [f"- **{p['id']}** (days {p['days'][0]:02d}–{p['days'][1]:02d}): "
          f"{p['core_sessions']} core sessions, {p['effort_hours'][0]}–{p['effort_hours'][1]} focused hours — {p['goal']}"
          for p in manifest["phases"]] + [
        "",
        f"Total core sessions: {sum(p['core_sessions'] for p in manifest['phases'])} "
        f"(days 01–09 complete). Remaining effort: "
        f"{sum(p['effort_hours'][0] for p in manifest['phases'])}–"
        f"{sum(p['effort_hours'][1] for p in manifest['phases'])} focused hours, "
        "including days 01–09. Estimate the remaining days from observed throughput.",
        "",
    ])


def render(manifest: dict) -> dict:
    days = manifest["days"]
    outputs = {
        PLAN_PATH: json.dumps(manifest, indent=2) + "\n",
        DOCS / "MendSpeech_Complete_56_Day_Plan.md": compiled_document(days),
        DOCS / "INDEX.md": index_document(),
        DOCS / "README.md": index_document(),
    }
    for week in range(1, 9):
        outputs[DOCS / f"Week_{week}_MendSpeech_Daily_Plan.md"] = week_document(week, days)
    return outputs


def verify(outputs: dict) -> list[str]:
    problems = []
    manifest = json.loads(outputs[PLAN_PATH])
    for day in manifest["days"]:
        for prereq in day["prerequisites"]:
            if prereq >= day["day"]:
                problems.append(
                    f"day {day['day']:02d} lists prerequisite day {prereq:02d}, "
                    "which is not earlier in the sequence"
                )
    compiled = outputs[DOCS / "MendSpeech_Complete_56_Day_Plan.md"]
    for day in manifest["days"]:
        if f"### Day {day['day']:02d}: {day['title']}" not in compiled:
            problems.append(f"compiled reference missing day {day['day']:02d} heading")
    for week in range(1, 9):
        document = outputs[DOCS / f"Week_{week}_MendSpeech_Daily_Plan.md"]
        low = (week - 1) * 7 + 1
        for day in range(low, low + 7):
            if f"### DAY {day:02d}:" not in document:
                problems.append(f"week {week} guide missing day {day:02d} section")
    if outputs[DOCS / "INDEX.md"] != outputs[DOCS / "README.md"]:
        problems.append("docs/INDEX.md and docs/README.md diverged")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest()
    outputs = render(manifest)
    problems = verify(outputs)

    if args.write:
        for path, content in outputs.items():
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                path.write_text(content, encoding="utf-8")
        stale = [
            path for path in outputs
            if not path.exists() or path.read_text(encoding="utf-8") != content
        ]
        if problems:
            print("Wrote with unresolved problems:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"Wrote {len(outputs)} derived documents.")
        return 0

    drift = [
        path for path, content in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
    if drift:
        print("Out of date; run: python scripts/plan_docs.py --write", file=sys.stderr)
        for path in sorted(drift):
            print(f"  - {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"Plan documents are consistent ({len(outputs)} files, version {manifest['version']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Declared data roles and leakage checks for the frozen benchmark.

Day 10, concepts 5 and 6. Four roles are declared for the frozen speech
benchmark — training, calibration, validation and final test — and this
module both records them and checks that the assignment is actually
speaker-separated. The split column in the manifest carries three roles
today; :data:`CALIBRATION_SPEAKER` names the speaker reassigned from training
to calibration so the fourth role exists before anything downstream is tuned.

Two rules make the checks worth running:

- **Roles are a property of speakers, not of rows.** Splitting utterances at
  random would put the same voice in training and test, and the resulting
  numbers would look clean while measuring memorization. Every check here
  operates on speaker IDs.
- **Repeated corruption is not an independent sample.** Scoring six
  corruptions of one utterance produces six rows but one observation. Any
  claim about statistical significance must count utterances and speakers,
  never rows, or it will be overstated by the corruption multiplier.

Nothing here mutates the benchmark. These are read-only audits whose output
belongs in reports/data_roles.md, plus the checksum that makes the freeze
verifiable.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

# The canonical manifest for the frozen benchmark. The byte-identical copy
# under data/benchmark/ is a legacy duplicate and is not the source of truth.
CANONICAL_MANIFEST = Path("data/benchmark_manifest.csv")

# Speaker reassigned from training to calibration, so the four declared roles
# are disjoint by speaker. Costs 6 of 18 training utterances; recorded rather
# than hidden, because downstream training is thin and may be blocked.
CALIBRATION_SPEAKER = "spk_1673"

# Role each manifest split maps to. The manifest has three splits; the
# training split is subdivided by speaker to create the calibration role.
SPLIT_TO_ROLE = {
    "train": "train",
    "val": "validation",
    "test": "final_test",
}

# The four roles required by docs/days/day_10.md.
DECLARED_ROLES = ("train", "calibration", "validation", "final_test")

# Corruption families used by SpeechDamageBench, recorded so the row-count
# inflation in the leakage report has a concrete multiplier.
CORRUPTION_FAMILIES = (
    "additive_noise",
    "clipping",
    "bandwidth",
    "dropout",
    "reverberation",
)


@dataclass
class RoleAssignment:
    """Declared role, speaker membership and utterance count for one role.

    Attributes:
        role: One of :data:`DECLARED_ROLES`.
        manifest_split: The manifest ``split`` value this role draws from.
        speakers: Speaker IDs assigned to the role.
        num_utterances: Utterances in the role.
        source: How the assignment was derived, recorded so a later reader
            can tell a measured split from a declared one.
    """

    role: str
    manifest_split: str
    speakers: List[str]
    num_utterances: int
    source: str

    def to_dict(self) -> dict:
        """Return a flat dict suitable for csv.DictWriter."""
        return asdict(self)


@dataclass
class LeakageReport:
    """Result of the leakage and grouping audit.

    Attributes:
        speaker_overlaps: Pairs of roles sharing at least one speaker. Must
            be empty for a valid speaker-separated assignment.
        duplicate_transcripts: Transcripts appearing more than once.
        cross_role_transcripts: Transcripts whose reference words occur in
            more than one role.
        independent_utterances: Distinct utterances in the corpus. This, not
            the row count, is the sample size for any statistical claim.
        independent_speakers: Distinct speakers in the corpus.
        scored_rows_if_fully_corrupted: Rows a clean-plus-corruption sweep
            would produce, kept to show how far it exceeds the true sample.
        checks_passed: True when no speaker overlap and no duplicate
            transcript was found.
        notes: Human-readable observations for the report.
    """

    speaker_overlaps: List[str]
    duplicate_transcripts: List[str]
    cross_role_transcripts: List[str]
    independent_utterances: int
    independent_speakers: int
    scored_rows_if_fully_corrupted: int
    checks_passed: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a flat dict suitable for csv.DictWriter."""
        return asdict(self)


def file_checksum(path: Path = CANONICAL_MANIFEST) -> str:
    """Return the SHA-256 of a file, making the freeze verifiable.

    Args:
        path: File to hash. Defaults to the canonical benchmark manifest.

    Returns:
        Lowercase hexadecimal SHA-256 digest of the file contents.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_roles(manifest: pd.DataFrame) -> Dict[str, RoleAssignment]:
    """Assign the four declared roles from the frozen manifest.

    The manifest carries train, val and test. The calibration role is carved
    out of training by speaker (:data:`CALIBRATION_SPEAKER`) so that all four
    roles are disjoint by speaker and none of them shares a voice.

    Args:
        manifest: Benchmark manifest with ``split``, ``speaker_id`` and
            ``clip_id`` columns.

    Returns:
        Mapping of role name to :class:`RoleAssignment`, in the order of
        :data:`DECLARED_ROLES`.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If the calibration speaker is absent from the manifest.
    """
    required = {"split", "speaker_id", "clip_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"manifest missing columns: {sorted(missing)}")

    train = manifest[manifest["split"] == "train"]
    if CALIBRATION_SPEAKER not in set(train["speaker_id"]):
        raise ValueError(
            f"calibration speaker {CALIBRATION_SPEAKER!r} not in training split"
        )

    calibration = train[train["speaker_id"] == CALIBRATION_SPEAKER]
    train_only = train[train["speaker_id"] != CALIBRATION_SPEAKER]

    def speakers_of(frame: pd.DataFrame) -> List[str]:
        return sorted(frame["speaker_id"].unique().tolist())

    return {
        "train": RoleAssignment(
            role="train",
            manifest_split="train",
            speakers=speakers_of(train_only),
            num_utterances=int(len(train_only)),
            source=f"manifest train minus calibration speaker {CALIBRATION_SPEAKER}",
        ),
        "calibration": RoleAssignment(
            role="calibration",
            manifest_split="train",
            speakers=speakers_of(calibration),
            num_utterances=int(len(calibration)),
            source=f"manifest train, speaker {CALIBRATION_SPEAKER} only",
        ),
        "validation": RoleAssignment(
            role="validation",
            manifest_split="val",
            speakers=speakers_of(manifest[manifest["split"] == "val"]),
            num_utterances=int((manifest["split"] == "val").sum()),
            source="manifest val, unchanged",
        ),
        "final_test": RoleAssignment(
            role="final_test",
            manifest_split="test",
            speakers=speakers_of(manifest[manifest["split"] == "test"]),
            num_utterances=int((manifest["split"] == "test").sum()),
            source="manifest test, sealed until the final frozen release",
        ),
    }


def role_for_clip(manifest: pd.DataFrame) -> Dict[str, str]:
    """Map each clip ID to its declared role.

    Args:
        manifest: Benchmark manifest with ``clip_id``, ``split`` and
            ``speaker_id`` columns.

    Returns:
        Mapping of clip ID to role name, for joining scored results back to
        the role they belong to.
    """
    mapping: Dict[str, str] = {}
    for _, row in manifest.iterrows():
        if row["split"] == "train":
            role = (
                "calibration"
                if row["speaker_id"] == CALIBRATION_SPEAKER
                else "train"
            )
        else:
            role = SPLIT_TO_ROLE[row["split"]]
        mapping[row["clip_id"]] = role
    return mapping


def check_leakage(
    manifest: pd.DataFrame, *, corruptions: int = len(CORRUPTION_FAMILIES)
) -> LeakageReport:
    """Audit the manifest for speaker overlap, duplicates and grouping traps.

    Checks three things a reader would otherwise have to trust: that no
    speaker appears in two roles, that no reference transcript is duplicated
    or shared across roles, and how far a corruption sweep inflates the row
    count beyond the true number of independent observations.

    Args:
        manifest: Benchmark manifest with ``split``, ``speaker_id``,
            ``clip_id`` and ``transcript`` columns.
        corruptions: Number of corruption families a sweep would apply, used
            to report the row-count inflation. Includes the clean condition
            plus ``corruptions`` damaged conditions.

    Returns:
        LeakageReport with the findings. ``checks_passed`` is False if any
        speaker overlap or duplicate transcript was found.
    """
    assignments = assign_roles(manifest)
    clip_roles = role_for_clip(manifest)

    # Speaker overlap between every pair of roles.
    speaker_sets = {role: set(a.speakers) for role, a in assignments.items()}
    overlaps: List[str] = []
    for left, right in itertools.combinations(speaker_sets, 2):
        shared = speaker_sets[left] & speaker_sets[right]
        if shared:
            overlaps.append(f"{left}/{right}: {sorted(shared)}")

    duplicates = sorted(
        manifest[manifest["transcript"].duplicated(keep=False)]["transcript"]
        .unique()
        .tolist()
    )

    # Transcript overlap across roles, judged on normalized reference words.
    from src.metrics.normalization import normalize_text

    words_by_role: Dict[str, set] = {role: set() for role in assignments}
    for _, row in manifest.iterrows():
        words_by_role[clip_roles[row["clip_id"]]].add(
            frozenset(normalize_text(row["transcript"]).split())
        )
    cross_role: List[str] = []
    for left, right in itertools.combinations(words_by_role, 2):
        if words_by_role[left] & words_by_role[right]:
            cross_role.append(f"{left}/{right}")

    n_utterances = int(len(manifest))
    n_rows = n_utterances * (corruptions + 1)

    notes = [
        f"{n_utterances} utterances across "
        f"{manifest['speaker_id'].nunique()} speakers; a full "
        f"{corruptions}-corruption sweep yields {n_rows} scored rows, "
        f"a {n_rows / n_utterances:.1f}x inflation of the true sample size.",
        "Report per-utterance aggregates; never treat corruption rows as "
        "independent observations in a significance claim.",
    ]
    if not assignments["calibration"].speakers:
        notes.append("Calibration role is empty; confidence fitting is blocked.")

    return LeakageReport(
        speaker_overlaps=overlaps,
        duplicate_transcripts=duplicates,
        cross_role_transcripts=cross_role,
        independent_utterances=n_utterances,
        independent_speakers=int(manifest["speaker_id"].nunique()),
        scored_rows_if_fully_corrupted=n_rows,
        checks_passed=not overlaps and not duplicates,
        notes=notes,
    )

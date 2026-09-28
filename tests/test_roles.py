"""Tests for declared data roles and leakage checks (Day 10, concepts 5-6)."""

import pandas as pd
import pytest

from src.metrics.roles import (
    CALIBRATION_SPEAKER,
    CANONICAL_MANIFEST,
    DECLARED_ROLES,
    LeakageReport,
    RoleAssignment,
    assign_roles,
    check_leakage,
    file_checksum,
    role_for_clip,
)

MANIFEST = "data/benchmark_manifest.csv"


@pytest.fixture(scope="module")
def manifest():
    return pd.read_csv(MANIFEST)


class TestRoleAssignment:
    """The four declared roles and their speaker membership."""

    def test_all_four_roles_declared(self, manifest):
        assert set(assign_roles(manifest)) == set(DECLARED_ROLES)

    def test_roles_partition_the_corpus(self, manifest):
        assignments = assign_roles(manifest)
        total = sum(a.num_utterances for a in assignments.values())
        assert total == len(manifest)

    def test_roles_are_speaker_disjoint(self, manifest):
        assignments = assign_roles(manifest)
        seen = set()
        for role, assignment in assignments.items():
            for speaker in assignment.speakers:
                assert speaker not in seen, f"{speaker} in multiple roles"
                seen.add(speaker)

    def test_calibration_role_is_populated(self, manifest):
        calibration = assign_roles(manifest)["calibration"]
        assert calibration.speakers == [CALIBRATION_SPEAKER]
        assert calibration.num_utterances > 0

    def test_test_role_is_sealed_and_unchanged(self, manifest):
        final_test = assign_roles(manifest)["final_test"]
        assert final_test.manifest_split == "test"
        assert (manifest["split"] == "test").sum() == final_test.num_utterances

    def test_training_data_cost_is_recorded(self, manifest):
        """Carving calibration from train is a real, declared cost."""
        assignments = assign_roles(manifest)
        train = manifest[manifest["split"] == "train"]
        assert assignments["train"].num_utterances == len(train) - 6
        assert "calibration speaker" in assignments["train"].source

    def test_missing_column_raises(self, manifest):
        with pytest.raises(KeyError, match="missing columns"):
            assign_roles(manifest.drop(columns=["speaker_id"]))

    def test_unknown_calibration_speaker_raises(self, manifest):
        import src.metrics.roles as roles_module

        original = roles_module.CALIBRATION_SPEAKER
        roles_module.CALIBRATION_SPEAKER = "spk_does_not_exist"
        try:
            with pytest.raises(ValueError, match="not in training split"):
                roles_module.assign_roles(manifest)
        finally:
            roles_module.CALIBRATION_SPEAKER = original

    def test_assignment_to_dict(self, manifest):
        record = assign_roles(manifest)["train"]
        assert isinstance(record, RoleAssignment)
        assert set(record.to_dict()) == {
            "role",
            "manifest_split",
            "speakers",
            "num_utterances",
            "source",
        }


class TestRoleForClip:
    """Mapping scored results back to their role."""

    def test_every_clip_has_a_role(self, manifest):
        mapping = role_for_clip(manifest)
        assert len(mapping) == len(manifest)
        assert set(mapping.values()) <= set(DECLARED_ROLES)

    def test_role_matches_assignment_speakers(self, manifest):
        mapping = role_for_clip(manifest)
        assignments = assign_roles(manifest)
        speakers = dict(zip(manifest["clip_id"], manifest["speaker_id"]))
        for role, assignment in assignments.items():
            for clip_id, assigned_role in mapping.items():
                if assigned_role == role:
                    assert speakers[clip_id] in assignment.speakers

    def test_no_clip_in_two_roles(self, manifest):
        mapping = role_for_clip(manifest)
        assert len(set(mapping)) == len(mapping)



class TestLeakageChecks:
    """Speaker overlap, duplicates and the grouping trap."""

    def test_frozen_benchmark_passes_leakage(self, manifest):
        assert check_leakage(manifest).checks_passed is True

    def test_no_speaker_overlap(self, manifest):
        assert check_leakage(manifest).speaker_overlaps == []

    def test_no_duplicate_transcripts(self, manifest):
        assert check_leakage(manifest).duplicate_transcripts == []

    def test_no_transcript_shared_across_roles(self, manifest):
        assert check_leakage(manifest).cross_role_transcripts == []

    def test_independent_sample_size_is_utterances(self, manifest):
        report = check_leakage(manifest)
        assert report.independent_utterances == len(manifest)
        assert report.independent_speakers == manifest["speaker_id"].nunique()

    def test_corruption_rows_inflate_sample_size(self, manifest):
        """The row count must be visibly larger than the true sample."""
        report = check_leakage(manifest)
        assert report.scored_rows_if_fully_corrupted > report.independent_utterances
        assert any("independent observations" in n for n in report.notes)

    def test_detects_injected_speaker_overlap(self, manifest):
        """A role that borrows a test speaker must be caught."""
        leaky = manifest.copy()
        leaky.loc[leaky["speaker_id"] == "spk_1272", "speaker_id"] = "spk_1988"
        report = check_leakage(leaky)
        assert report.checks_passed is False
        assert any("1988" in o for o in report.speaker_overlaps)

    def test_detects_injected_duplicate_transcript(self, manifest):
        dupe = manifest.copy()
        dupe.loc[dupe.index[1], "transcript"] = dupe.loc[dupe.index[0], "transcript"]
        report = check_leakage(dupe)
        assert report.checks_passed is False
        assert report.duplicate_transcripts

    def test_report_to_dict(self, manifest):
        record = check_leakage(manifest)
        assert isinstance(record, LeakageReport)
        assert set(record.to_dict()) == {
            "speaker_overlaps",
            "duplicate_transcripts",
            "cross_role_transcripts",
            "independent_utterances",
            "independent_speakers",
            "scored_rows_if_fully_corrupted",
            "checks_passed",
            "notes",
        }


class TestFreezeChecksum:
    """The freeze must be verifiable, not merely asserted."""

    def test_canonical_manifest_hash_is_stable(self):
        assert file_checksum() == file_checksum()

    def test_hash_is_sha256_hex(self):
        digest = file_checksum()
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_legacy_duplicate_is_byte_identical(self):
        """The legacy copy must not silently drift from the canonical file."""
        import hashlib

        with open("data/benchmark/manifest.csv", "rb") as handle:
            legacy_digest = hashlib.sha256(handle.read()).hexdigest()
        assert legacy_digest == file_checksum()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            file_checksum(tmp_path / "nope.csv")

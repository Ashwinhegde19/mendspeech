"""Optional unit test for the Modal dry-run path."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("modal") is None,
    reason="Modal is not installed in this environment",
)


def test_modal_dry_run_does_not_launch_gpu(tmp_path: Path) -> None:
    from infra.modal_hinglish_asr import dry_run

    audio_path = tmp_path / "smoke.wav"
    audio_path.write_bytes(b"not-audio" * 10)
    assert dry_run(audio_path, "T4", "smoke", "hi", "transcribe") == 0

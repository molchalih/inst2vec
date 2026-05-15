"""End-to-end VAD checks against the real fixtures in data/audio/.

Skipped when ffmpeg or silero-vad isn't available.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURE_DIR = Path("data/audio")
SPEECH = FIXTURE_DIR / "test_audio_yes_speech.mp3"
SILENCE = FIXTURE_DIR / "test_audio_no_speech.mp3"

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and SPEECH.exists() and SILENCE.exists()),
    reason="ffmpeg and data/audio/* fixtures are required",
)


@pytest.fixture(autouse=True)
def _reset_model():
    """Reset the global silero-vad model cache before each test."""
    import modules.speech.vad as vad_module

    vad_module._MODEL = None
    yield
    vad_module._MODEL = None


def _cfg():
    from modules.speech.vad import VadConfig

    return VadConfig(
        enabled=True,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=150,
        min_total_speech_s=0.5,
    )


def test_real_speech_audio_detected(tmp_path):
    from modules.speech.vad import prepare_for_whisper

    res = prepare_for_whisper(SPEECH, tmp_path, _cfg())
    assert res.is_speech_detected is True
    assert res.speech_audio_path is not None
    assert res.speech_audio_path.exists()
    assert res.total_speech_seconds > 0.5
    assert len(res.segments) >= 1


def test_real_silent_audio_rejected(tmp_path):
    from modules.speech.vad import prepare_for_whisper

    res = prepare_for_whisper(SILENCE, tmp_path, _cfg())
    assert res.is_speech_detected is False
    assert res.speech_audio_path is None
    assert res.total_speech_seconds < 0.5

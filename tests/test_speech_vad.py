"""Tests for modules/speech/vad.py."""

from pathlib import Path

from modules.speech.vad import VadConfig, prepare_for_whisper


def test_vad_config_defaults_disabled():
    cfg = VadConfig()
    assert cfg.enabled is False


def test_prepare_for_whisper_returns_input_when_disabled(tmp_path: Path):
    video = tmp_path / "10.mp4"
    video.write_bytes(b"fake")
    out = prepare_for_whisper(video, tmp_path, VadConfig())
    assert out == video


def test_prepare_for_whisper_returns_none_for_missing_input(tmp_path: Path):
    missing = tmp_path / "nope.mp4"
    out = prepare_for_whisper(missing, tmp_path, VadConfig(enabled=True))
    assert out is None

"""Unit tests for modules/speech/vad.py (mock backend)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from modules.speech.vad import VadConfig, VadResult, prepare_for_whisper


def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    """Write a mono PCM16 WAV with stdlib `wave` for use as ffmpeg input."""
    import wave

    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _silence(seconds: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def _stub_silero(monkeypatch, timestamps):
    """Patch silero-vad entry points used by the helper."""
    model = MagicMock(name="silero_model")
    fake = SimpleNamespace(
        load_silero_vad=lambda: model,
        get_speech_timestamps=lambda *a, **k: list(timestamps),
    )
    monkeypatch.setattr("modules.speech.vad._silero", fake)
    return model


def test_vad_config_defaults_match_config_block():
    cfg = VadConfig(
        enabled=True,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=150,
        min_total_speech_s=0.5,
    )
    assert cfg.enabled is True
    assert cfg.sampling_rate == 16000


def test_disabled_returns_passthrough(tmp_path):
    media = tmp_path / "10.wav"
    _write_wav(media, _silence(1.0))
    out = prepare_for_whisper(
        media, tmp_path, VadConfig(enabled=False, sampling_rate=16000)
    )
    assert isinstance(out, VadResult)
    assert out.is_speech_detected is True
    assert out.speech_audio_path == media
    assert out.segments == []
    assert out.total_speech_seconds == 0.0


def test_no_speech_segments_returns_false(tmp_path, monkeypatch):
    media = tmp_path / "10.wav"
    _write_wav(media, _silence(2.0))
    _stub_silero(monkeypatch, [])
    cfg = VadConfig(
        enabled=True,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=150,
        min_total_speech_s=0.5,
    )
    res = prepare_for_whisper(media, tmp_path, cfg)
    assert res.is_speech_detected is False
    assert res.speech_audio_path is None
    assert res.segments == []
    assert res.total_speech_seconds == 0.0


def test_below_min_total_speech_returns_false(tmp_path, monkeypatch):
    media = tmp_path / "10.wav"
    _write_wav(media, _silence(2.0))
    # 100 ms of detected speech — below 500 ms gate
    sr = 16000
    _stub_silero(monkeypatch, [{"start": 0, "end": int(0.1 * sr)}])
    cfg = VadConfig(
        enabled=True,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=150,
        min_total_speech_s=0.5,
    )
    res = prepare_for_whisper(media, tmp_path, cfg)
    assert res.is_speech_detected is False
    assert res.speech_audio_path is None


def test_speech_segments_write_concatenated_wav(tmp_path, monkeypatch):
    media = tmp_path / "10.wav"
    sr = 16000
    # 3 s of low-amplitude tone so concat output is non-empty
    t = np.arange(3 * sr) / sr
    _write_wav(media, 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32), sr)
    _stub_silero(
        monkeypatch,
        [
            {"start": int(0.2 * sr), "end": int(0.8 * sr)},
            {"start": int(1.2 * sr), "end": int(2.0 * sr)},
        ],
    )
    cfg = VadConfig(
        enabled=True,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=150,
        min_total_speech_s=0.5,
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    res = prepare_for_whisper(media, out_dir, cfg)
    assert res.is_speech_detected is True
    assert res.speech_audio_path is not None
    assert res.speech_audio_path.exists()
    assert res.speech_audio_path.parent == out_dir
    assert res.speech_audio_path.suffix == ".wav"
    assert len(res.segments) == 2
    assert pytest.approx(res.total_speech_seconds, abs=0.05) == 1.4  # 0.6 + 0.8


def test_ffmpeg_failure_raises(tmp_path, monkeypatch):
    media = tmp_path / "broken.wav"
    media.write_bytes(b"not-a-wav")
    cfg = VadConfig(enabled=True, sampling_rate=16000)
    # Force ffmpeg-runner to return False
    monkeypatch.setattr("modules.speech.vad._run_ffmpeg", lambda *a, **k: False)
    with pytest.raises(RuntimeError):
        prepare_for_whisper(media, tmp_path, cfg)

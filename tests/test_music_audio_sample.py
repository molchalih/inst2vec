"""Tests for modules.music.audio_sample.extract_audio_sample."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from modules.config import MusicSettings
from modules.music.audio_sample import extract_audio_sample


def _settings(**overrides) -> MusicSettings:
    base = dict(
        audio_fingerprint_confidence=0.8,
        commit_every=50,
        http_timeout=20.0,
        spotify_search_limit=5,
        spotify_token_skew_seconds=30,
        spotify_request_timeout=8.0,
        reccobeats_batch_size=20,
        reccobeats_delay_min=0.0,
        reccobeats_delay_max=0.0,
        manual_features_max_seconds=20,
        manual_features_sample_rate=44100,
        manual_features_max_mb=5.0,
        manual_features_mp3_bitrate="128k",
        api_max_attempts=3,
        api_retry_delay=0.0,
        api_retry_jitter=0.0,
        acr_max_attempts=2,
        ffmpeg_timeout_seconds=60,
    )
    base.update(overrides)
    return MusicSettings(**base)


def test_extract_audio_sample_returns_wav_on_success(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    wav = tmp_path / "out" / "v.wav"
    wav.parent.mkdir()

    def fake_run(cmd, **kwargs):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"\x00" * 100)
        r = MagicMock()
        r.returncode = 0
        return r

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = extract_audio_sample(video, wav.parent, _settings())
    assert result is not None
    assert result.suffix == ".wav"


def test_extract_audio_sample_falls_back_to_mp3_when_wav_too_large(
    tmp_path, monkeypatch
):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()

    def fake_run(cmd, **kwargs):
        out_path = Path(cmd[-1])
        if out_path.suffix == ".wav":
            out_path.write_bytes(b"\x00" * 200_000)  # huge
        else:
            out_path.write_bytes(b"\x00" * 50)  # small
        r = MagicMock()
        r.returncode = 0
        return r

    monkeypatch.setattr(subprocess, "run", fake_run)
    settings = _settings(manual_features_max_mb=0.0001)  # ~100 bytes
    result = extract_audio_sample(video, out, settings)
    assert result is not None
    assert result.suffix == ".mp3"


def test_extract_audio_sample_returns_none_when_ffmpeg_times_out(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 60))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert extract_audio_sample(video, out, _settings()) is None


def test_extract_audio_sample_returns_none_when_ffmpeg_fails(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()

    def fake_run(cmd, **kwargs):
        r = MagicMock()
        r.returncode = 1
        return r

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert extract_audio_sample(video, out, _settings()) is None


def test_extract_audio_sample_passes_timeout_to_subprocess(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"\x00" * 100)
        r = MagicMock()
        r.returncode = 0
        return r

    monkeypatch.setattr(subprocess, "run", fake_run)
    extract_audio_sample(video, out, _settings(ffmpeg_timeout_seconds=42))
    assert captured["timeout"] == 42

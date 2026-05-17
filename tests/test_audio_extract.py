import os
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    Music,
    StageState,
    User,
    get_engine,
    get_session,
)
from modules.ingest.audio import extract_audio


def test_extracts_mp3_from_mp4(sample_mp4_with_audio, tmp_path):
    out = tmp_path / "1.mp3"
    ok = extract_audio(
        str(sample_mp4_with_audio),
        str(out),
        bitrate_kbps=128,
        sample_rate_hz=44100,
        timeout_s=60,
    )
    assert ok
    assert out.exists()
    assert out.stat().st_size > 5_000  # non-empty mp3


def test_skip_when_audio_newer_than_video(sample_mp4_with_audio, tmp_path):
    out = tmp_path / "1.mp3"
    assert extract_audio(
        str(sample_mp4_with_audio),
        str(out),
        bitrate_kbps=128,
        sample_rate_hz=44100,
        timeout_s=60,
    )
    first_mtime = out.stat().st_mtime_ns
    time.sleep(0.05)
    # Second call must be a no-op (no ffmpeg run, mtime unchanged).
    assert extract_audio(
        str(sample_mp4_with_audio),
        str(out),
        bitrate_kbps=128,
        sample_rate_hz=44100,
        timeout_s=60,
    )
    assert out.stat().st_mtime_ns == first_mtime


def test_re_extracts_when_video_newer(sample_mp4_with_audio, tmp_path):
    # Copy the fixture so we can bump its mtime without affecting other tests.
    src = tmp_path / "video.mp4"
    src.write_bytes(Path(sample_mp4_with_audio).read_bytes())
    out = tmp_path / "1.mp3"
    extract_audio(
        str(src), str(out), bitrate_kbps=128, sample_rate_hz=44100, timeout_s=60
    )
    first_mtime = out.stat().st_mtime_ns
    # Bump video mtime to simulate re-download.
    future = time.time() + 10
    os.utime(src, (future, future))
    time.sleep(0.05)
    extract_audio(
        str(src), str(out), bitrate_kbps=128, sample_rate_hz=44100, timeout_s=60
    )
    assert out.exists()
    # File was re-extracted (mtime advanced past the original).
    assert out.stat().st_mtime_ns > first_mtime


# ── stage tests ──────────────────────────────────────────────────────────────


@dataclass
class _PathsStub:
    video_dir: str
    audio_dir: str


@dataclass
class _EmbeddingsStub:
    gemini_enabled: bool = False
    audio_bitrate_kbps: int = 128
    audio_sample_rate_hz: int = 44100
    audio_extract_timeout_s: int = 60


@dataclass
class _SettingsStub:
    paths: _PathsStub
    embeddings: _EmbeddingsStub


def _make_settings(*, audio_dir, enabled: bool, video_dir) -> _SettingsStub:
    return _SettingsStub(
        paths=_PathsStub(video_dir=str(video_dir), audio_dir=str(audio_dir)),
        embeddings=_EmbeddingsStub(gemini_enabled=enabled),
    )


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, ClipEmbedding, Clip, Music, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


def test_disabled_short_circuits(tmp_path, db_session):
    from modules.ingest.audio import extract_audio_stage

    audio_dir = tmp_path / "audio"
    video_dir = tmp_path / "video"
    settings = _make_settings(audio_dir=audio_dir, enabled=False, video_dir=video_dir)
    with patch("modules.ingest.audio.run_ffmpeg") as ff:
        extract_audio_stage(settings)
    ff.assert_not_called()
    assert audio_dir.exists() is False
    assert db_session.get(StageState, ("audio_extract", "default")) is None


def test_stage_fingerprint_seals(tmp_path, sample_mp4_with_audio, db_session):
    from modules.ingest.audio import extract_audio_stage

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    target = vid_dir / "1.mp4"
    target.write_bytes(Path(sample_mp4_with_audio).read_bytes())

    db_session.add(User(id=1, is_selected=True))
    db_session.add(Clip(id=1, user_id=1, is_selected=True, is_downloaded=True))
    db_session.commit()

    audio_dir = tmp_path / "audio"
    settings = _make_settings(audio_dir=audio_dir, enabled=True, video_dir=vid_dir)
    extract_audio_stage(settings)

    assert (audio_dir / "1.mp3").exists()
    row = db_session.get(StageState, ("audio_extract", "default"))
    assert row is not None  # sealed

    # Second run is a no-op (fingerprint matches → no ffmpeg invocation).
    with patch("modules.ingest.audio.run_ffmpeg") as ff:
        extract_audio_stage(settings)
    ff.assert_not_called()

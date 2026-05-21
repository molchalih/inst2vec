"""End-to-end tests for extract_audio_mir_stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from core.database import (
    Base,
    Clip,
    StageState,
    User,
    get_engine,
    get_session,
)


@dataclass
class _PathsStub:
    video_dir: str
    audio_mir_dir: str

    def video_for(self, clip_id):
        return Path(self.video_dir) / f"{clip_id}.mp4"

    def audio_mir_for(self, clip_id):
        return Path(self.audio_mir_dir) / f"{clip_id}.wav"


@dataclass
class _AudioExtractionStub:
    mir_codec: str = "pcm_s16le"
    mir_extension: str = "wav"
    mir_sample_rate_hz: int = 44_100
    mir_channels: int = 2
    mir_extract_timeout_s: int = 60


@dataclass
class _DownloadStub:
    concurrency: int = 2


@dataclass
class _SettingsStub:
    paths: _PathsStub
    audio_extraction: _AudioExtractionStub
    download: _DownloadStub


def _make_settings(*, video_dir, audio_mir_dir) -> _SettingsStub:
    return _SettingsStub(
        paths=_PathsStub(video_dir=str(video_dir), audio_mir_dir=str(audio_mir_dir)),
        audio_extraction=_AudioExtractionStub(),
        download=_DownloadStub(),
    )


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, Clip, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


def test_extract_audio_mir_stage_writes_wav(
    sample_mp4_with_audio, tmp_path, db_session
):
    from modules.ingest.audio import extract_audio_mir_stage

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True))
    db_session.commit()

    audio_mir_dir = tmp_path / "audio_mir"
    settings = _make_settings(video_dir=vid_dir, audio_mir_dir=audio_mir_dir)

    extract_audio_mir_stage(settings)

    out = audio_mir_dir / "1.wav"
    assert out.exists()
    assert out.read_bytes()[:4] == b"RIFF"
    # Fingerprint was sealed (no failures)
    assert db_session.get(StageState, ("audio_extract_mir", "default")) is not None


def test_extract_audio_mir_stage_logs_err_when_video_missing(tmp_path, db_session):
    from modules.ingest.audio import extract_audio_mir_stage

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()

    db_session.add(User(id=1))
    db_session.add(Clip(id=99, user_id=1, is_downloaded=True))
    db_session.commit()

    audio_mir_dir = tmp_path / "audio_mir"
    settings = _make_settings(video_dir=vid_dir, audio_mir_dir=audio_mir_dir)

    extract_audio_mir_stage(settings)

    # No WAV produced; fingerprint not sealed (failure path).
    assert not (audio_mir_dir / "99.wav").exists()
    assert db_session.get(StageState, ("audio_extract_mir", "default")) is None


def test_extract_audio_mir_stage_reencodes_on_config_drift(
    sample_mp4_with_audio, tmp_path, db_session
):
    """Config change re-runs extraction and rewrites WAV files (no mtime skip)."""
    from modules.ingest.audio import extract_audio_mir_stage

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True))
    db_session.commit()

    audio_mir_dir = tmp_path / "audio_mir"
    settings = _make_settings(video_dir=vid_dir, audio_mir_dir=audio_mir_dir)

    extract_audio_mir_stage(settings)
    out = audio_mir_dir / "1.wav"
    mtime1 = out.stat().st_mtime_ns
    size1 = out.stat().st_size

    # Drift only the config: lower sample rate. Output must be rewritten.
    settings.audio_extraction.mir_sample_rate_hz = 22_050
    extract_audio_mir_stage(settings)

    assert out.stat().st_mtime_ns > mtime1, "WAV mtime unchanged on config drift"
    assert out.stat().st_size != size1, "WAV size unchanged on config drift"


def test_extract_audio_mir_stage_skip_on_second_run(
    sample_mp4_with_audio, tmp_path, db_session
):
    from modules.ingest.audio import extract_audio_mir_stage

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True))
    db_session.commit()

    audio_mir_dir = tmp_path / "audio_mir"
    settings = _make_settings(video_dir=vid_dir, audio_mir_dir=audio_mir_dir)

    extract_audio_mir_stage(settings)
    out = audio_mir_dir / "1.wav"
    mtime1 = out.stat().st_mtime_ns

    # Second run should fingerprint-skip without re-extracting.
    extract_audio_mir_stage(settings)
    assert out.stat().st_mtime_ns == mtime1

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
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
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
    db_session.add(Clip(id=99, user_id=1, is_downloaded=True, is_selected=True))
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
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
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
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    audio_mir_dir = tmp_path / "audio_mir"
    settings = _make_settings(video_dir=vid_dir, audio_mir_dir=audio_mir_dir)

    extract_audio_mir_stage(settings)
    out = audio_mir_dir / "1.wav"
    mtime1 = out.stat().st_mtime_ns

    # Second run should fingerprint-skip without re-extracting.
    extract_audio_mir_stage(settings)
    assert out.stat().st_mtime_ns == mtime1


def test_extract_audio_mir_stage_skips_unselected_clips(
    sample_mp4_with_audio, tmp_path, db_session
):
    """Clips with is_selected=False do not get a WAV extracted."""
    from modules.ingest.audio import extract_audio_mir_stage

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())
    (vid_dir / "2.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
    db_session.add(Clip(id=2, user_id=1, is_downloaded=True, is_selected=False))
    db_session.commit()

    audio_mir_dir = tmp_path / "audio_mir"
    settings = _make_settings(video_dir=vid_dir, audio_mir_dir=audio_mir_dir)

    extract_audio_mir_stage(settings)

    assert (audio_mir_dir / "1.wav").exists()
    assert not (audio_mir_dir / "2.wav").exists()


def test_extract_audio_mir_force_reencode_passes_new_sample_rate(
    monkeypatch, tmp_path, db_session
):
    """Mutating mir_sample_rate_hz triggers ffmpeg with -ar set to the new value."""
    from modules.ingest import audio as audio_mod

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(b"\x00fakebytes")

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    audio_mir_dir = tmp_path / "audio_mir"
    settings = _make_settings(video_dir=vid_dir, audio_mir_dir=audio_mir_dir)

    calls: list[list[str]] = []

    def fake_run_ffmpeg(cmd, *, timeout):
        calls.append(list(cmd))
        # Simulate a successful ffmpeg by creating the .part file the caller renames.
        tmp = cmd[-1]
        Path(tmp).write_bytes(b"RIFF" + b"\x00" * 32)
        return True

    monkeypatch.setattr(audio_mod, "run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(audio_mod, "has_audio_stream", lambda p: True)

    settings.audio_extraction.mir_sample_rate_hz = 16_000
    audio_mod.extract_audio_mir_stage(settings)
    first_call = calls[-1]
    assert "16000" in first_call

    # Drift the sample rate; expect a second ffmpeg call with the new -ar.
    settings.audio_extraction.mir_sample_rate_hz = 22_050
    audio_mod.extract_audio_mir_stage(settings)
    second_call = calls[-1]
    assert "22050" in second_call
    # Sanity: more than one invocation total — the force_reencode bypassed mtime skip.
    assert len(calls) >= 2


def test_sweep_orphan_mir_wavs_deletes_unselected_clip_wavs(
    monkeypatch, tmp_path, db_session
):
    """A WAV whose clip_id is not in the selected set is removed."""
    from modules.ingest.audio import sweep_orphan_mir_wavs

    audio_mir_dir = tmp_path / "audio_mir"
    audio_mir_dir.mkdir()
    (audio_mir_dir / "1.wav").write_bytes(b"RIFF")  # selected
    (audio_mir_dir / "99.wav").write_bytes(b"RIFF")  # not selected (no clip row)

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    monkeypatch.setattr(
        "modules.ingest.audio._probe_wav_format",
        lambda path: (16_000, 1),
    )

    n_deleted = sweep_orphan_mir_wavs(
        session=db_session,
        audio_mir_dir=str(audio_mir_dir),
        expected_sample_rate_hz=16_000,
        expected_channels=1,
    )

    assert (audio_mir_dir / "1.wav").exists()
    assert not (audio_mir_dir / "99.wav").exists()
    assert n_deleted == 1


def test_sweep_orphan_mir_wavs_deletes_format_drifted_wavs(
    monkeypatch, tmp_path, db_session
):
    """A WAV whose probed (sr, channels) differs from current config is removed."""
    from modules.ingest.audio import sweep_orphan_mir_wavs

    audio_mir_dir = tmp_path / "audio_mir"
    audio_mir_dir.mkdir()
    (audio_mir_dir / "5.wav").write_bytes(b"RIFF")

    db_session.add(User(id=1))
    db_session.add(Clip(id=5, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    monkeypatch.setattr(
        "modules.ingest.audio._probe_wav_format",
        lambda path: (44_100, 2),
    )

    n_deleted = sweep_orphan_mir_wavs(
        session=db_session,
        audio_mir_dir=str(audio_mir_dir),
        expected_sample_rate_hz=16_000,
        expected_channels=1,
    )

    assert not (audio_mir_dir / "5.wav").exists()
    assert n_deleted == 1


def test_sweep_orphan_mir_wavs_is_noop_when_all_match(
    monkeypatch, tmp_path, db_session
):
    from modules.ingest.audio import sweep_orphan_mir_wavs

    audio_mir_dir = tmp_path / "audio_mir"
    audio_mir_dir.mkdir()
    (audio_mir_dir / "1.wav").write_bytes(b"RIFF")

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    monkeypatch.setattr(
        "modules.ingest.audio._probe_wav_format",
        lambda path: (16_000, 1),
    )

    n_deleted = sweep_orphan_mir_wavs(
        session=db_session,
        audio_mir_dir=str(audio_mir_dir),
        expected_sample_rate_hz=16_000,
        expected_channels=1,
    )

    assert (audio_mir_dir / "1.wav").exists()
    assert n_deleted == 0


def test_sweep_orphan_mir_wavs_keeps_unprobeable_wavs(
    monkeypatch, tmp_path, db_session
):
    """An unprobeable WAV is conservatively kept (probe failures != format drift)."""
    from modules.ingest.audio import sweep_orphan_mir_wavs

    audio_mir_dir = tmp_path / "audio_mir"
    audio_mir_dir.mkdir()
    (audio_mir_dir / "1.wav").write_bytes(b"RIFF")

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    monkeypatch.setattr("modules.ingest.audio._probe_wav_format", lambda path: None)

    n_deleted = sweep_orphan_mir_wavs(
        session=db_session,
        audio_mir_dir=str(audio_mir_dir),
        expected_sample_rate_hz=16_000,
        expected_channels=1,
    )

    assert (audio_mir_dir / "1.wav").exists()
    assert n_deleted == 0


def test_force_reencode_fires_on_dependency_drift(monkeypatch, tmp_path, db_session):
    """When config_hash matches but dependency_hash drifts (e.g. upstream
    StageState row updated), the stage must pass force_reencode=True so the
    ffmpeg call re-encodes instead of trusting a stale WAV via mtime skip."""
    from core import fingerprint as fp
    from modules.ingest import audio as audio_mod

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(b"\x00fakebytes")

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    audio_mir_dir = tmp_path / "audio_mir"
    settings = _make_settings(video_dir=vid_dir, audio_mir_dir=audio_mir_dir)

    # Pre-seed StageState so config_hash matches but dependency_hash does not.
    ae_payload = audio_mod._mir_config_payload(settings.audio_extraction)
    cfg_hash = fp.hash_text(ae_payload)
    db_session.merge(
        StageState(
            stage_name=audio_mod.AUDIO_EXTRACT_MIR_STAGE,
            scope_key=audio_mod.AUDIO_EXTRACT_MIR_SCOPE,
            data_hash="ignored-but-must-be-stale",
            config_hash=cfg_hash,
            dependency_hash="OLD-DEPENDENCY-HASH",
        )
    )
    db_session.commit()

    calls: list[list[str]] = []

    def fake_run_ffmpeg(cmd, *, timeout):
        calls.append(list(cmd))
        tmp_out = cmd[-1]
        Path(tmp_out).write_bytes(b"RIFF" + b"\x00" * 32)
        return True

    monkeypatch.setattr(audio_mod, "run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(audio_mod, "has_audio_stream", lambda _p: True)

    audio_mod.extract_audio_mir_stage(settings)

    assert calls, "ffmpeg was not invoked"
    # The WAV now exists; a second call with the same settings but a stale
    # dependency_hash must still invoke ffmpeg (force_reencode bypasses mtime skip).
    calls.clear()
    db_session.merge(
        StageState(
            stage_name=audio_mod.AUDIO_EXTRACT_MIR_STAGE,
            scope_key=audio_mod.AUDIO_EXTRACT_MIR_SCOPE,
            data_hash="ignored-but-must-be-stale",
            config_hash=cfg_hash,
            dependency_hash="OLD-DEPENDENCY-HASH",
        )
    )
    db_session.commit()

    audio_mod.extract_audio_mir_stage(settings)
    assert calls, "expected force_reencode=True (ffmpeg invoked) on dependency drift"


def test_extract_audio_mir_config_payload_is_field_reorder_safe():
    """The payload string must be a deterministic JSON, not a positional
    pipe-delimited string. This catches the original brittle f-string form."""
    import json

    from core.config import AudioExtractionSettings
    from modules.ingest.audio import _mir_config_payload

    a = AudioExtractionSettings(
        mir_codec="pcm_s16le",
        mir_extension="wav",
        mir_sample_rate_hz=16000,
        mir_channels=1,
    )
    out = _mir_config_payload(a)
    parsed = json.loads(out)
    assert parsed == {
        "mir_channels": 1,
        "mir_codec": "pcm_s16le",
        "mir_extension": "wav",
        "mir_sample_rate_hz": 16000,
    }

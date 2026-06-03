import os
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from core.database import (
    Base,
    Clip,
    ClipEmbedding,
    StageState,
    User,
    get_engine,
    get_session,
)
from core.ffmpeg import has_audio_stream
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


@dataclass
class _PathsStub:
    video_dir: str
    audio_dir: str

    def video_for(self, clip_id):
        return Path(self.video_dir) / f"{clip_id}.mp4"

    def audio_for(self, clip_id):
        return Path(self.audio_dir) / f"{clip_id}.mp3"

    def thumbnail_for(self, clip_id):
        return Path(self.video_dir) / f"{clip_id}.jpg"


@dataclass
class _EmbeddingsStub:
    embed_batch_size: int = 1


@dataclass
class _AudioExtractionStub:
    audio_bitrate_kbps: int = 128
    audio_sample_rate_hz: int = 44100
    audio_extract_timeout_s: int = 60


@dataclass
class _DownloadStub:
    concurrency: int = 2


@dataclass
class _SettingsStub:
    paths: _PathsStub
    embeddings: _EmbeddingsStub
    audio_extraction: _AudioExtractionStub
    download: _DownloadStub


def _make_settings(*, audio_dir, video_dir, concurrency: int = 2) -> _SettingsStub:
    return _SettingsStub(
        paths=_PathsStub(video_dir=str(video_dir), audio_dir=str(audio_dir)),
        embeddings=_EmbeddingsStub(),
        audio_extraction=_AudioExtractionStub(),
        download=_DownloadStub(concurrency=concurrency),
    )


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, ClipEmbedding, Clip, User):
        session.query(model).delete()
    session.commit()
    yield session
    for model in (StageState, ClipEmbedding, Clip, User):
        session.query(model).delete()
    session.commit()
    session.close()


def test_runs_and_seals(tmp_path, sample_mp4_with_audio, db_session):
    """Audio extraction runs over downloaded selected clips and seals."""
    from modules.ingest.audio import extract_audio_stage

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())

    db_session.add(User(id=1, is_selected=True))
    db_session.add(Clip(id=1, user_id=1, is_selected=True, is_downloaded=True))
    db_session.commit()

    audio_dir = tmp_path / "audio"
    settings = _make_settings(audio_dir=audio_dir, video_dir=vid_dir)

    extract_audio_stage(settings)

    assert (audio_dir / "1.mp3").exists()
    assert db_session.get(StageState, ("audio_extract", "default")) is not None


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
    settings = _make_settings(audio_dir=audio_dir, video_dir=vid_dir)
    extract_audio_stage(settings)

    assert (audio_dir / "1.mp3").exists()
    row = db_session.get(StageState, ("audio_extract", "default"))
    assert row is not None  # sealed

    # Second run is a no-op (fingerprint matches → no ffmpeg invocation).
    with patch("modules.ingest.audio.run_ffmpeg") as ff:
        extract_audio_stage(settings)
    ff.assert_not_called()


def test_extract_audio_stage_dispatches_through_thread_pool(
    tmp_path, sample_mp4_with_audio, db_session
):
    """extract_audio_stage must run ffmpeg invocations via a
    ThreadPoolExecutor sized by settings.download.concurrency."""
    from modules.ingest import audio as audio_mod
    from modules.ingest.audio import extract_audio_stage

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    for cid in (1, 2, 3):
        (vid_dir / f"{cid}.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())
        db_session.add(User(id=cid, is_selected=True))
        db_session.add(Clip(id=cid, user_id=cid, is_selected=True, is_downloaded=True))
    db_session.commit()

    audio_dir = tmp_path / "audio"
    settings = _make_settings(audio_dir=audio_dir, video_dir=vid_dir, concurrency=4)

    with patch.object(
        audio_mod, "ThreadPoolExecutor", wraps=audio_mod.ThreadPoolExecutor
    ) as TPE:
        extract_audio_stage(settings)

    TPE.assert_called_once_with(max_workers=4)
    assert all((audio_dir / f"{cid}.mp3").exists() for cid in (1, 2, 3))


def test_has_audio_stream_true_when_video_has_audio(sample_mp4_with_audio):
    assert has_audio_stream(str(sample_mp4_with_audio)) is True


def test_has_audio_stream_false_when_video_is_silent(sample_mp4_no_audio):
    assert has_audio_stream(str(sample_mp4_no_audio)) is False


def test_has_audio_stream_false_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    assert has_audio_stream(str(missing)) is False


def test_stage_skips_clips_without_audio_stream_and_seals(
    tmp_path, sample_mp4_with_audio, sample_mp4_no_audio, db_session
):
    """A clip whose video has no audio stream must not count as a failure.

    Reproduces the production bug where 3 Reels with video-only streams
    forced the stage to re-run on every pipeline invocation. The stage
    must seal its fingerprint despite the silent clips.
    """
    from modules.ingest.audio import extract_audio_stage

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    # Clip 1 has audio, clip 2 is video-only.
    (vid_dir / "1.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())
    (vid_dir / "2.mp4").write_bytes(Path(sample_mp4_no_audio).read_bytes())

    db_session.add(User(id=1, is_selected=True))
    db_session.add(Clip(id=1, user_id=1, is_selected=True, is_downloaded=True))
    db_session.add(Clip(id=2, user_id=1, is_selected=True, is_downloaded=True))
    db_session.commit()

    audio_dir = tmp_path / "audio"
    settings = _make_settings(audio_dir=audio_dir, video_dir=vid_dir)

    extract_audio_stage(settings)

    # Clip 1 produced an mp3; clip 2 did not (skipped, not failed).
    assert (audio_dir / "1.mp3").exists()
    assert not (audio_dir / "2.mp3").exists()
    # Fingerprint sealed despite silent clip.
    assert db_session.get(StageState, ("audio_extract", "default")) is not None


def test_stage_does_not_retry_silent_clips_on_second_run(
    tmp_path, sample_mp4_no_audio, db_session
):
    """After sealing, a stage with only no-audio clips must short-circuit.

    Guard against any regression that re-introduces stale-fingerprint
    behavior for video-only Reels.
    """
    from modules.ingest.audio import extract_audio_stage

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(Path(sample_mp4_no_audio).read_bytes())

    db_session.add(User(id=1, is_selected=True))
    db_session.add(Clip(id=1, user_id=1, is_selected=True, is_downloaded=True))
    db_session.commit()

    audio_dir = tmp_path / "audio"
    settings = _make_settings(audio_dir=audio_dir, video_dir=vid_dir)

    # First run seals.
    extract_audio_stage(settings)
    assert db_session.get(StageState, ("audio_extract", "default")) is not None

    # Second run must be a no-op: no ffmpeg invocation, no ffprobe rescan.
    with (
        patch("modules.ingest.audio.run_ffmpeg") as ff,
        patch("modules.ingest.audio.has_audio_stream") as probe,
    ):
        extract_audio_stage(settings)
    ff.assert_not_called()
    probe.assert_not_called()


def test_stage_skips_already_extracted_clips_without_reprobing(
    tmp_path, sample_mp4_with_audio, db_session
):
    """When the clip set grows (fingerprint goes stale), clips that already
    have a fresh mp3 must be skipped without re-running ffprobe.

    Reproduces the production complaint: adding a creator re-walks the whole
    history and re-probes every old clip. Only genuinely new clips should be
    probed/extracted; the rest are dismissed by the mtime shortcut.
    """
    from core.ffmpeg import has_audio_stream as real_probe
    from modules.ingest.audio import extract_audio_stage

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())

    db_session.add(User(id=1, is_selected=True))
    db_session.add(Clip(id=1, user_id=1, is_selected=True, is_downloaded=True))
    db_session.commit()

    audio_dir = tmp_path / "audio"
    settings = _make_settings(audio_dir=audio_dir, video_dir=vid_dir)

    # First run extracts clip 1 and seals.
    extract_audio_stage(settings)
    assert (audio_dir / "1.mp3").exists()

    # A new creator/clip arrives -> fingerprint goes stale, stage re-runs.
    (vid_dir / "2.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())
    db_session.add(Clip(id=2, user_id=1, is_selected=True, is_downloaded=True))
    db_session.commit()

    with patch("modules.ingest.audio.has_audio_stream", wraps=real_probe) as probe:
        extract_audio_stage(settings)

    probed = [call.args[0] for call in probe.call_args_list]
    # Old, already-extracted clip must NOT be re-probed.
    assert str(vid_dir / "1.mp4") not in probed
    # The new clip is probed and extracted.
    assert str(vid_dir / "2.mp4") in probed
    assert (audio_dir / "2.mp3").exists()
    # Stage still seals after handling the new clip.
    assert db_session.get(StageState, ("audio_extract", "default")) is not None


def test_extracts_wav_pcm_stereo(sample_mp4_with_audio, tmp_path):
    from modules.ingest.audio import extract_audio

    out = tmp_path / "1.wav"
    result = extract_audio(
        str(sample_mp4_with_audio),
        str(out),
        bitrate_kbps=0,
        sample_rate_hz=44_100,
        timeout_s=60,
        codec="pcm_s16le",
        extension="wav",
        channels=2,
    )
    assert result.ok
    assert out.exists()
    assert out.stat().st_size > 5_000
    # WAV PCM s16le starts with "RIFF....WAVE"
    head = out.read_bytes()[:12]
    assert head[:4] == b"RIFF" and head[8:12] == b"WAVE"

"""Tests for scripts/retry_failed_music_features.py"""

from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database import Base, Music
from modules.music.state import _NO_MATCH


def _make_db_with_failed_music(rows):
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    s = Session(eng)
    for row in rows:
        s.add(Music(**row))
    s.commit()
    return s


def _settings():
    from core.config import MusicSettings, PathsSettings
    from modules.music.features import MusicSecrets

    paths = PathsSettings(
        video_dir="/tmp",
        plots_dir="/tmp",
        model_path="/tmp",
        profile_pic_dir="/tmp",
        thumbnail_dir="/tmp",
        speech_audio_dir="/tmp/audio",
        audio_dir="/tmp/audio",
        data_csv_path="/tmp/data.csv",
    )
    secrets = MusicSecrets(spotify_client_id="i", spotify_client_secret="s")
    music = MusicSettings(
        audio_fingerprint_confidence=0.5,
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
    return music, paths, secrets


def test_retry_resets_failed_rows_to_pending(monkeypatch):
    s = _make_db_with_failed_music(
        [
            {
                "id": 1,
                "artist": "a",
                "track": "t",
                "spotify_id": _NO_MATCH,
                "reccobeats_id": _NO_MATCH,
                "is_audio_features_extracted": False,
            }
        ]
    )
    monkeypatch.setattr("scripts.retry_failed_music_features.get_session", lambda: s)
    fake_ext = MagicMock()
    monkeypatch.setattr(
        "scripts.retry_failed_music_features.extract_music_features", fake_ext
    )

    from scripts.retry_failed_music_features import retry_failed_music_features

    music, paths, secrets = _settings()
    retry_failed_music_features(music, paths, secrets)

    row = s.query(Music).filter_by(id=1).one()
    assert row.is_audio_features_extracted is None
    assert row.spotify_id is None
    assert row.reccobeats_id is None
    fake_ext.assert_called_once()
    music_arg = fake_ext.call_args.kwargs["music"]
    assert music_arg.api_max_attempts == 1


def test_retry_preserves_non_no_match_ids(monkeypatch):
    s = _make_db_with_failed_music(
        [
            {
                "id": 1,
                "artist": "a",
                "track": "t",
                "spotify_id": "real-spotify",
                "reccobeats_id": "real-rb",
                "is_audio_features_extracted": False,
            }
        ]
    )
    monkeypatch.setattr("scripts.retry_failed_music_features.get_session", lambda: s)
    monkeypatch.setattr(
        "scripts.retry_failed_music_features.extract_music_features",
        MagicMock(),
    )

    from scripts.retry_failed_music_features import retry_failed_music_features

    music, paths, secrets = _settings()
    retry_failed_music_features(music, paths, secrets)

    row = s.query(Music).filter_by(id=1).one()
    assert row.spotify_id == "real-spotify"
    assert row.reccobeats_id == "real-rb"


def test_retry_skips_when_no_failed_rows(monkeypatch):
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    s = Session(eng)
    monkeypatch.setattr("scripts.retry_failed_music_features.get_session", lambda: s)
    fake_ext = MagicMock()
    monkeypatch.setattr(
        "scripts.retry_failed_music_features.extract_music_features", fake_ext
    )

    from scripts.retry_failed_music_features import retry_failed_music_features

    music, paths, secrets = _settings()
    retry_failed_music_features(music, paths, secrets)

    fake_ext.assert_not_called()

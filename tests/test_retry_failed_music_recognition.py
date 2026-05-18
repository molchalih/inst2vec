"""Tests for scripts/retry_failed_music_recognition.py"""

from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.config import MusicSettings, PathsSettings
from core.database import Base, Clip, User
from modules.music.classify import AcrSecrets


def _music_settings(**overrides) -> MusicSettings:
    base = dict(
        audio_fingerprint_confidence=0.5,
        commit_every=10,
        http_timeout=30.0,
        spotify_search_limit=5,
        spotify_token_skew_seconds=30,
        spotify_request_timeout=10.0,
        reccobeats_batch_size=10,
        reccobeats_delay_min=0.1,
        reccobeats_delay_max=0.3,
        manual_features_max_seconds=30,
        manual_features_sample_rate=22050,
        manual_features_max_mb=8.0,
        manual_features_mp3_bitrate="128k",
        api_max_attempts=3,
        api_retry_delay=1.0,
        api_retry_jitter=0.5,
        acr_max_attempts=3,
        ffmpeg_timeout_seconds=30,
    )
    base.update(overrides)
    return MusicSettings(**base)


def _paths(tmp_path: Path) -> PathsSettings:
    return PathsSettings(
        video_dir=str(tmp_path),
        plots_dir=str(tmp_path),
        model_path=str(tmp_path),
        profile_pic_dir=str(tmp_path),
        thumbnail_dir=str(tmp_path),
        speech_audio_dir=str(tmp_path),
        audio_dir=str(tmp_path),
        data_csv_path=str(tmp_path / "x.csv"),
    )


def _acr_secrets() -> AcrSecrets:
    return AcrSecrets(host="h", access_key="k", access_secret="s")


def _make_db(tmp_path: Path):
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    s = Session(eng)
    s.add(User(id=1, parse_status="success", is_selected=True))
    s.add(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            is_music_recognized=False,
        )
    )
    s.commit()
    return s


def test_retry_nulls_failed_rows_then_calls_classify_music(tmp_path, monkeypatch):
    """Failed rows must be flipped to NULL and classify_music re-invoked
    with a single-attempt MusicSettings copy."""
    s = _make_db(tmp_path)
    monkeypatch.setattr("scripts.retry_failed_music_recognition.get_session", lambda: s)

    captured: dict = {}

    def fake_classify_music(*, music, paths, secrets):
        captured["music"] = music
        captured["paths"] = paths
        captured["secrets"] = secrets
        # Inspect state of the row at call time: must be NULL by now.
        captured["row_state_at_call"] = (
            s.query(Clip).filter_by(id=10).one().is_music_recognized
        )

    monkeypatch.setattr(
        "scripts.retry_failed_music_recognition.classify_music", fake_classify_music
    )

    from scripts.retry_failed_music_recognition import retry_failed_music_recognition

    retry_failed_music_recognition(
        music=_music_settings(), paths=_paths(tmp_path), secrets=_acr_secrets()
    )

    assert captured["row_state_at_call"] is None
    assert captured["music"].api_max_attempts == 1
    assert captured["music"].acr_max_attempts == 1
    assert captured["music"].api_retry_delay == 0.0
    assert captured["music"].api_retry_jitter == 0.0


def test_retry_skips_when_no_failed_rows(tmp_path, monkeypatch):
    """No is_music_recognized=False rows → classify_music must not be called."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    s = Session(eng)
    monkeypatch.setattr("scripts.retry_failed_music_recognition.get_session", lambda: s)

    fake_classify = MagicMock(side_effect=AssertionError("should not be called"))
    monkeypatch.setattr(
        "scripts.retry_failed_music_recognition.classify_music", fake_classify
    )

    from scripts.retry_failed_music_recognition import retry_failed_music_recognition

    retry_failed_music_recognition(
        music=_music_settings(), paths=_paths(tmp_path), secrets=_acr_secrets()
    )

    fake_classify.assert_not_called()

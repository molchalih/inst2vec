"""Behavior tests for classify_music."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.config import MusicSettings, PathsSettings
from modules.database import Base, Clip, Music, User
from modules.music.classify import AcrSecrets, classify_music


def _music_settings(**overrides) -> MusicSettings:
    base = dict(
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
    base.update(overrides)
    return MusicSettings(**base)


def _paths(video_dir: str) -> PathsSettings:
    return PathsSettings(
        video_dir=video_dir,
        plots_dir="/tmp",
        model_path="/tmp",
        profile_pic_dir="/tmp",
        thumbnail_dir="/tmp",
        data_csv_path="/tmp/data.csv",
    )


@pytest.fixture
def db_session(monkeypatch, tmp_path):
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
            is_music_recognized=None,
        )
    )
    s.commit()
    video_file = tmp_path / "10.mp4"
    video_file.write_bytes(b"fake")

    monkeypatch.setattr("modules.music.classify.get_session", lambda: s)

    yield s, tmp_path
    s.close()


def test_classify_match_sets_is_music_recognized_true(db_session, monkeypatch):
    s, tmp_path = db_session

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.return_value = (
        '{"status":{"code":0},"metadata":{"music":[{'
        '"title":"Song","artists":[{"name":"Artist"}],"score":90}]}}'
    )
    monkeypatch.setattr(
        "modules.music.classify.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    classify_music(
        music=_music_settings(),
        paths=_paths(str(tmp_path)),
        secrets=AcrSecrets(host="h", access_key="k", access_secret="s"),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_music_recognized is True
    assert clip.music_id is not None
    music = s.query(Music).filter_by(id=clip.music_id).one()
    assert music.artist == "Artist"
    assert music.track == "Song"


def test_classify_clean_no_match_sets_false(db_session, monkeypatch):
    s, tmp_path = db_session

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.return_value = (
        '{"status":{"code":0},"metadata":{"music":[]}}'
    )
    monkeypatch.setattr(
        "modules.music.classify.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    classify_music(
        music=_music_settings(),
        paths=_paths(str(tmp_path)),
        secrets=AcrSecrets(host="h", access_key="k", access_secret="s"),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_music_recognized is False
    assert clip.music_id is None


def test_classify_transient_exhaustion_sets_false(db_session, monkeypatch):
    s, tmp_path = db_session

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.side_effect = RuntimeError("network")
    monkeypatch.setattr(
        "modules.music.classify.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    classify_music(
        music=_music_settings(acr_max_attempts=2),
        paths=_paths(str(tmp_path)),
        secrets=AcrSecrets(host="h", access_key="k", access_secret="s"),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_music_recognized is False
    assert fake_acr.recognize_by_file.call_count == 2


def test_classify_existing_music_row_reused(db_session, monkeypatch):
    s, tmp_path = db_session
    s.add(Music(id=42, artist="Artist", track="Song"))
    s.commit()

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.return_value = (
        '{"status":{"code":0},"metadata":{"music":[{'
        '"title":"Song","artists":[{"name":"Artist"}],"score":90}]}}'
    )
    monkeypatch.setattr(
        "modules.music.classify.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    classify_music(
        music=_music_settings(),
        paths=_paths(str(tmp_path)),
        secrets=AcrSecrets(host="h", access_key="k", access_secret="s"),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.music_id == 42


def test_classify_skips_non_null_clips(db_session, monkeypatch):
    s, tmp_path = db_session
    clip = s.query(Clip).filter_by(id=10).one()
    clip.is_music_recognized = True
    s.commit()

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.side_effect = AssertionError("should not be called")
    monkeypatch.setattr(
        "modules.music.classify.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    classify_music(
        music=_music_settings(),
        paths=_paths(str(tmp_path)),
        secrets=AcrSecrets(host="h", access_key="k", access_secret="s"),
    )

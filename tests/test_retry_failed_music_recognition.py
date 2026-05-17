"""Tests for scripts/retry_failed_music_recognition.py"""

from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database import Base, Clip, User


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
    (tmp_path / "10.mp4").write_bytes(b"fake")
    return s


def test_retry_recovers_failed_clip(tmp_path, monkeypatch):
    s = _make_db(tmp_path)
    monkeypatch.setattr("scripts.retry_failed_music_recognition.get_session", lambda: s)

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.return_value = (
        '{"status":{"code":0},"metadata":{"music":[{'
        '"title":"S","artists":[{"name":"A"}],"score":90}]}}'
    )
    monkeypatch.setattr(
        "scripts.retry_failed_music_recognition.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    from scripts.retry_failed_music_recognition import retry_failed_music_recognition

    retry_failed_music_recognition(
        video_dir=str(tmp_path),
        min_confidence=0.5,
        arc_host="h",
        arc_access_key="k",
        arc_access_secret="s",
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_music_recognized is True
    assert clip.music_id is not None


def test_retry_leaves_clean_no_match_as_false(tmp_path, monkeypatch):
    s = _make_db(tmp_path)
    monkeypatch.setattr("scripts.retry_failed_music_recognition.get_session", lambda: s)

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.return_value = (
        '{"status":{"code":0},"metadata":{"music":[]}}'
    )
    monkeypatch.setattr(
        "scripts.retry_failed_music_recognition.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    from scripts.retry_failed_music_recognition import retry_failed_music_recognition

    retry_failed_music_recognition(
        video_dir=str(tmp_path),
        min_confidence=0.5,
        arc_host="h",
        arc_access_key="k",
        arc_access_secret="s",
    )

    assert s.query(Clip).filter_by(id=10).one().is_music_recognized is False


def test_retry_skips_when_no_failed_rows(tmp_path, monkeypatch):
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    s = Session(eng)
    monkeypatch.setattr("scripts.retry_failed_music_recognition.get_session", lambda: s)

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.side_effect = AssertionError("should not be called")
    monkeypatch.setattr(
        "scripts.retry_failed_music_recognition.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    from scripts.retry_failed_music_recognition import retry_failed_music_recognition

    retry_failed_music_recognition(
        video_dir=str(tmp_path),
        min_confidence=0.5,
        arc_host="h",
        arc_access_key="k",
        arc_access_secret="s",
    )

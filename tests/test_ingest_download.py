"""Behavior tests for download_files retry_failed kwarg."""

import pytest

from core.config import DownloadSettings, PathsSettings
from core.database import (
    Base,
    Clip,
    StageState,
    User,
    get_engine,
    get_session,
)
from modules.ingest.download import download_files


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, Clip, User):
        session.query(model).delete()
    session.commit()
    try:
        yield session
    finally:
        session.rollback()
        for model in (StageState, Clip, User):
            session.query(model).delete()
        session.commit()
        session.close()


def _paths(tmp_path) -> PathsSettings:
    return PathsSettings(
        video_dir=str(tmp_path / "v"),
        model_path=str(tmp_path),
        profile_pic_dir=str(tmp_path / "p"),
        thumbnail_dir=str(tmp_path / "t"),
        speech_audio_dir=str(tmp_path),
        audio_dir=str(tmp_path),
        data_csv_path=str(tmp_path / "x.csv"),
    )


def _download_settings() -> DownloadSettings:
    return DownloadSettings(
        max_attempts=1, retry_delay=0, retry_jitter=0, concurrency=1
    )


def test_download_files_retry_failed_picks_up_false_rows(
    db_session, monkeypatch, tmp_path
):
    """retry_failed=True must re-attempt clips left at is_downloaded=False."""
    s = db_session
    s.add(User(id=1, parse_status="success", is_selected=True))
    s.add(
        Clip(
            id=1,
            user_id=1,
            video_url="http://example.com/v.mp4",
            is_selected=True,
            is_downloaded=False,
        )
    )
    s.commit()

    called: list[tuple[str, str]] = []

    def fake_fetch(url, path, *_, **__):
        called.append((url, path))
        return True

    monkeypatch.setattr("modules.ingest.download.fetch_file", fake_fetch)

    download_files(_download_settings(), _paths(tmp_path), retry_failed=True)

    assert any(p.endswith("1.mp4") for _, p in called)
    assert s.query(Clip).filter_by(id=1).one().is_downloaded is True


def test_download_files_default_skips_false_rows(db_session, monkeypatch, tmp_path):
    """Default retry_failed=False must NOT re-attempt False rows."""
    s = db_session
    s.add(User(id=1, parse_status="success", is_selected=True))
    s.add(
        Clip(
            id=1,
            user_id=1,
            video_url="http://example.com/v.mp4",
            is_selected=True,
            is_downloaded=False,
        )
    )
    s.commit()

    called: list[tuple] = []
    monkeypatch.setattr(
        "modules.ingest.download.fetch_file",
        lambda *a, **k: called.append(a[:2]) or True,
    )

    download_files(_download_settings(), _paths(tmp_path))

    assert not [c for c in called if c[1].endswith("1.mp4")]
    assert s.query(Clip).filter_by(id=1).one().is_downloaded is False

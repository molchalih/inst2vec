"""Fingerprint-stability guard for the download stage.

The download stage owns no fingerprint seal: it relies purely on row-level
DB state (``Clip.is_downloaded``) for idempotence and writes no
``StageState`` row. This test pins that contract so a perf change cannot
silently introduce — or drift — a stage fingerprint.
"""

from __future__ import annotations

import httpx
import pytest

from core.config import DownloadSettings, PathsSettings
from core.database import Clip, StageState, User, get_session, init_db
from modules.ingest import download as dl_mod


@pytest.fixture
def isolated_db(tmp_path):
    from core.database import engine as engine_mod

    original_main = engine_mod._main_engine
    original_identity = engine_mod._identity_engine

    init_db(f"sqlite:///{tmp_path}/main.db", f"sqlite:///{tmp_path}/id.db")
    yield

    engine_mod._main_engine = original_main
    engine_mod._identity_engine = original_identity


def _make_response(status_code=200, content=b"x" * 1024):
    from unittest.mock import MagicMock

    r = MagicMock(spec=httpx.Response)
    r.status_code = status_code
    r.content = content
    r.raise_for_status.return_value = None
    return r


def _paths(tmp_path):
    return PathsSettings(
        profile_pic_dir=str(tmp_path / "pics"),
        thumbnail_dir=str(tmp_path / "thumbs"),
        video_dir=str(tmp_path / "vids"),
        speech_audio_dir=str(tmp_path / "audio"),
        audio_dir=str(tmp_path / "audio"),
        model_path="",
        data_csv_path="",
    )


def _stage_rows() -> list[tuple[str, str]]:
    session = get_session()
    try:
        return [(s.stage_name, s.scope_key) for s in session.query(StageState).all()]
    finally:
        session.close()


def test_download_stage_writes_no_fingerprint_row(tmp_path, monkeypatch, isolated_db):
    session = get_session()
    if not session.get(User, 1):
        session.add(User(id=1, is_selected=True))
    session.add(
        Clip(
            id=100,
            user_id=1,
            is_selected=True,
            is_downloaded=None,
            video_url="https://x/v.mp4",
            thumbnail_url="https://x/t.jpg",
        )
    )
    session.commit()
    session.close()

    monkeypatch.setattr(dl_mod, "get_profile_pic_url", lambda uid: None)
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_response())
    monkeypatch.setattr(dl_mod.time, "sleep", lambda _: None)

    download = DownloadSettings(
        max_attempts=1, retry_delay=0, retry_jitter=0, concurrency=2
    )
    dl_mod.download_files(download, _paths(tmp_path))

    assert _stage_rows() == []

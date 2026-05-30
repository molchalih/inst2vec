"""Fingerprint-stability guards for the download and upload stages.

Neither stage owns a fingerprint seal: both rely purely on row-level DB
state (``Clip.is_downloaded`` / ``Clip.is_uploaded``) for idempotence and
write no ``StageState`` row. These tests pin that contract so a perf change
cannot silently introduce — or drift — a stage fingerprint.
"""

from __future__ import annotations

import os

import httpx
import pytest

from core.config import DownloadSettings, PathsSettings
from core.database import Clip, StageState, User, get_session, init_db
from modules.ingest import download as dl_mod
from modules.upload import upload_videos


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


def _upload_settings(tmp_path, *, bucket: str):
    from unittest.mock import MagicMock

    settings = MagicMock()
    settings.storage.bucket = bucket
    settings.storage.prefix = "videos/"
    settings.storage.backend = "s3"
    settings.storage.region = ""
    settings.storage.verify_concurrency = 4
    settings.storage.upload_concurrency = 2
    settings.paths.video_dir = str(tmp_path / "videos")
    settings.embeddings.gemini_enabled = False
    return settings


def _upload_secrets(*, embedder_token: str = "tok"):
    from core.config import Secrets

    return Secrets(
        database_url=os.environ.get("DATABASE_URL", "sqlite:///:memory:"),
        identity_db_url=os.environ.get("IDENTITY_DB_URL", "sqlite:///:memory:"),
        hiker_api_key="x",
        huggingface_token="x",
        embedder_token=embedder_token,
        object_store_endpoint="",
        object_store_access_key="t",
        object_store_secret_key="t",
    )


def test_upload_stage_skip_path_writes_no_fingerprint_row(tmp_path, isolated_db):
    """Bucket unset → skip; must not seal a fingerprint row."""
    session = get_session()
    if not session.get(User, 1):
        session.add(User(id=1))
    session.add(Clip(id=1, user_id=1, is_selected=True, is_downloaded=True))
    session.commit()
    session.close()

    settings = _upload_settings(tmp_path, bucket="")
    upload_videos(settings, _upload_secrets())

    assert _stage_rows() == []


def test_upload_stage_run_path_writes_no_fingerprint_row(tmp_path, isolated_db):
    """Remote active + bucket set → stage runs (verify+upload) yet still seals
    no fingerprint row; idempotence stays row-level (Clip.is_uploaded)."""
    moto = pytest.importorskip("moto")

    from core.config import StorageSettings
    from core.storage import ObjectStore

    with moto.mock_aws():
        store = ObjectStore(
            settings=StorageSettings(
                backend="s3", bucket="test-bucket", prefix="videos/"
            ),
            endpoint_url=None,
            access_key="t",
            secret_key="t",
        )
        store.client.create_bucket(Bucket="test-bucket")

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "1.mp4").write_bytes(b"\x00" * 256)

        session = get_session()
        if not session.get(User, 1):
            session.add(User(id=1))
        session.add(Clip(id=1, user_id=1, is_selected=True, is_downloaded=True))
        session.commit()
        session.close()

        settings = _upload_settings(tmp_path, bucket="test-bucket")
        upload_videos(settings, _upload_secrets())

    assert _stage_rows() == []

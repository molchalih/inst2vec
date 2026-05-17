"""Tests for the Upload pipeline stage."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

pytest.importorskip("moto")
from moto import mock_aws

from core.config import StorageSettings
from core.database import Clip, User, get_session, init_db
from core.storage import ObjectStore
from modules.upload import upload_videos


@pytest.fixture
def isolated_db(tmp_path):
    """Point the global engine at a fresh per-test DB, restore after."""
    from core.database import engine as engine_mod

    original_main = engine_mod._main_engine
    original_identity = engine_mod._identity_engine

    init_db(f"sqlite:///{tmp_path}/main.db", f"sqlite:///{tmp_path}/id.db")
    yield

    engine_mod._main_engine = original_main
    engine_mod._identity_engine = original_identity


def _seed_clip(session, *, clip_id: int, selected: bool, downloaded: bool):
    user_id = clip_id // 100 or 1
    if not session.get(User, user_id):
        session.add(User(id=user_id))
    session.add(
        Clip(
            id=clip_id,
            user_id=user_id,
            is_selected=selected,
            is_downloaded=downloaded,
            is_uploaded=None,
        )
    )
    session.commit()


def test_noop_when_bucket_unset(tmp_path, isolated_db):
    """When storage.bucket is empty, upload_videos is a no-op."""
    settings = _make_settings(tmp_path, bucket="")
    secrets = _make_secrets()
    upload_videos(settings, secrets)  # should not raise


BUCKET = "test-bucket"


def test_uploads_selected_downloaded_clips(tmp_path, isolated_db):
    with mock_aws():
        store = ObjectStore(
            settings=StorageSettings(
                backend="s3", bucket=BUCKET, prefix="videos/", signed_url_ttl_s=60
            ),
            endpoint_url=None,
            access_key="t",
            secret_key="t",
        )
        store.client.create_bucket(Bucket=BUCKET)

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "123.mp4").write_bytes(b"\x00" * 1024)

        session = get_session()
        _seed_clip(session, clip_id=123, selected=True, downloaded=True)
        session.close()

        settings = _make_settings(tmp_path, bucket=BUCKET, video_dir=str(video_dir))
        secrets = _make_secrets()

        upload_videos(settings, secrets)

        assert store.head("videos/123.mp4")["size"] == 1024
        session = get_session()
        assert session.get(Clip, 123).is_uploaded is True
        session.close()


def test_skips_already_uploaded_clips(tmp_path, isolated_db):
    with mock_aws():
        store = ObjectStore(
            settings=StorageSettings(
                backend="s3", bucket=BUCKET, prefix="videos/", signed_url_ttl_s=60
            ),
            endpoint_url=None,
            access_key="t",
            secret_key="t",
        )
        store.client.create_bucket(Bucket=BUCKET)

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "5.mp4").write_bytes(b"new-bytes")

        session = get_session()
        _seed_clip(session, clip_id=5, selected=True, downloaded=True)
        clip = session.get(Clip, 5)
        clip.is_uploaded = True
        session.commit()
        session.close()

        settings = _make_settings(tmp_path, bucket=BUCKET, video_dir=str(video_dir))
        secrets = _make_secrets()

        upload_videos(settings, secrets)

        # head should return None — we never uploaded
        assert store.head("videos/5.mp4") is None


def test_ignores_unselected_and_undownloaded_clips(tmp_path, isolated_db):
    with mock_aws():
        store = ObjectStore(
            settings=StorageSettings(
                backend="s3", bucket=BUCKET, prefix="videos/", signed_url_ttl_s=60
            ),
            endpoint_url=None,
            access_key="t",
            secret_key="t",
        )
        store.client.create_bucket(Bucket=BUCKET)

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "1.mp4").write_bytes(b"x")
        (video_dir / "2.mp4").write_bytes(b"x")

        session = get_session()
        _seed_clip(session, clip_id=1, selected=False, downloaded=True)
        _seed_clip(session, clip_id=2, selected=True, downloaded=False)
        session.close()

        settings = _make_settings(tmp_path, bucket=BUCKET, video_dir=str(video_dir))
        secrets = _make_secrets()

        upload_videos(settings, secrets)

        assert store.head("videos/1.mp4") is None
        assert store.head("videos/2.mp4") is None


# ── helpers ────────────────────────────────────────────────────────────────


def _make_settings(tmp_path, *, bucket: str, video_dir: str | None = None):
    """Build a minimal settings mock exposing only what upload_videos needs."""
    settings = MagicMock()
    settings.storage.bucket = bucket
    settings.storage.prefix = "videos/"
    settings.storage.signed_url_ttl_s = 60
    settings.storage.backend = "s3"
    settings.paths.video_dir = video_dir or str(tmp_path / "videos")
    return settings


def _make_secrets():
    from core.config import Secrets

    return Secrets(
        database_url=os.environ.get("DATABASE_URL", "sqlite:///:memory:"),
        identity_db_url=os.environ.get("IDENTITY_DB_URL", "sqlite:///:memory:"),
        hiker_api_key="x",
        arc_host="x",
        arc_access_key="x",
        arc_secret_key="x",
        spotify_client_id="x",
        spotify_client_secret="x",
        huggingface_token="x",
        object_store_endpoint="",
        object_store_access_key="t",
        object_store_secret_key="t",
    )

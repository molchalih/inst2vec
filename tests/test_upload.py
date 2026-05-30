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
            settings=StorageSettings(backend="s3", bucket=BUCKET, prefix="videos/"),
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


def test_trusts_is_uploaded_skips_already_uploaded(tmp_path, isolated_db):
    """DB-trust: a clip already is_uploaded=True is NOT HEADed or re-uploaded
    (the old bucket-authoritative self-heal is traded for speed). The flag and
    the (absent) object both stay as-is."""
    with mock_aws():
        store = ObjectStore(
            settings=StorageSettings(backend="s3", bucket=BUCKET, prefix="videos/"),
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

        # No verify/upload happened: the object is still absent, flag still True.
        assert store.head("videos/5.mp4") is None
        session = get_session()
        assert session.get(Clip, 5).is_uploaded is True
        session.close()


def test_skips_whole_stage_when_token_unset(tmp_path, isolated_db, monkeypatch):
    """Remote path inactive (no embedder token) → no clip query, no HEADs."""
    from modules import upload as upload_mod

    def _boom(*a, **kw):
        raise AssertionError("get_object_store must not be called when inactive")

    monkeypatch.setattr(upload_mod, "get_object_store", _boom)

    session = get_session()
    _seed_clip(session, clip_id=7, selected=True, downloaded=True)
    session.close()

    settings = _make_settings(tmp_path, bucket=BUCKET)
    secrets = _make_secrets(embedder_token="")

    upload_videos(settings, secrets)  # must not raise / must not touch store

    session = get_session()
    assert session.get(Clip, 7).is_uploaded is None
    session.close()


def test_skips_whole_stage_when_no_active_served_remotely_case(
    tmp_path, isolated_db, monkeypatch
):
    """Remote path inactive (active cases are all local-only) → stage skips."""
    from modules import upload as upload_mod

    def _boom(*a, **kw):
        raise AssertionError("get_object_store must not be called when inactive")

    monkeypatch.setattr(upload_mod, "get_object_store", _boom)
    # maest is served_remotely=False; intersection with served-remotely is empty.
    monkeypatch.setattr(upload_mod, "default_cases", lambda s: ("maest",))

    session = get_session()
    _seed_clip(session, clip_id=8, selected=True, downloaded=True)
    session.close()

    settings = _make_settings(tmp_path, bucket=BUCKET)
    secrets = _make_secrets()

    upload_videos(settings, secrets)

    session = get_session()
    assert session.get(Clip, 8).is_uploaded is None
    session.close()


def test_uploads_only_pending_when_remote_active(tmp_path, isolated_db):
    """Remote active + bucket set: only is_uploaded False/NULL rows are
    verified+uploaded; already-uploaded rows are left untouched."""
    with mock_aws():
        store = ObjectStore(
            settings=StorageSettings(backend="s3", bucket=BUCKET, prefix="videos/"),
            endpoint_url=None,
            access_key="t",
            secret_key="t",
        )
        store.client.create_bucket(Bucket=BUCKET)

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "10.mp4").write_bytes(b"\x00" * 512)  # pending
        (video_dir / "11.mp4").write_bytes(b"\x00" * 512)  # already uploaded

        session = get_session()
        _seed_clip(session, clip_id=10, selected=True, downloaded=True)
        _seed_clip(session, clip_id=11, selected=True, downloaded=True)
        session.get(Clip, 11).is_uploaded = True
        session.commit()
        session.close()

        settings = _make_settings(tmp_path, bucket=BUCKET, video_dir=str(video_dir))
        secrets = _make_secrets()

        upload_videos(settings, secrets)

        assert store.head("videos/10.mp4")["size"] == 512  # pending uploaded
        assert store.head("videos/11.mp4") is None  # uploaded row never touched
        session = get_session()
        assert session.get(Clip, 10).is_uploaded is True
        assert session.get(Clip, 11).is_uploaded is True
        session.close()


def test_ignores_unselected_and_undownloaded_clips(tmp_path, isolated_db):
    with mock_aws():
        store = ObjectStore(
            settings=StorageSettings(backend="s3", bucket=BUCKET, prefix="videos/"),
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
    """Build a minimal settings mock exposing only what upload_videos needs.

    ``embeddings.gemini_enabled`` is left False so ``default_cases`` resolves to
    the always-on cases (video/sandwich/audio) — at least one served_remotely,
    so the remote path is active when the token is set.
    """
    settings = MagicMock()
    settings.storage.bucket = bucket
    settings.storage.prefix = "videos/"
    settings.storage.backend = "s3"
    settings.storage.region = ""
    settings.storage.verify_concurrency = 8
    settings.storage.upload_concurrency = 4
    settings.paths.video_dir = video_dir or str(tmp_path / "videos")
    settings.embeddings.gemini_enabled = False
    return settings


def _make_secrets(*, embedder_token: str = "tok"):
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

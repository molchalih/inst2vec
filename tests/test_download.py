"""Tests for modules.download — fetch_file primitive."""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import MagicMock

import httpx
import pytest

import modules.database as db_mod
from modules import download as dl_mod
from modules.config import DownloadSettings, PathsSettings
from modules.database import Clip, User, get_session, init_db


def _dl(tmp_path, max_attempts=1, retry_delay=0, retry_jitter=0, concurrency=2):
    return (
        DownloadSettings(
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            retry_jitter=retry_jitter,
            concurrency=concurrency,
        ),
        PathsSettings(
            profile_pic_dir=str(tmp_path / "pics"),
            thumbnail_dir=str(tmp_path / "thumbs"),
            video_dir=str(tmp_path / "vids"),
            speech_audio_dir=str(tmp_path / "audio"),
            plots_dir="",
            model_path="",
            data_csv_path="",
        ),
    )


def _make_response(status_code=200, content=b"OK"):
    r = MagicMock(spec=httpx.Response)
    r.status_code = status_code
    r.content = content
    if status_code >= 400:
        r.raise_for_status.side_effect = httpx.HTTPStatusError(
            "fail", request=MagicMock(), response=r
        )
    else:
        r.raise_for_status.return_value = None
    return r


def test_fetch_file_success_writes_atomic(tmp_path, monkeypatch):
    target = tmp_path / "out.mp4"
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_response(200, b"video"))
    monkeypatch.setattr(dl_mod.time, "sleep", lambda _: None)

    ok = dl_mod.fetch_file(
        "https://example.com/v.mp4",
        str(target),
        max_attempts=3,
        retry_delay=15,
        retry_jitter=5,
    )

    assert ok is True
    assert target.read_bytes() == b"video"
    assert not (tmp_path / "out.mp4.part").exists()


def test_fetch_file_retries_then_succeeds(tmp_path, monkeypatch):
    target = tmp_path / "out.mp4"
    calls = {"n": 0}

    def fake_get(*a, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            return _make_response(500)
        return _make_response(200, b"ok")

    sleeps: list[float] = []
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(dl_mod.time, "sleep", lambda s: sleeps.append(s))

    ok = dl_mod.fetch_file(
        "u",
        str(target),
        max_attempts=3,
        retry_delay=15,
        retry_jitter=5,
    )

    assert ok is True
    assert calls["n"] == 3
    assert len(sleeps) == 2
    for s in sleeps:
        assert 15 <= s <= 20


def test_fetch_file_exhausted_returns_false(tmp_path, monkeypatch):
    target = tmp_path / "out.mp4"
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_response(500))
    monkeypatch.setattr(dl_mod.time, "sleep", lambda _: None)

    ok = dl_mod.fetch_file(
        "u",
        str(target),
        max_attempts=3,
        retry_delay=0,
        retry_jitter=0,
    )

    assert ok is False
    assert not target.exists()
    assert not (tmp_path / "out.mp4.part").exists()


def test_fetch_file_no_partial_on_write_failure(tmp_path, monkeypatch):
    """If the HTTP succeeds but os.replace fails, leave no final file."""
    target = tmp_path / "out.mp4"
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_response(200, b"x"))
    monkeypatch.setattr(dl_mod.time, "sleep", lambda _: None)

    real_replace = os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(dl_mod.os, "replace", boom)

    ok = dl_mod.fetch_file(
        "u",
        str(target),
        max_attempts=1,
        retry_delay=0,
        retry_jitter=0,
    )

    monkeypatch.setattr(dl_mod.os, "replace", real_replace)

    assert ok is False
    assert not target.exists()


# ============== Integration Tests (download_files) ==============


@pytest.fixture
def isolated_db(tmp_path):
    """Point the global engine at a fresh per-test DB, restore after."""
    original_main = db_mod._engine
    from modules import identity as id_mod

    original_id = getattr(id_mod, "_engine", None)

    init_db(f"sqlite:///{tmp_path}/main.db", f"sqlite:///{tmp_path}/id.db")
    yield

    db_mod._engine = original_main
    id_mod._engine = original_id


def _seed(session, user_id, clip_id, is_selected, is_downloaded, video_url, thumb_url):
    if not session.get(User, user_id):
        session.add(User(id=user_id, is_selected=True))
    session.add(
        Clip(
            id=clip_id,
            user_id=user_id,
            is_selected=is_selected,
            is_downloaded=is_downloaded,
            video_url=video_url,
            thumbnail_url=thumb_url,
        )
    )
    session.commit()


def test_download_files_happy_path(tmp_path, monkeypatch, isolated_db):
    session = get_session()
    _seed(session, 1, 100, True, None, "https://x/v.mp4", "https://x/t.jpg")
    session.close()

    monkeypatch.setattr(
        dl_mod, "get_profile_pic_url", lambda uid: f"https://x/{uid}.jpg"
    )
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_response(200, b"data"))
    monkeypatch.setattr(dl_mod.time, "sleep", lambda _: None)

    dl_mod.download_files(*_dl(tmp_path, max_attempts=3, concurrency=2))

    session = get_session()
    clip = session.get(Clip, 100)
    assert clip.is_downloaded is True
    session.close()
    assert (tmp_path / "vids" / "100.mp4").exists()
    assert (tmp_path / "thumbs" / "100.jpg").exists()
    assert (tmp_path / "pics" / "1.jpg").exists()


def test_download_files_failed_video_terminal(tmp_path, monkeypatch, isolated_db):
    session = get_session()
    _seed(session, 1, 100, True, None, "https://x/v.mp4", None)
    session.close()

    monkeypatch.setattr(dl_mod, "get_profile_pic_url", lambda uid: None)
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_response(500))
    monkeypatch.setattr(dl_mod.time, "sleep", lambda _: None)

    dl_mod.download_files(*_dl(tmp_path, max_attempts=2))

    session = get_session()
    clip = session.get(Clip, 100)
    assert clip.is_downloaded is False
    session.close()


def test_download_files_rerun_skips_completed(tmp_path, monkeypatch, isolated_db):
    session = get_session()
    _seed(session, 1, 100, True, True, "https://x/v.mp4", None)  # already done
    _seed(session, 1, 101, True, None, "https://x/v.mp4", None)  # pending
    session.close()

    calls = {"n": 0}

    def counting_get(*a, **kw):
        calls["n"] += 1
        return _make_response(200, b"data")

    monkeypatch.setattr(dl_mod, "get_profile_pic_url", lambda uid: None)
    monkeypatch.setattr(httpx, "get", counting_get)
    monkeypatch.setattr(dl_mod.time, "sleep", lambda _: None)

    dl_mod.download_files(*_dl(tmp_path))

    # Only clip 101 should have been fetched (one video call).
    assert calls["n"] == 1
    session = get_session()
    assert session.get(Clip, 100).is_downloaded is True
    assert session.get(Clip, 101).is_downloaded is True
    session.close()


def test_download_files_rerun_skips_failed(tmp_path, monkeypatch, isolated_db):
    session = get_session()
    _seed(session, 1, 100, True, False, "https://x/v.mp4", None)  # failed
    session.close()

    calls = {"n": 0}

    def counting_get(*a, **kw):
        calls["n"] += 1
        return _make_response(200, b"data")

    monkeypatch.setattr(dl_mod, "get_profile_pic_url", lambda uid: None)
    monkeypatch.setattr(httpx, "get", counting_get)
    monkeypatch.setattr(dl_mod.time, "sleep", lambda _: None)

    dl_mod.download_files(*_dl(tmp_path))

    assert calls["n"] == 0  # False clip is terminal


def test_download_files_missing_video_url_marks_failed(
    tmp_path, monkeypatch, isolated_db
):
    session = get_session()
    _seed(session, 1, 100, True, None, None, None)
    session.close()

    monkeypatch.setattr(dl_mod, "get_profile_pic_url", lambda uid: None)
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_response(200, b"x"))
    monkeypatch.setattr(dl_mod.time, "sleep", lambda _: None)

    dl_mod.download_files(*_dl(tmp_path))

    session = get_session()
    assert session.get(Clip, 100).is_downloaded is False
    session.close()


def test_download_files_respects_concurrency(tmp_path, monkeypatch, isolated_db):
    session = get_session()
    for cid in range(200, 212):
        _seed(session, 1, cid, True, None, f"https://x/{cid}.mp4", None)
    session.close()

    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def slow_get(*a, **kw):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return _make_response(200, b"x")

    monkeypatch.setattr(dl_mod, "get_profile_pic_url", lambda uid: None)
    monkeypatch.setattr(httpx, "get", slow_get)
    monkeypatch.setattr(dl_mod.time, "sleep", lambda _: None)

    dl_mod.download_files(*_dl(tmp_path, concurrency=3))

    assert max_in_flight <= 3

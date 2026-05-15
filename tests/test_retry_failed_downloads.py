"""Tests for scripts.retry_failed_downloads."""

from __future__ import annotations

import pytest

import modules.database as db_mod
from modules.database import Clip, User, get_session, init_db
from scripts import retry_failed_downloads as rfd


@pytest.fixture
def isolated_db(tmp_path):
    original_main = db_mod._engine
    from modules import identity as id_mod

    original_id = getattr(id_mod, "_engine", None)
    init_db(f"sqlite:///{tmp_path}/m.db", f"sqlite:///{tmp_path}/id.db")
    yield
    db_mod._engine = original_main
    id_mod._engine = original_id


def _setup():
    session = get_session()
    session.add(User(id=1, is_selected=True))
    session.add(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=False,
            video_url="https://x/10.mp4",
        )
    )
    session.add(
        Clip(
            id=11,
            user_id=1,
            is_selected=True,
            is_downloaded=False,
            video_url="https://x/11.mp4",
        )
    )
    session.add(
        Clip(
            id=12,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            video_url="https://x/12.mp4",
        )
    )
    session.commit()
    session.close()


def test_retry_recovers_successful_downloads(tmp_path, monkeypatch, isolated_db):
    _setup()

    calls: list[int] = []

    def fake_fetch(url, path, max_attempts, retry_delay, retry_jitter):
        calls.append(int(url.rsplit("/", 1)[1].split(".")[0]))
        assert max_attempts == 1
        return True

    monkeypatch.setattr(rfd, "fetch_file", fake_fetch)
    monkeypatch.setattr(rfd.time, "sleep", lambda _: None)

    rfd.retry_failed_downloads(video_dir=str(tmp_path / "v"))

    session = get_session()
    assert session.get(Clip, 10).is_downloaded is True
    assert session.get(Clip, 11).is_downloaded is True
    assert session.get(Clip, 12).is_downloaded is True
    session.close()
    assert sorted(calls) == [10, 11]


def test_retry_leaves_persistent_failures_false(tmp_path, monkeypatch, isolated_db):
    _setup()

    monkeypatch.setattr(rfd, "fetch_file", lambda *a, **kw: False)
    monkeypatch.setattr(rfd.time, "sleep", lambda _: None)

    rfd.retry_failed_downloads(video_dir=str(tmp_path / "v"))

    session = get_session()
    assert session.get(Clip, 10).is_downloaded is False
    assert session.get(Clip, 11).is_downloaded is False
    session.close()


def test_retry_no_retries_means_max_attempts_one(tmp_path, monkeypatch, isolated_db):
    _setup()
    captured: list[int] = []

    def fake_fetch(url, path, max_attempts, retry_delay, retry_jitter):
        captured.append(max_attempts)
        return True

    monkeypatch.setattr(rfd, "fetch_file", fake_fetch)
    monkeypatch.setattr(rfd.time, "sleep", lambda _: None)

    rfd.retry_failed_downloads(video_dir=str(tmp_path / "v"))

    assert captured == [1, 1]

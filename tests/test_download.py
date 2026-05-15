"""Tests for modules.download — fetch_file primitive."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import httpx

from modules import download as dl_mod


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

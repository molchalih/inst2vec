"""Tests for core.concurrency.retry_with_backoff."""

from __future__ import annotations

import pytest

from core import concurrency as cc


def test_returns_on_first_success(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(cc.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    result = cc.retry_with_backoff(fn, max_attempts=3, retry_delay=15, retry_jitter=5)

    assert result == "ok"
    assert calls["n"] == 1
    assert sleeps == []  # no sleep when first attempt succeeds


def test_retries_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(cc.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    result = cc.retry_with_backoff(fn, max_attempts=3, retry_delay=15, retry_jitter=5)

    assert result == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2
    for s in sleeps:
        assert 15 <= s <= 20


def test_reraises_after_exhaustion(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(cc.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ValueError("always")

    with pytest.raises(ValueError, match="always"):
        cc.retry_with_backoff(fn, max_attempts=3, retry_delay=0, retry_jitter=0)

    assert calls["n"] == 3
    assert len(sleeps) == 2  # sleeps between attempts, not after the last


def test_rejects_zero_max_attempts():
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        cc.retry_with_backoff(
            lambda: "x", max_attempts=0, retry_delay=0, retry_jitter=0
        )

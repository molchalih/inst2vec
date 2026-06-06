"""Telegram wrapper endpoints — feature-flagged on a configured bot token."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from swipe_anchor.backend.app import build_app
from swipe_anchor.backend.tg_auth import code_for_telegram_id
from swipe_anchor.db import create_app_engine, make_session_factory, session_scope

BOT_TOKEN = "123456:fake-bot-token"
INTERNAL_TOKEN = "internal-secret"


def _session_factory(engine):
    factory = make_session_factory(engine)

    @contextmanager
    def session_factory():
        with session_scope(factory) as s:
            yield s

    return session_factory


def make_init_data(tg_id: int, token: str = BOT_TOKEN) -> str:
    fields = {
        "user": json.dumps(
            {"id": tg_id, "first_name": "Ann", "username": "ann"},
            separators=(",", ":"),
        ),
        "auth_date": str(int(time.time())),
        "query_id": "AAESTING",
    }
    dcs = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    h = hmac.new(secret, dcs.encode(), hashlib.sha256).hexdigest()
    return urlencode({**fields, "hash": h})


@pytest.fixture
def wrapped_client(tmp_path) -> Iterator[TestClient]:
    engine = create_app_engine(str(tmp_path / "app.db"))
    app = build_app(
        session_factory=_session_factory(engine),
        token="",
        tg_bot_token=BOT_TOKEN,
        tg_internal_token=INTERNAL_TOKEN,
    )
    yield TestClient(app)


@pytest.fixture
def plain_client(tmp_path) -> Iterator[TestClient]:
    engine = create_app_engine(str(tmp_path / "app.db"))
    app = build_app(session_factory=_session_factory(engine), token="")
    yield TestClient(app)


def test_register_then_auth_returns_code(wrapped_client: TestClient) -> None:
    reg = wrapped_client.post(
        "/tg/register",
        json={"telegram_id": 42, "name": "ann"},
        headers={"X-Internal-Token": INTERNAL_TOKEN},
    )
    expected = code_for_telegram_id(42, BOT_TOKEN)
    assert expected != "tg42"  # opaque, not the raw id
    assert reg.status_code == 200
    assert reg.json() == {"ok": True, "access_code": expected}

    auth = wrapped_client.post("/tg/auth", json={"init_data": make_init_data(42)})
    assert auth.status_code == 200
    assert auth.json() == {"access_code": expected}


def test_auth_unregistered_is_403(wrapped_client: TestClient) -> None:
    r = wrapped_client.post("/tg/auth", json={"init_data": make_init_data(99)})
    assert r.status_code == 403


def test_auth_bad_signature_is_401(wrapped_client: TestClient) -> None:
    bad = make_init_data(42, token="999:wrong")
    r = wrapped_client.post("/tg/auth", json={"init_data": bad})
    assert r.status_code == 401


def test_register_without_internal_token_is_401(wrapped_client: TestClient) -> None:
    r = wrapped_client.post("/tg/register", json={"telegram_id": 42})
    assert r.status_code == 401
    r2 = wrapped_client.post(
        "/tg/register",
        json={"telegram_id": 42},
        headers={"X-Internal-Token": "wrong"},
    )
    assert r2.status_code == 401


def test_register_is_idempotent_upsert(wrapped_client: TestClient) -> None:
    h = {"X-Internal-Token": INTERNAL_TOKEN}
    wrapped_client.post("/tg/register", json={"telegram_id": 7, "name": "a"}, headers=h)
    wrapped_client.post("/tg/register", json={"telegram_id": 7, "name": "b"}, headers=h)
    ex = wrapped_client.post("/tg/exists", json={"telegram_id": 7}, headers=h)
    assert ex.status_code == 200
    assert ex.json() == {"exists": True, "access_code": code_for_telegram_id(7, BOT_TOKEN)}
    missing = wrapped_client.post("/tg/exists", json={"telegram_id": 8}, headers=h)
    assert missing.json() == {
        "exists": False,
        "access_code": code_for_telegram_id(8, BOT_TOKEN),
    }


def test_auth_disabled_when_no_bot_token(plain_client: TestClient) -> None:
    assert plain_client.post("/tg/auth", json={"init_data": "x"}).status_code == 404
    assert (
        plain_client.post("/tg/register", json={"telegram_id": 1}).status_code == 404
    )


def test_registered_tg_code_admitted_on_next_batch(wrapped_client: TestClient) -> None:
    wrapped_client.post(
        "/tg/register",
        json={"telegram_id": 42},
        headers={"X-Internal-Token": INTERNAL_TOKEN},
    )
    r = wrapped_client.post(
        "/next-batch",
        json={"n": 1},
        headers={"X-Access-Code": code_for_telegram_id(42, BOT_TOKEN)},
    )
    assert r.status_code == 200


def test_active_tg_code_auth_ok(wrapped_client: TestClient) -> None:
    wrapped_client.post(
        "/tg/register",
        json={"telegram_id": 55},
        headers={"X-Internal-Token": INTERNAL_TOKEN},
    )
    r = wrapped_client.post("/tg/auth", json={"init_data": make_init_data(55)})
    assert r.status_code == 200


def test_stats_requires_internal_token(wrapped_client: TestClient) -> None:
    assert wrapped_client.post("/tg/stats").status_code == 401
    r = wrapped_client.post("/tg/stats", headers={"X-Internal-Token": INTERNAL_TOKEN})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"totals", "response_times", "per_annotator"}
    assert "responses" in body["totals"] and "comparisons" in body["totals"]


def test_stats_disabled_when_no_bot_token(plain_client: TestClient) -> None:
    assert plain_client.post("/tg/stats").status_code == 404


def test_remind_requires_valid_init_data(wrapped_client: TestClient) -> None:
    bad = wrapped_client.post("/tg/remind", json={"init_data": "garbage"})
    assert bad.status_code == 401
    ok = wrapped_client.post("/tg/remind", json={"init_data": make_init_data(42)})
    assert ok.status_code == 200
    body = ok.json()
    assert body["ok"] is True and "due_at" in body


def test_due_reminders_requires_internal_token(wrapped_client: TestClient) -> None:
    assert wrapped_client.post("/tg/due-reminders").status_code == 401
    # Nothing is due yet (just scheduled ~20h out), so the list is empty.
    wrapped_client.post("/tg/remind", json={"init_data": make_init_data(42)})
    r = wrapped_client.post(
        "/tg/due-reminders", headers={"X-Internal-Token": INTERNAL_TOKEN}
    )
    assert r.status_code == 200
    assert r.json() == {"telegram_ids": []}


def test_remind_disabled_when_no_bot_token(plain_client: TestClient) -> None:
    assert plain_client.post("/tg/remind", json={"init_data": "x"}).status_code == 404
    assert plain_client.post("/tg/due-reminders").status_code == 404

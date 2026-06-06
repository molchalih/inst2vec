"""Unit tests for the pure Telegram WebApp initData validator (no HTTP, no DB)."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

from swipe_anchor.backend.tg_auth import (
    TelegramUser,
    code_for_telegram_id,
    validate_init_data,
)

FAKE_TOKEN = "123456:fake-bot-token"


def make_init_data(
    token: str = FAKE_TOKEN,
    *,
    user: dict | None = None,
    auth_date: int | None = None,
    extra: dict | None = None,
    tamper_hash: bool = False,
) -> str:
    """Build a *validly signed* initData string using the same algorithm as prod."""
    user = user or {"id": 42, "first_name": "Ann", "username": "ann"}
    auth_date = auth_date if auth_date is not None else int(time.time())
    fields: dict[str, str] = {
        "user": json.dumps(user, separators=(",", ":")),
        "auth_date": str(auth_date),
        "query_id": "AAESTING",
    }
    if extra:
        fields.update(extra)
    data_check_string = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret_key = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    digest = hmac.new(
        secret_key, data_check_string.encode(), hashlib.sha256
    ).hexdigest()
    if tamper_hash:
        digest = "0" * len(digest)
    return urlencode({**fields, "hash": digest})


def test_valid_signature_parses_user() -> None:
    init_data = make_init_data(user={"id": 7, "first_name": "Bo", "username": "bo"})
    result = validate_init_data(init_data, FAKE_TOKEN)
    assert isinstance(result, TelegramUser)
    assert result.id == 7
    assert result.first_name == "Bo"
    assert result.username == "bo"


def test_tampered_hash_rejected() -> None:
    init_data = make_init_data(tamper_hash=True)
    assert validate_init_data(init_data, FAKE_TOKEN) is None


def test_wrong_token_rejected() -> None:
    init_data = make_init_data(token=FAKE_TOKEN)
    assert validate_init_data(init_data, "999:other-token") is None


def test_stale_auth_date_rejected() -> None:
    old = int(time.time()) - 10_000
    init_data = make_init_data(auth_date=old)
    assert validate_init_data(init_data, FAKE_TOKEN, max_age_s=3600) is None
    assert validate_init_data(init_data, FAKE_TOKEN, max_age_s=100_000) is not None


def test_missing_user_rejected() -> None:
    fields = {"auth_date": str(int(time.time())), "query_id": "X"}
    dcs = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hmac.new(b"WebAppData", FAKE_TOKEN.encode(), hashlib.sha256).digest()
    h = hmac.new(secret, dcs.encode(), hashlib.sha256).hexdigest()
    init_data = urlencode({**fields, "hash": h})
    assert validate_init_data(init_data, FAKE_TOKEN) is None


def test_garbage_user_json_rejected() -> None:
    fields = {"user": "{not json", "auth_date": str(int(time.time()))}
    dcs = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hmac.new(b"WebAppData", FAKE_TOKEN.encode(), hashlib.sha256).digest()
    h = hmac.new(secret, dcs.encode(), hashlib.sha256).hexdigest()
    init_data = urlencode({**fields, "hash": h})
    assert validate_init_data(init_data, FAKE_TOKEN) is None


def test_empty_init_data_rejected() -> None:
    assert validate_init_data("", FAKE_TOKEN) is None
    assert validate_init_data("nonsense-no-hash", FAKE_TOKEN) is None


def test_empty_token_rejected() -> None:
    assert validate_init_data(make_init_data(), "") is None


def test_code_for_telegram_id_is_opaque_and_deterministic() -> None:
    a = code_for_telegram_id(42, "secret")
    assert a == code_for_telegram_id(42, "secret")  # deterministic per (id, secret)
    assert a != code_for_telegram_id(43, "secret")  # distinct per user
    assert a != code_for_telegram_id(42, "other-secret")  # secret-dependent
    # Opaque: the raw telegram id must NOT appear in the code (it would otherwise
    # be a guessable bearer credential).
    assert "42" not in a
    assert a.startswith("tg_")

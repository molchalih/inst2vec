"""Pure Telegram WebApp ``initData`` validation (no FastAPI, no DB).

Implements the official Telegram algorithm so it is fully unit-testable and can
be swapped out without touching HTTP or storage. Endpoints in ``app.py`` are a
thin shell over this module.

Algorithm (https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app):
    secret_key = HMAC_SHA256(key=b"WebAppData", msg=bot_token)
    data_check_string = "\\n".join(sorted "key=value" lines, hash excluded)
    valid iff HMAC_SHA256(key=secret_key, msg=data_check_string) == provided hash
Comparison is constant-time; a stale ``auth_date`` is rejected.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from urllib.parse import parse_qsl


@dataclass(frozen=True)
class TelegramUser:
    """The subset of the Telegram ``user`` object we rely on."""

    id: int
    first_name: str = ""
    last_name: str = ""
    username: str = ""
    language_code: str = ""


def code_for_telegram_id(tg_id: int, secret: str) -> str:
    """Map a Telegram user id to its (opaque, unforgeable) ``AccessCode`` key.

    The code must NOT be derivable from the telegram id alone: the existing
    ``/next-batch`` / ``/respond`` paths trust ``X-Access-Code`` directly, so a
    predictable ``tg<id>`` would let anyone bypass the initData check by guessing
    a registered user's (enumerable) telegram id. We therefore key an HMAC with a
    server-side ``secret`` (the bot token) — deterministic for both ``/tg/auth``
    and ``/tg/register`` (which share the secret), but unguessable without it.
    """
    mac = hmac.new(
        secret.encode(), f"tg-access-v1:{tg_id}".encode(), hashlib.sha256
    ).hexdigest()
    return f"tg_{mac[:32]}"


def validate_init_data(
    init_data: str,
    bot_token: str,
    *,
    max_age_s: int = 86_400,
) -> TelegramUser | None:
    """Validate signed Telegram WebApp ``initData``; return the user or ``None``.

    Returns ``None`` (never raises) for: empty input, missing/bad signature,
    missing token, stale ``auth_date``, or missing/garbage ``user`` JSON.
    """
    if not init_data or not bot_token:
        return None

    pairs = parse_qsl(init_data, keep_blank_values=True)
    if not pairs:
        return None
    data = dict(pairs)

    provided_hash = data.pop("hash", None)
    if not provided_hash:
        return None

    data_check_string = "\n".join(f"{k}={data[k]}" for k in sorted(data))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    computed = hmac.new(
        secret_key, data_check_string.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(computed, provided_hash):
        return None

    auth_date_raw = data.get("auth_date")
    if auth_date_raw is not None and auth_date_raw != "":
        try:
            auth_date = int(auth_date_raw)
        except ValueError:
            return None
        if max_age_s > 0 and (time.time() - auth_date) > max_age_s:
            return None

    user_raw = data.get("user")
    if not user_raw:
        return None
    try:
        user = json.loads(user_raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(user, dict) or "id" not in user:
        return None
    try:
        user_id = int(user["id"])
    except (ValueError, TypeError):
        return None

    return TelegramUser(
        id=user_id,
        first_name=str(user.get("first_name", "") or ""),
        last_name=str(user.get("last_name", "") or ""),
        username=str(user.get("username", "") or ""),
        language_code=str(user.get("language_code", "") or ""),
    )

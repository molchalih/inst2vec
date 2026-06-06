"""Pure, network-free bot helpers (unit-tested in isolation)."""

from __future__ import annotations

import secrets


def parse_admin_ids(raw: str | None) -> frozenset[int]:
    """Parse ``SA_TG_ADMIN_IDS`` (comma/space separated) into a set of ids.

    Malformed entries are ignored; an empty/unset value yields no admins (so the
    admin commands stay locked until an id is explicitly configured).
    """
    out: set[int] = set()
    for part in (raw or "").replace(",", " ").split():
        try:
            out.add(int(part))
        except ValueError:
            continue
    return frozenset(out)


def is_admin(user_id: int, admin_ids: frozenset[int]) -> bool:
    """True iff this telegram id is allow-listed for admin commands."""
    return user_id in admin_ids


def new_invite_code() -> str:
    """A fresh, unguessable invite payload for a ``/start`` deeplink."""
    return f"inv-{secrets.token_hex(8)}"


def build_invite_link(bot_username: str, code: str) -> str:
    """The shareable deeplink that opens the bot and carries the invite code."""
    return f"https://t.me/{bot_username}?start={code}"


def is_valid_welcome(payload: str | None, expected: str) -> bool:
    """True iff the /start deeplink payload matches the configured welcome code.

    An empty/unset ``expected`` never admits (fail closed), so a misconfigured
    bot cannot accidentally let everyone in.
    """
    if not expected:
        return False
    if payload is None:
        return False
    return payload.strip() == expected


def typing_delay_seconds(
    text: str, *, per_char: float = 0.02, floor: float = 0.5, cap: float = 3.0
) -> float:
    """Seconds to 'type' a message: proportional to length, floored and capped.

    Drives a ``ChatAction.TYPING`` then a sleep so messages feel hand-written
    instead of instant. The ``floor`` keeps even short one-liners showing a brief
    "typing…" beat (so a stepped how-to reads as a real person typing).
    """
    return min(cap, max(floor, len(text) * per_char))

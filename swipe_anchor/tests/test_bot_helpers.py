"""Pure bot helpers — no telegram client, no network."""

from __future__ import annotations

import random

from swipe_anchor.bot.copy import ALL_STICKERS, pick_sticker_id
from swipe_anchor.bot.helpers import (
    build_invite_link,
    is_admin,
    is_valid_welcome,
    new_invite_code,
    parse_admin_ids,
    typing_delay_seconds,
)


def test_is_valid_welcome_exact_match() -> None:
    assert is_valid_welcome("openme", "openme") is True


def test_is_valid_welcome_trims_whitespace() -> None:
    assert is_valid_welcome("  openme  ", "openme") is True


def test_is_valid_welcome_rejects_mismatch_and_empty() -> None:
    assert is_valid_welcome("nope", "openme") is False
    assert is_valid_welcome("", "openme") is False
    assert is_valid_welcome(None, "openme") is False
    assert is_valid_welcome("anything", "") is False


def test_typing_delay_scales_with_length_and_caps() -> None:
    short = typing_delay_seconds("hi")
    long = typing_delay_seconds("x" * 5000)
    assert 0.0 <= short < long
    assert long <= 3.0
    assert typing_delay_seconds("") >= 0.0


def test_pick_sticker_id_is_from_curated_set() -> None:
    rng = random.Random(0)
    sid = pick_sticker_id(rng=rng)
    assert sid in ALL_STICKERS


def test_parse_admin_ids_handles_separators_and_junk() -> None:
    assert parse_admin_ids("123, 456 789") == frozenset({123, 456, 789})
    assert parse_admin_ids("407683617") == frozenset({407683617})
    assert parse_admin_ids("123,nope,456") == frozenset({123, 456})
    assert parse_admin_ids("") == frozenset()
    assert parse_admin_ids(None) == frozenset()


def test_is_admin() -> None:
    admins = frozenset({407683617})
    assert is_admin(407683617, admins) is True
    assert is_admin(1, admins) is False
    assert is_admin(1, frozenset()) is False  # no admins configured -> locked


def test_new_invite_code_is_prefixed_and_unique() -> None:
    a = new_invite_code()
    b = new_invite_code()
    assert a.startswith("inv-") and b.startswith("inv-")
    assert a != b
    assert len(a) > len("inv-")


def test_build_invite_link() -> None:
    assert (
        build_invite_link("inst2vecbot", "inv-abc123")
        == "https://t.me/inst2vecbot?start=inv-abc123"
    )

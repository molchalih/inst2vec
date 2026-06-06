"""File-backed invite store — create / validate / count / revoke / persist."""

from __future__ import annotations

from swipe_anchor.bot.invites import InviteStore

TS = "2026-01-01T00:00:00+00:00"


def test_create_and_validate(tmp_path) -> None:
    store = InviteStore(tmp_path / "inv.json")
    inv = store.add("inv-1", "gym friends", now=TS)
    assert inv.code == "inv-1"
    assert inv.label == "gym friends"
    assert inv.uses == 0
    assert inv.active is True
    assert store.is_valid("inv-1") is True
    assert store.is_valid("unknown") is False
    assert store.is_valid("") is False
    assert store.is_valid(None) is False


def test_record_use_persists_across_reload(tmp_path) -> None:
    path = tmp_path / "inv.json"
    store = InviteStore(path)
    store.add("inv-1", now=TS)
    store.record_use("inv-1")
    store.record_use("inv-1")
    store.record_use("unknown")  # no-op, must not raise
    reloaded = InviteStore(path)
    assert reloaded.active()[0].uses == 2


def test_deactivate(tmp_path) -> None:
    store = InviteStore(tmp_path / "inv.json")
    store.add("inv-1", now=TS)
    assert store.deactivate("inv-1") is True
    assert store.is_valid("inv-1") is False
    assert store.deactivate("inv-1") is False  # already inactive
    assert store.deactivate("never") is False
    assert store.active() == []


def test_active_lists_only_active_sorted(tmp_path) -> None:
    store = InviteStore(tmp_path / "inv.json")
    store.add("inv-2", "b", now=TS)
    store.add("inv-1", "a", now=TS)
    store.add("inv-3", "c", now=TS)
    store.deactivate("inv-1")
    assert [i.code for i in store.active()] == ["inv-2", "inv-3"]


def test_corrupt_file_is_ignored(tmp_path) -> None:
    path = tmp_path / "inv.json"
    path.write_text("{not valid json")
    store = InviteStore(path)
    assert store.active() == []
    store.add("inv-1", now=TS)  # still usable after recovering from junk
    assert store.is_valid("inv-1") is True

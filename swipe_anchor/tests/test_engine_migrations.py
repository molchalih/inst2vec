"""Additive-column self-heal, read-only engine, and 64-bit telegram ids."""

from __future__ import annotations

from sqlalchemy import BigInteger, create_engine, inspect

from swipe_anchor.db import apply_additive_migrations, create_app_engine
from swipe_anchor.db.models import Reminder


def test_apply_additive_migrations_adds_shown_clips(tmp_path) -> None:
    # Simulate an older DB whose responses table predates shown_clips.
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE responses (response_id TEXT PRIMARY KEY)")

    apply_additive_migrations(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("responses")}
    assert "shown_clips" in cols

    apply_additive_migrations(engine)  # idempotent — no error, no duplicate


def test_create_app_engine_self_heals_missing_column(tmp_path) -> None:
    # A reopen with the default migrate=True should reconcile the column even when
    # the table was created without it.
    url = f"sqlite:///{tmp_path / 'heal.db'}"
    engine = create_engine(url.replace("sqlite:///", "sqlite:///"))
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE responses (response_id TEXT PRIMARY KEY)")
    engine.dispose()

    healed = create_app_engine(url)
    assert "shown_clips" in {
        c["name"] for c in inspect(healed).get_columns("responses")
    }


def test_migrate_false_creates_nothing(tmp_path) -> None:
    # Read-only mode (used by --dry-run): no tables created, no DDL.
    engine = create_app_engine(f"sqlite:///{tmp_path / 'empty.db'}", migrate=False)
    assert inspect(engine).get_table_names() == []


def test_reminder_telegram_id_is_64_bit() -> None:
    assert isinstance(Reminder.__table__.c.telegram_id.type, BigInteger)

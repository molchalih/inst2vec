"""WAL + busy_timeout must be applied to SQLite connections so the
single-writer embeddings sink coexists with concurrent reads."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from core.database.engine import _apply_sqlite_pragmas


def test_pragmas_applied_to_sqlite_connection(tmp_path):
    db = tmp_path / "x.db"
    engine = create_engine(f"sqlite:///{db}")
    # Register the listener the same way init_db does.
    from sqlalchemy import event

    event.listen(engine, "connect", _apply_sqlite_pragmas)
    with engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        busy = conn.execute(text("PRAGMA busy_timeout")).scalar()
    assert str(mode).lower() == "wal"
    assert int(busy) >= 5000


def test_pragmas_noop_for_non_sqlite():
    # A non-sqlite DBAPI connection (fake) must not raise.
    class _FakeCursor:
        def execute(self, *a):
            raise AssertionError("should not execute on non-sqlite")

        def close(self):
            pass

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

    # Connection record arg is unused; pass None.
    _apply_sqlite_pragmas(_FakeConn(), None)

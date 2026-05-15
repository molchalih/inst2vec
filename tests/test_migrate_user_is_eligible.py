"""Tests for scripts/migrate_user_is_eligible.py"""

from sqlalchemy import create_engine, inspect, text

from scripts.migrate_user_is_eligible import migrate_database

_LEGACY_DDL = """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        eligibility INTEGER NOT NULL DEFAULT 0
    )
"""


def _make_legacy_engine(rows: list[tuple[int, int]]):
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with eng.begin() as conn:
        conn.execute(text(_LEGACY_DDL))
        for user_id, eligibility in rows:
            conn.execute(
                text("INSERT INTO users (id, eligibility) VALUES (:id, :e)"),
                {"id": user_id, "e": eligibility},
            )
    return eng


def _read_users(eng) -> list[dict]:
    with eng.connect() as conn:
        rows = conn.execute(
            text("SELECT id, is_eligible FROM users ORDER BY id")
        ).fetchall()
        return [{"id": r[0], "is_eligible": r[1]} for r in rows]


def test_migrate_pending_to_null():
    eng = _make_legacy_engine([(1, 0)])
    migrate_database(eng)
    rows = _read_users(eng)
    assert rows == [{"id": 1, "is_eligible": None}]


def test_migrate_eligible_to_true():
    eng = _make_legacy_engine([(1, 1)])
    migrate_database(eng)
    rows = _read_users(eng)
    assert rows[0]["is_eligible"] in (1, True)


def test_migrate_disqualified_to_false():
    eng = _make_legacy_engine([(1, 2)])
    migrate_database(eng)
    rows = _read_users(eng)
    assert rows[0]["is_eligible"] in (0, False)


def test_migrate_mixed_rows():
    eng = _make_legacy_engine([(1, 0), (2, 1), (3, 2)])
    migrate_database(eng)
    rows = _read_users(eng)
    assert rows[0]["is_eligible"] is None
    assert rows[1]["is_eligible"] in (1, True)
    assert rows[2]["is_eligible"] in (0, False)


def test_migrate_drops_old_eligibility_column():
    eng = _make_legacy_engine([(1, 1)])
    migrate_database(eng)
    col_names = {c["name"] for c in inspect(eng).get_columns("users")}
    assert "eligibility" not in col_names
    assert "is_eligible" in col_names


def test_migrate_idempotent_when_already_migrated():
    eng = _make_legacy_engine([(1, 1)])
    migrate_database(eng)
    migrate_database(eng)  # second run must not crash
    rows = _read_users(eng)
    assert len(rows) == 1


def test_migrate_preserves_child_table_fk_integrity():
    """After migration, child tables must still reference `users`, not the dropped `users_old`."""
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with eng.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                eligibility INTEGER NOT NULL DEFAULT 0
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE clips (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id)
            )
        """)
        )
        conn.execute(text("INSERT INTO users (id, eligibility) VALUES (1, 1)"))
        conn.execute(text("INSERT INTO clips (id, user_id) VALUES (10, 1)"))

    migrate_database(eng)

    with eng.connect() as conn:
        # PRAGMA foreign_key_check returns rows for each FK violation.
        violations = conn.execute(text("PRAGMA foreign_key_check")).fetchall()
        assert violations == [], f"FK violations after migration: {violations}"

        clip_rows = conn.execute(
            text("SELECT id, user_id FROM clips ORDER BY id")
        ).fetchall()
        assert clip_rows == [(10, 1)]

        user_rows = conn.execute(text("SELECT id FROM users ORDER BY id")).fetchall()
        assert user_rows == [(1,)]

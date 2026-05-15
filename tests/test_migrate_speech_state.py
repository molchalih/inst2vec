"""Tests for scripts/migrate_speech_state.py"""

from sqlalchemy import create_engine, event, inspect, text

from scripts.migrate_speech_state import migrate_database

_LEGACY_CLIPS_DDL = """
    CREATE TABLE clips (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        music_id INTEGER REFERENCES music(id),
        has_speech BOOLEAN,
        is_selected BOOLEAN,
        is_downloaded BOOLEAN
    )
"""

_MUSIC_DDL = """
    CREATE TABLE music (
        id INTEGER PRIMARY KEY,
        artist TEXT NOT NULL DEFAULT '',
        track TEXT NOT NULL DEFAULT ''
    )
"""


def _make_legacy_engine(clip_rows=()):
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with eng.begin() as conn:
        conn.execute(text(_MUSIC_DDL))
        conn.execute(text(_LEGACY_CLIPS_DDL))
        conn.execute(text("INSERT INTO music (id, artist, track) VALUES (1, 'a', 't')"))
        for row in clip_rows:
            conn.execute(
                text(
                    "INSERT INTO clips (id, user_id, music_id, has_speech) "
                    "VALUES (:id, :uid, :mid, :hs)"
                ),
                row,
            )
    return eng


def test_migrate_renames_has_speech_to_is_speech_detected():
    eng = _make_legacy_engine(
        clip_rows=[{"id": 10, "uid": 1, "mid": 1, "hs": True}],
    )
    migrate_database(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("clips")}
    assert "is_speech_detected" in cols
    assert "has_speech" not in cols


def test_migrate_preserves_has_speech_values():
    eng = _make_legacy_engine(
        clip_rows=[
            {"id": 10, "uid": 1, "mid": 1, "hs": True},
            {"id": 11, "uid": 1, "mid": 1, "hs": False},
            {"id": 12, "uid": 1, "mid": None, "hs": None},
        ],
    )
    migrate_database(eng)
    with eng.connect() as conn:
        rows = conn.execute(
            text("SELECT id, is_speech_detected FROM clips ORDER BY id")
        ).fetchall()
    assert rows[0][1] in (1, True)
    assert rows[1][1] in (0, False)
    assert rows[2][1] is None


def test_migrate_is_idempotent():
    eng = _make_legacy_engine(
        clip_rows=[{"id": 10, "uid": 1, "mid": 1, "hs": True}],
    )
    migrate_database(eng)
    migrate_database(eng)
    with eng.connect() as conn:
        v = conn.execute(
            text("SELECT is_speech_detected FROM clips WHERE id=10")
        ).scalar()
    assert v in (1, True)


def test_migrate_preserves_clip_music_fk():
    eng = _make_legacy_engine(
        clip_rows=[{"id": 10, "uid": 1, "mid": 1, "hs": True}],
    )
    migrate_database(eng)
    with eng.connect() as conn:
        fks = conn.execute(text("PRAGMA foreign_key_list(clips)")).fetchall()
        targets = sorted((row[2], row[3], row[4]) for row in fks)
        assert ("music", "music_id", "id") in targets
        violations = conn.execute(text("PRAGMA foreign_key_check")).fetchall()
        assert violations == []


def test_migrate_works_with_fk_enforcement_enabled():
    """Regression: SQLite ignores ``PRAGMA foreign_keys`` inside a transaction.
    With FK enforcement on at connect-time, the table rebuild must still
    succeed — i.e. the migration must toggle FK=OFF on an AUTOCOMMIT
    connection, not inside a transaction."""
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

    @event.listens_for(eng, "connect")
    def _enable_fks(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.close()

    with eng.begin() as conn:
        conn.execute(text(_MUSIC_DDL))
        conn.execute(text(_LEGACY_CLIPS_DDL))
        conn.execute(text("INSERT INTO music (id, artist, track) VALUES (1, 'a', 't')"))
        conn.execute(
            text(
                "INSERT INTO clips (id, user_id, music_id, has_speech) "
                "VALUES (10, 1, 1, 1)"
            )
        )

    migrate_database(eng)

    with eng.connect() as conn:
        mid = conn.execute(text("SELECT music_id FROM clips WHERE id=10")).scalar()
        assert mid == 1
        violations = conn.execute(text("PRAGMA foreign_key_check")).fetchall()
        assert violations == []

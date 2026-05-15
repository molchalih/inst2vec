"""Tests for scripts/migrate_clip_downloads_refactor.py"""

from sqlalchemy import create_engine, inspect, text

from scripts.migrate_clip_downloads_refactor import migrate_database

_LEGACY_CLIPS_DDL = """
    CREATE TABLE clips (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        is_selected BOOLEAN,
        eligibility INTEGER NOT NULL DEFAULT 0
    )
"""

_LEGACY_DOWNLOADS_DDL = """
    CREATE TABLE downloads (
        entity_id INTEGER NOT NULL,
        file_type TEXT NOT NULL,
        success BOOLEAN,
        parse_available BOOLEAN,
        PRIMARY KEY (entity_id, file_type)
    )
"""


def _make_legacy_engine(clip_rows, download_rows):
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with eng.begin() as conn:
        conn.execute(text(_LEGACY_CLIPS_DDL))
        conn.execute(text(_LEGACY_DOWNLOADS_DDL))
        for row in clip_rows:
            conn.execute(
                text(
                    "INSERT INTO clips (id, user_id, is_selected, eligibility) "
                    "VALUES (:id, :uid, :sel, :el)"
                ),
                row,
            )
        for row in download_rows:
            conn.execute(
                text(
                    "INSERT INTO downloads (entity_id, file_type, success, parse_available) "
                    "VALUES (:eid, :ft, :ok, 1)"
                ),
                row,
            )
    return eng


def _read_clips(eng):
    with eng.connect() as conn:
        rows = conn.execute(
            text("SELECT id, is_selected, is_downloaded FROM clips ORDER BY id")
        ).fetchall()
        return [{"id": r[0], "is_selected": r[1], "is_downloaded": r[2]} for r in rows]


def test_migrate_adds_is_downloaded_column():
    eng = _make_legacy_engine(
        [{"id": 1, "uid": 1, "sel": True, "el": 1}],
        [],
    )
    migrate_database(eng)
    col_names = {c["name"] for c in inspect(eng).get_columns("clips")}
    assert "is_downloaded" in col_names


def test_migrate_drops_eligibility_column():
    eng = _make_legacy_engine(
        [{"id": 1, "uid": 1, "sel": True, "el": 1}],
        [],
    )
    migrate_database(eng)
    col_names = {c["name"] for c in inspect(eng).get_columns("clips")}
    assert "eligibility" not in col_names


def test_migrate_drops_downloads_table():
    eng = _make_legacy_engine(
        [{"id": 1, "uid": 1, "sel": True, "el": 1}],
        [{"eid": 1, "ft": "video", "ok": 1}],
    )
    migrate_database(eng)
    table_names = set(inspect(eng).get_table_names())
    assert "downloads" not in table_names


def test_migrate_backfills_successful_video_downloads():
    eng = _make_legacy_engine(
        [{"id": 1, "uid": 1, "sel": True, "el": 1}],
        [{"eid": 1, "ft": "video", "ok": 1}],
    )
    migrate_database(eng)
    rows = _read_clips(eng)
    assert rows[0]["is_downloaded"] in (1, True)


def test_migrate_leaves_failed_video_downloads_as_null():
    eng = _make_legacy_engine(
        [{"id": 1, "uid": 1, "sel": True, "el": 1}],
        [{"eid": 1, "ft": "video", "ok": 0}],
    )
    migrate_database(eng)
    rows = _read_clips(eng)
    assert rows[0]["is_downloaded"] is None


def test_migrate_ignores_thumbnail_downloads_when_backfilling():
    eng = _make_legacy_engine(
        [{"id": 1, "uid": 1, "sel": True, "el": 1}],
        [{"eid": 1, "ft": "thumbnail", "ok": 1}],
    )
    migrate_database(eng)
    rows = _read_clips(eng)
    assert rows[0]["is_downloaded"] is None


def test_migrate_only_backfills_selected_clips():
    eng = _make_legacy_engine(
        [
            {"id": 1, "uid": 1, "sel": True, "el": 1},
            {"id": 2, "uid": 1, "sel": False, "el": 1},
        ],
        [
            {"eid": 1, "ft": "video", "ok": 1},
            {"eid": 2, "ft": "video", "ok": 1},
        ],
    )
    migrate_database(eng)
    rows = _read_clips(eng)
    assert rows[0]["is_downloaded"] in (1, True)
    assert rows[1]["is_downloaded"] is None


def test_migrate_skips_backfill_when_is_selected_column_absent():
    """Pre-filter-refactor schemas lack is_selected; backfill must be skipped, not crash."""
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE clips ("
                "id INTEGER PRIMARY KEY, "
                "user_id INTEGER NOT NULL, "
                "eligibility INTEGER NOT NULL DEFAULT 0"
                ")"
            )
        )
        conn.execute(text(_LEGACY_DOWNLOADS_DDL))
        conn.execute(
            text("INSERT INTO clips (id, user_id, eligibility) VALUES (1, 1, 1)")
        )
        conn.execute(
            text(
                "INSERT INTO downloads (entity_id, file_type, success, parse_available) "
                "VALUES (1, 'video', 1, 1)"
            )
        )

    migrate_database(eng)

    with eng.connect() as conn:
        col_names = {c["name"] for c in inspect(eng).get_columns("clips")}
        assert "is_downloaded" in col_names
        assert "eligibility" not in col_names
        assert "downloads" not in set(inspect(eng).get_table_names())
        row = conn.execute(text("SELECT is_downloaded FROM clips")).fetchone()
        assert row[0] is None


def test_migrate_is_idempotent():
    eng = _make_legacy_engine(
        [{"id": 1, "uid": 1, "sel": True, "el": 1}],
        [{"eid": 1, "ft": "video", "ok": 1}],
    )
    migrate_database(eng)
    migrate_database(eng)
    rows = _read_clips(eng)
    assert len(rows) == 1
    assert rows[0]["is_downloaded"] in (1, True)


def test_migrate_preserves_child_table_fk_integrity():
    """clip_embeddings.clip_id must still reference the rebuilt clips table."""
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with eng.begin() as conn:
        conn.execute(text(_LEGACY_CLIPS_DDL))
        conn.execute(text(_LEGACY_DOWNLOADS_DDL))
        conn.execute(
            text("""
                CREATE TABLE clip_embeddings (
                    clip_id INTEGER NOT NULL REFERENCES clips(id),
                    embedding_case TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (clip_id, embedding_case)
                )
            """)
        )
        conn.execute(
            text(
                "INSERT INTO clips (id, user_id, is_selected, eligibility) "
                "VALUES (10, 1, 1, 1)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO clip_embeddings (clip_id, embedding_case, embedding) "
                "VALUES (10, 'video', X'00')"
            )
        )

    migrate_database(eng)

    with eng.connect() as conn:
        violations = conn.execute(text("PRAGMA foreign_key_check")).fetchall()
        assert violations == [], f"FK violations after migration: {violations}"

        rows = conn.execute(
            text("SELECT clip_id FROM clip_embeddings ORDER BY clip_id")
        ).fetchall()
        assert rows == [(10,)]


def test_migrate_preserves_clips_outbound_fks():
    """clips.user_id and clips.music_id FK constraints must survive rebuild."""
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE music (id INTEGER PRIMARY KEY)"))
        conn.execute(
            text("""
                CREATE TABLE clips (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    music_id INTEGER REFERENCES music(id),
                    is_selected BOOLEAN,
                    eligibility INTEGER NOT NULL DEFAULT 0
                )
            """)
        )
        conn.execute(text(_LEGACY_DOWNLOADS_DDL))
        conn.execute(text("INSERT INTO users (id) VALUES (1)"))
        conn.execute(text("INSERT INTO music (id) VALUES (5)"))
        conn.execute(
            text(
                "INSERT INTO clips (id, user_id, music_id, is_selected, eligibility) "
                "VALUES (10, 1, 5, 1, 1)"
            )
        )

    migrate_database(eng)

    with eng.connect() as conn:
        fks = conn.execute(text("PRAGMA foreign_key_list(clips)")).fetchall()
        fk_targets = sorted(
            (row[2], row[3], row[4]) for row in fks
        )  # (table, from, to)
        assert ("users", "user_id", "id") in fk_targets
        assert ("music", "music_id", "id") in fk_targets

        # And data is preserved
        rows = conn.execute(text("SELECT id, user_id, music_id FROM clips")).fetchall()
        assert rows == [(10, 1, 5)]

"""Tests for scripts/migrate_music_state.py"""

from sqlalchemy import create_engine, event, inspect, text

from scripts.migrate_music_state import migrate_database

_LEGACY_CLIPS_DDL = """
    CREATE TABLE clips (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        music_id INTEGER REFERENCES music(id),
        has_music BOOLEAN,
        is_selected BOOLEAN,
        is_downloaded BOOLEAN
    )
"""

_LEGACY_MUSIC_DDL = """
    CREATE TABLE music (
        id INTEGER PRIMARY KEY,
        artist TEXT NOT NULL DEFAULT '',
        track TEXT NOT NULL DEFAULT '',
        spotify_id TEXT,
        reccobeats_id TEXT,
        has_features TEXT,
        acousticness REAL,
        danceability REAL,
        energy REAL,
        instrumentalness REAL,
        key INTEGER,
        liveness REAL,
        loudness REAL,
        mode INTEGER,
        speechiness REAL,
        tempo REAL,
        valence REAL
    )
"""


def _make_legacy_engine(music_rows=(), clip_rows=()):
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with eng.begin() as conn:
        conn.execute(text(_LEGACY_MUSIC_DDL))
        conn.execute(text(_LEGACY_CLIPS_DDL))
        for row in music_rows:
            conn.execute(
                text(
                    "INSERT INTO music (id, artist, track, has_features) "
                    "VALUES (:id, :artist, :track, :hf)"
                ),
                row,
            )
        for row in clip_rows:
            conn.execute(
                text(
                    "INSERT INTO clips (id, user_id, music_id, has_music) "
                    "VALUES (:id, :uid, :mid, :hm)"
                ),
                row,
            )
    return eng


def test_migrate_renames_has_music_to_is_music_recognized():
    eng = _make_legacy_engine(
        music_rows=[{"id": 1, "artist": "a", "track": "t", "hf": None}],
        clip_rows=[{"id": 10, "uid": 1, "mid": 1, "hm": True}],
    )
    migrate_database(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("clips")}
    assert "is_music_recognized" in cols
    assert "has_music" not in cols


def test_migrate_preserves_has_music_values():
    eng = _make_legacy_engine(
        music_rows=[{"id": 1, "artist": "a", "track": "t", "hf": None}],
        clip_rows=[
            {"id": 10, "uid": 1, "mid": 1, "hm": True},
            {"id": 11, "uid": 1, "mid": 1, "hm": False},
            {"id": 12, "uid": 1, "mid": None, "hm": None},
        ],
    )
    migrate_database(eng)
    with eng.connect() as conn:
        rows = conn.execute(
            text("SELECT id, is_music_recognized FROM clips ORDER BY id")
        ).fetchall()
    assert rows[0][1] in (1, True)
    assert rows[1][1] in (0, False)
    assert rows[2][1] is None


def test_migrate_adds_is_audio_features_extracted():
    eng = _make_legacy_engine(
        music_rows=[{"id": 1, "artist": "a", "track": "t", "hf": None}]
    )
    migrate_database(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("music")}
    assert "is_audio_features_extracted" in cols


def test_migrate_drops_has_features():
    eng = _make_legacy_engine(
        music_rows=[{"id": 1, "artist": "a", "track": "t", "hf": "yes"}]
    )
    migrate_database(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("music")}
    assert "has_features" not in cols


def test_migrate_maps_has_features_yes_to_true():
    eng = _make_legacy_engine(
        music_rows=[{"id": 1, "artist": "a", "track": "t", "hf": "yes"}]
    )
    migrate_database(eng)
    with eng.connect() as conn:
        v = conn.execute(
            text("SELECT is_audio_features_extracted FROM music WHERE id=1")
        ).scalar()
    assert v in (1, True)


def test_migrate_maps_has_features_none_to_false():
    eng = _make_legacy_engine(
        music_rows=[{"id": 1, "artist": "a", "track": "t", "hf": "none"}]
    )
    migrate_database(eng)
    with eng.connect() as conn:
        v = conn.execute(
            text("SELECT is_audio_features_extracted FROM music WHERE id=1")
        ).scalar()
    assert v in (0, False)


def test_migrate_maps_has_features_null_to_null():
    eng = _make_legacy_engine(
        music_rows=[{"id": 1, "artist": "a", "track": "t", "hf": None}]
    )
    migrate_database(eng)
    with eng.connect() as conn:
        v = conn.execute(
            text("SELECT is_audio_features_extracted FROM music WHERE id=1")
        ).scalar()
    assert v is None


def test_migrate_is_idempotent():
    eng = _make_legacy_engine(
        music_rows=[{"id": 1, "artist": "a", "track": "t", "hf": "yes"}],
        clip_rows=[{"id": 10, "uid": 1, "mid": 1, "hm": True}],
    )
    migrate_database(eng)
    migrate_database(eng)
    with eng.connect() as conn:
        v = conn.execute(
            text("SELECT is_audio_features_extracted FROM music WHERE id=1")
        ).scalar()
        hm = conn.execute(
            text("SELECT is_music_recognized FROM clips WHERE id=10")
        ).scalar()
    assert v in (1, True)
    assert hm in (1, True)


def test_migrate_preserves_clip_music_fk():
    """clips.music_id → music.id FK must survive the music table rebuild."""
    eng = _make_legacy_engine(
        music_rows=[{"id": 5, "artist": "a", "track": "t", "hf": "yes"}],
        clip_rows=[{"id": 10, "uid": 1, "mid": 5, "hm": True}],
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
        conn.execute(text(_LEGACY_MUSIC_DDL))
        conn.execute(text(_LEGACY_CLIPS_DDL))
        conn.execute(
            text(
                "INSERT INTO music (id, artist, track, has_features) "
                "VALUES (5, 'a', 't', 'yes')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO clips (id, user_id, music_id, has_music) "
                "VALUES (10, 1, 5, 1)"
            )
        )

    migrate_database(eng)

    with eng.connect() as conn:
        mid = conn.execute(text("SELECT music_id FROM clips WHERE id=10")).scalar()
        assert mid == 5
        violations = conn.execute(text("PRAGMA foreign_key_check")).fetchall()
        assert violations == []

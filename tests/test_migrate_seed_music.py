"""Tests for scripts/migrate_seed_music_from_old_db.py.

Seeds a target Music table from a legacy SQLite source DB so ACR
matches reuse pre-extracted features without re-calling Spotify/RB.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.database import Base, Music, StageState, get_engine, get_session
from scripts.migrate_seed_music_from_old_db import seed_from_old_db

OLD_SCHEMA = """
CREATE TABLE music (
    id INTEGER NOT NULL PRIMARY KEY,
    artist VARCHAR NOT NULL,
    track VARCHAR NOT NULL,
    spotify_id VARCHAR,
    reccobeats_id VARCHAR,
    acousticness FLOAT,
    danceability FLOAT,
    energy FLOAT,
    instrumentalness FLOAT,
    "key" INTEGER,
    liveness FLOAT,
    loudness FLOAT,
    mode INTEGER,
    speechiness FLOAT,
    tempo FLOAT,
    valence FLOAT,
    has_features TEXT
);
"""


def _build_source(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    cols = [
        "id",
        "artist",
        "track",
        "spotify_id",
        "reccobeats_id",
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "key",
        "liveness",
        "loudness",
        "mode",
        "speechiness",
        "tempo",
        "valence",
        "has_features",
    ]
    for r in rows:
        conn.execute(
            f"INSERT INTO music ({','.join(cols)}) VALUES ({','.join(['?'] * len(cols))})",
            [r.get(c) for c in cols],
        )
    conn.commit()
    conn.close()


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    s = get_session()
    for model in (StageState, Music):
        s.query(model).delete()
    s.commit()
    try:
        yield s
    finally:
        s.rollback()
        for model in (StageState, Music):
            s.query(model).delete()
        s.commit()
        s.close()


def _full_features() -> dict:
    return dict(
        acousticness=0.1,
        danceability=0.2,
        energy=0.3,
        instrumentalness=0.4,
        key=5,
        liveness=0.6,
        loudness=-7.0,
        mode=1,
        speechiness=0.05,
        tempo=120.0,
        valence=0.5,
        has_features="yes",
    )


def test_seed_catalog_row_marks_resolved_and_extracted(db_session, tmp_path):
    """Source row with real spotify_id + real reccobeats_id + features
    must produce a Music row with recognition_status='matched',
    is_reccobeats_resolved=True, is_audio_features_extracted=True, and
    every feature column copied verbatim."""
    src = tmp_path / "old.db"
    _build_source(
        src,
        [
            {
                "id": 1,
                "artist": "A",
                "track": "T",
                "spotify_id": "sp1",
                "reccobeats_id": "rb-uuid",
                **_full_features(),
            }
        ],
    )
    seed_from_old_db(src, db_session)

    row = db_session.query(Music).filter_by(artist="A", track="T").one()
    assert row.spotify_id == "sp1"
    assert row.reccobeats_id == "rb-uuid"
    assert row.recognition_status == "matched"
    assert row.is_reccobeats_resolved is True
    assert row.is_audio_features_extracted is True
    assert row.tempo == 120.0
    assert row.key == 5


def test_seed_translates_none_sentinel_to_null(db_session, tmp_path):
    """Legacy 'none' string sentinels must become NULL; a row with
    spotify_id='none' becomes recognition_status='no_match' and
    is_reccobeats_resolved stays NULL (Stage 2 never runs without a
    Spotify match)."""
    src = tmp_path / "old.db"
    _build_source(
        src,
        [
            {
                "id": 1,
                "artist": "A",
                "track": "T",
                "spotify_id": "none",
                "reccobeats_id": "none",
                **_full_features(),
            }
        ],
    )
    seed_from_old_db(src, db_session)

    row = db_session.query(Music).filter_by(artist="A", track="T").one()
    assert row.spotify_id is None
    assert row.reccobeats_id is None
    assert row.recognition_status == "no_match"
    assert row.is_reccobeats_resolved is None
    assert row.is_audio_features_extracted is True


def test_seed_marks_upload_fallback_only_row(db_session, tmp_path):
    """Source with real spotify_id but reccobeats_id='none' represents
    a track that used the manual upload analyzer. Target row must have
    recognition_status='matched', is_reccobeats_resolved=False,
    is_audio_features_extracted=True so all four sub-stages skip."""
    src = tmp_path / "old.db"
    _build_source(
        src,
        [
            {
                "id": 1,
                "artist": "A",
                "track": "T",
                "spotify_id": "sp1",
                "reccobeats_id": "none",
                **_full_features(),
            }
        ],
    )
    seed_from_old_db(src, db_session)

    row = db_session.query(Music).filter_by(artist="A", track="T").one()
    assert row.recognition_status == "matched"
    assert row.is_reccobeats_resolved is False
    assert row.is_audio_features_extracted is True


def test_seed_skips_row_without_full_features(db_session, tmp_path):
    """A source row missing any of the 9 upload feature columns must
    still seed identifiers but is_audio_features_extracted stays NULL
    so the pipeline retries enrichment for that row."""
    src = tmp_path / "old.db"
    partial = _full_features()
    partial["tempo"] = None
    _build_source(
        src,
        [
            {
                "id": 1,
                "artist": "A",
                "track": "T",
                "spotify_id": "sp1",
                "reccobeats_id": "rb-uuid",
                **partial,
            }
        ],
    )
    seed_from_old_db(src, db_session)

    row = db_session.query(Music).filter_by(artist="A", track="T").one()
    assert row.is_audio_features_extracted is None


def test_seed_is_idempotent(db_session, tmp_path):
    """Re-running the seed must not duplicate rows or overwrite an
    already-enriched target row."""
    src = tmp_path / "old.db"
    _build_source(
        src,
        [
            {
                "id": 1,
                "artist": "A",
                "track": "T",
                "spotify_id": "sp1",
                "reccobeats_id": "rb-uuid",
                **_full_features(),
            }
        ],
    )
    seed_from_old_db(src, db_session)
    seed_from_old_db(src, db_session)

    assert db_session.query(Music).filter_by(artist="A", track="T").count() == 1


def test_seed_preserves_existing_target_when_already_enriched(db_session, tmp_path):
    """If the target already has an enriched row for (artist, track),
    the seed must NOT overwrite it."""
    db_session.add(
        Music(
            artist="A",
            track="T",
            spotify_id="current-sp",
            reccobeats_id="current-rb",
            recognition_status="matched",
            is_reccobeats_resolved=True,
            is_audio_features_extracted=True,
            tempo=999.0,
        )
    )
    db_session.commit()

    src = tmp_path / "old.db"
    _build_source(
        src,
        [
            {
                "id": 1,
                "artist": "A",
                "track": "T",
                "spotify_id": "sp1",
                "reccobeats_id": "rb-uuid",
                **_full_features(),
            }
        ],
    )
    seed_from_old_db(src, db_session)

    row = db_session.query(Music).filter_by(artist="A", track="T").one()
    assert row.tempo == 999.0
    assert row.spotify_id == "current-sp"


def test_seed_translates_case_variant_none_to_null(db_session, tmp_path):
    """Legacy DBs can store the missing-id sentinel in any casing
    ('none', 'None', 'NULL', 'null'). All variants must collapse to NULL."""
    src = tmp_path / "old.db"
    _build_source(
        src,
        [
            {
                "id": 1,
                "artist": "A",
                "track": "T",
                "spotify_id": "None",
                "reccobeats_id": "NULL",
                **_full_features(),
            }
        ],
    )
    seed_from_old_db(src, db_session)

    row = db_session.query(Music).filter_by(artist="A", track="T").one()
    assert row.spotify_id is None
    assert row.reccobeats_id is None


def test_seed_strips_artist_and_track(db_session, tmp_path):
    """Artist/track whitespace must be stripped to match the convention
    in `_get_or_create_music` (modules/music/classify.py)."""
    src = tmp_path / "old.db"
    _build_source(
        src,
        [
            {
                "id": 1,
                "artist": "  A  ",
                "track": "  T  ",
                "spotify_id": "sp1",
                "reccobeats_id": "rb-uuid",
                **_full_features(),
            }
        ],
    )
    seed_from_old_db(src, db_session)

    row = db_session.query(Music).filter_by(artist="A", track="T").one()
    assert row.spotify_id == "sp1"

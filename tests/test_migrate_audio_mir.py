"""Tests for the audio_mir table-creation migration."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from scripts.migrate_audio_mir import migrate_database


def test_migration_creates_table_on_empty_sqlite(tmp_path):
    url = f"sqlite:///{tmp_path / 'mir.db'}"
    engine = create_engine(url)
    migrate_database(engine)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='audio_mir'"
            )
        ).fetchall()
        assert rows


def test_migration_is_idempotent(tmp_path):
    url = f"sqlite:///{tmp_path / 'mir.db'}"
    engine = create_engine(url)
    migrate_database(engine)
    migrate_database(engine)
    with engine.connect() as conn:
        cols = {
            r[1] for r in conn.execute(text("PRAGMA table_info(audio_mir)")).fetchall()
        }
        assert {
            "clip_id",
            "is_mir_extracted",
            "danceability",
            "genre_labels",
            "moodtheme_scores",
        }.issubset(cols)

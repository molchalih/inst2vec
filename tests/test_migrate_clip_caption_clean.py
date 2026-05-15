"""Migration test: adds Clip.caption_clean on legacy databases."""

from __future__ import annotations

from sqlalchemy import create_engine, text


def _legacy_clips_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE clips ("
                "id INTEGER PRIMARY KEY, "
                "user_id INTEGER, "
                "caption_text TEXT, "
                "caption_language TEXT, "
                "caption_translation TEXT"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT INTO clips (id, user_id, caption_text) "
                "VALUES (1, 1, 'hello @bob world')"
            )
        )


def test_migration_adds_caption_clean(tmp_path):
    from scripts.migrate_clip_caption_clean import migrate_database

    db = tmp_path / "legacy.db"
    eng = create_engine(f"sqlite:///{db}")
    _legacy_clips_table(eng)

    migrate_database(eng)

    with eng.connect() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(clips)")).fetchall()}
        assert "caption_clean" in cols
        rows = conn.execute(text("SELECT caption_clean FROM clips")).fetchall()
        assert rows == [(None,)]


def test_migration_is_idempotent(tmp_path):
    from scripts.migrate_clip_caption_clean import migrate_database

    db = tmp_path / "legacy.db"
    eng = create_engine(f"sqlite:///{db}")
    _legacy_clips_table(eng)
    migrate_database(eng)
    migrate_database(eng)

    with eng.connect() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(clips)")).fetchall()}
        assert "caption_clean" in cols

"""Production migration: add clips.caption_clean.

Idempotent: skips if the column already exists. No backfill — the cleaner
stage fills caption_clean on the next pipeline run for clips matching
needs_caption_cleaning().

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db uv run python scripts/migrate_clip_caption_clean.py
    DATABASE_URL=postgresql://... uv run python scripts/migrate_clip_caption_clean.py
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def migrate_database(database_url_or_engine: str | Engine) -> None:
    if isinstance(database_url_or_engine, Engine):
        engine = database_url_or_engine
    else:
        engine = create_engine(database_url_or_engine)

    dialect = engine.dialect.name
    if dialect == "sqlite":
        _sqlite_migrate(engine)
    elif dialect == "postgresql":
        with engine.begin() as conn:
            _postgres_migrate(conn)
    else:
        raise RuntimeError(f"Unsupported dialect: {dialect}")


def _sqlite_migrate(engine: Engine) -> None:
    with engine.begin() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(clips)")).fetchall()}
        if "caption_clean" in cols:
            print("  SKIP: clips.caption_clean already present")
            return
        conn.execute(text("ALTER TABLE clips ADD COLUMN caption_clean TEXT"))
        print("  OK: added clips.caption_clean")


def _postgres_migrate(conn) -> None:
    cols = {
        r[0]
        for r in conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='clips'"
            )
        ).fetchall()
    }
    if "caption_clean" in cols:
        print("  SKIP: clips.caption_clean already present")
        return
    conn.execute(text("ALTER TABLE clips ADD COLUMN caption_clean TEXT"))
    print("  OK: added clips.caption_clean")


if __name__ == "__main__":
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    migrate_database(database_url)

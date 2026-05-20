"""Production migration: add music.is_reccobeats_resolved column.

Backfill rule:
    is_reccobeats_resolved =
        TRUE  iff reccobeats_id IS NOT NULL  (already matched)
        NULL  otherwise                       (retry once; new code resolves to True/False)

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db uv run python scripts/migrate_music_reccobeats_resolved.py
    DATABASE_URL=postgresql://... uv run python scripts/migrate_music_reccobeats_resolved.py
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, make_url, text
from sqlalchemy.engine import Engine


def migrate_database(database_url_or_engine: str | Engine) -> None:
    if isinstance(database_url_or_engine, Engine):
        engine = database_url_or_engine
    else:
        engine = create_engine(database_url_or_engine)

    with engine.begin() as conn:
        dialect = engine.dialect.name
        if dialect == "sqlite":
            _sqlite_migrate(conn)
        elif dialect == "postgresql":
            _postgres_migrate(conn)
        else:
            raise RuntimeError(f"Unsupported dialect: {dialect}")


def _sqlite_migrate(conn) -> None:
    cols = {r[1] for r in conn.execute(text("PRAGMA table_info(music)")).fetchall()}
    if "is_reccobeats_resolved" in cols:
        print("  SKIP: already migrated")
        return
    conn.execute(text("ALTER TABLE music ADD COLUMN is_reccobeats_resolved BOOLEAN"))
    conn.execute(
        text(
            "UPDATE music SET is_reccobeats_resolved = 1 "
            "WHERE reccobeats_id IS NOT NULL"
        )
    )
    print("  OK: added music.is_reccobeats_resolved and backfilled matched rows")


def _postgres_migrate(conn) -> None:
    exists = conn.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'music' AND column_name = 'is_reccobeats_resolved'"
        )
    ).scalar()
    if exists:
        print("  SKIP: already migrated")
        return
    conn.execute(text("ALTER TABLE music ADD COLUMN is_reccobeats_resolved BOOLEAN"))
    conn.execute(
        text(
            "UPDATE music SET is_reccobeats_resolved = TRUE "
            "WHERE reccobeats_id IS NOT NULL"
        )
    )
    print("  OK: added music.is_reccobeats_resolved and backfilled matched rows")


def main() -> int:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 1
    safe_url = make_url(url).render_as_string(hide_password=True)
    print(f"Migrating {safe_url} ...")
    migrate_database(url)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Production migration: backfill ClipEmbedding.source_hash.

The clip-embedding stage is now incremental: it re-embeds only clips whose
per-row source hash differs from what is stored on the row. After this
schema change every existing row needs its ``source_hash`` populated from
current upstream state. Without backfill the stage still works — every
NULL counts as "stale" and gets re-embedded on first run — but that costs
hours on large datasets.

This script:
  1. Adds the ``source_hash`` column to ``clip_embeddings`` if missing
     (SQLite + PostgreSQL both accept ``ALTER TABLE ADD COLUMN``).
  2. For each ``embedding_case`` present in the table, computes per-clip
     dependency hashes via ``per_clip_source_hashes_and_aggregate`` and
     writes them into rows whose ``source_hash`` is NULL.

Idempotent: re-running on a fully-backfilled DB is a no-op.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db \\
        uv run python scripts/migrate_clip_embeddings_source_hash.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.database import ClipEmbedding
from modules.embeddings.state import (
    per_clip_source_hashes_and_aggregate,
)

TABLE = "clip_embeddings"
NEW_COLUMN = "source_hash"


def _ensure_column(engine: Engine) -> None:
    inspector = inspect(engine)
    if TABLE not in inspector.get_table_names():
        print(f"Table {TABLE!r} does not exist — nothing to migrate.")
        return
    existing = {col["name"] for col in inspector.get_columns(TABLE)}
    if NEW_COLUMN in existing:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {NEW_COLUMN} TEXT"))
    print(f"  OK: added {TABLE}.{NEW_COLUMN}")


def _backfill(engine: Engine) -> None:
    inspector = inspect(engine)
    if TABLE not in inspector.get_table_names():
        return
    with Session(engine) as session:
        cases = [
            r.embedding_case
            for r in session.query(ClipEmbedding.embedding_case).distinct().all()
        ]
        for case in cases:
            null_ids = [
                r.clip_id
                for r in session.query(ClipEmbedding.clip_id)
                .filter(
                    ClipEmbedding.embedding_case == case,
                    ClipEmbedding.source_hash.is_(None),
                )
                .all()
            ]
            if not null_ids:
                print(f"  case={case!r}: no NULL rows.")
                continue

            per_clip, _ = per_clip_source_hashes_and_aggregate(
                session, case, sorted(null_ids)
            )
            updated = 0
            for clip_id, h in per_clip.items():
                session.query(ClipEmbedding).filter_by(
                    clip_id=clip_id, embedding_case=case
                ).update({ClipEmbedding.source_hash: h})
                updated += 1
            session.commit()
            print(f"  case={case!r}: backfilled {updated} rows.")


def migrate_database(engine: Engine) -> None:
    _ensure_column(engine)
    _backfill(engine)
    print("Migration complete.")


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("Set DATABASE_URL environment variable.", file=sys.stderr)
        raise SystemExit(1)
    migrate_database(create_engine(url))


if __name__ == "__main__":
    main()

"""Production migration: backfill UserEmbedding.source_hash.

Mirrors scripts/migrate_clip_embeddings_source_hash.py. The user-embedding
stage is now incremental: it recomputes only users whose per-user hash
differs from what is stored on the row. After this schema change every
existing row needs its ``source_hash`` populated from current upstream —
but **only when we can prove the stored embedding still matches current
upstream**. Without that check the backfill would stamp current hashes
onto stale rows, and the next incremental run's diff would treat those
stale embeddings as already-current.

This script:

  1. Adds the ``source_hash`` column to ``user_embeddings`` if missing
     (SQLite + PostgreSQL both accept ``ALTER TABLE ADD COLUMN``).
  2. For each ``embedding_case`` present in the table, reconstructs the
     stage's ``Fingerprint`` against current upstream and compares it to
     the case's stored ``StageState``. Only when the fingerprints match
     does it write per-user hashes onto NULL rows.

Idempotent: re-running on a fully-backfilled DB is a no-op.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db \\
        uv run python scripts/migrate_user_embeddings_source_hash.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import fingerprint as fp
from modules.database import UserEmbedding
from modules.embeddings.state import (
    get_clip_embedding_rows_for_user_aggregation,
    per_user_source_hashes,
)
from modules.embeddings.users import _compute_fingerprint

TABLE = "user_embeddings"
NEW_COLUMN = "source_hash"
STAGE = "user_embeddings"


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


def _backfill(engine: Engine, settings) -> None:
    inspector = inspect(engine)
    if TABLE not in inspector.get_table_names():
        return
    with Session(engine) as session:
        cases = [
            r.embedding_case
            for r in session.query(UserEmbedding.embedding_case).distinct().all()
        ]
        for case in cases:
            null_ids = [
                r.user_id
                for r in session.query(UserEmbedding.user_id)
                .filter(
                    UserEmbedding.embedding_case == case,
                    UserEmbedding.source_hash.is_(None),
                )
                .all()
            ]
            if not null_ids:
                print(f"  case={case!r}: no NULL rows.")
                continue

            rows = get_clip_embedding_rows_for_user_aggregation(
                session, case, settings.embeddings.exclude_disqualified_users
            )
            current = _compute_fingerprint(session, case, rows)

            if fp.is_stale(session, STAGE, case, current):
                print(
                    f"  case={case!r}: stage fingerprint missing or stale — "
                    f"leaving {len(null_ids)} row(s) NULL (next pipeline run "
                    f"will re-aggregate)."
                )
                continue

            per_user = per_user_source_hashes(rows)

            updated = 0
            skipped_orphans = 0
            for user_id in null_ids:
                h = per_user.get(user_id)
                if h is None:
                    skipped_orphans += 1
                    continue
                session.query(UserEmbedding).filter_by(
                    user_id=user_id, embedding_case=case
                ).update({UserEmbedding.source_hash: h})
                updated += 1
            session.commit()
            msg = f"  case={case!r}: backfilled {updated} row(s)"
            if skipped_orphans:
                msg += f" ({skipped_orphans} orphan(s) left NULL)"
            print(msg + ".")


def migrate_database(engine: Engine, settings) -> None:
    _ensure_column(engine)
    _backfill(engine, settings)
    print("Migration complete.")


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("Set DATABASE_URL environment variable.", file=sys.stderr)
        raise SystemExit(1)
    from modules.config import load_runtime_config

    settings, _ = load_runtime_config()
    migrate_database(create_engine(url), settings)


if __name__ == "__main__":
    main()

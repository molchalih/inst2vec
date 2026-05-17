"""Production migration: backfill ClipEmbedding.source_hash.

The clip-embedding stage is now incremental: it re-embeds only clips whose
per-row source hash differs from what is stored on the row. After this
schema change every existing row needs its ``source_hash`` populated from
current upstream state — but **only when we can prove the stored
embedding still matches current upstream**. Without that check the
backfill would stamp current hashes onto stale embedding rows, and the
next incremental run's diff would treat those stale embeddings as
already-current.

This script:

  1. Adds the ``source_hash`` column to ``clip_embeddings`` if missing
     (SQLite + PostgreSQL both accept ``ALTER TABLE ADD COLUMN``).
  2. For each ``embedding_case`` present in the table, reconstructs the
     runner's ``Fingerprint`` against current upstream and compares it
     to the case's stored ``StageState``. Only when the fingerprints
     match (i.e. the stage was sealed against today's upstream) does it
     write the per-clip hash onto NULL rows. Otherwise it leaves them
     NULL — a NULL counts as "stale" and the next pipeline run will
     re-embed, which is the safe fallback.

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

from modules import fingerprint as fp
from modules.database import ClipEmbedding
from modules.embeddings.cases import CASE_REGISTRY, case_config_identity
from modules.embeddings.state import (
    get_clip_embedding_candidates,
    per_clip_source_hashes_and_aggregate,
)

TABLE = "clip_embeddings"
NEW_COLUMN = "source_hash"
STAGE = "clip_embeddings"


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

            spec = CASE_REGISTRY.get(case)
            if spec is None:
                print(
                    f"  case={case!r}: unknown case — leaving "
                    f"{len(null_ids)} row(s) NULL."
                )
                continue

            # Reproduce the runner's Fingerprint over current upstream.
            candidates = get_clip_embedding_candidates(
                session, settings.embeddings.exclude_disqualified_users
            )
            candidate_ids = sorted(c.id for c in candidates)
            per_clip, dep_agg = per_clip_source_hashes_and_aggregate(
                session, case, candidate_ids, settings=settings
            )
            current = fp.Fingerprint(
                data=fp.hash_rows((cid,) for cid in candidate_ids),
                config=fp.hash_text(case_config_identity(spec, settings)),
                dependency=dep_agg,
            )

            if fp.is_stale(session, STAGE, case, current):
                print(
                    f"  case={case!r}: stage fingerprint missing or stale — "
                    f"leaving {len(null_ids)} row(s) NULL (next pipeline run "
                    f"will re-embed)."
                )
                continue

            updated = 0
            skipped_orphans = 0
            for clip_id in null_ids:
                h = per_clip.get(clip_id)
                if h is None:
                    # Row exists for a clip no longer in the candidate set
                    # (deselected, undownloaded, or ineligible user). Leave
                    # NULL — aggregation already filters these out.
                    skipped_orphans += 1
                    continue
                session.query(ClipEmbedding).filter_by(
                    clip_id=clip_id, embedding_case=case
                ).update({ClipEmbedding.source_hash: h})
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

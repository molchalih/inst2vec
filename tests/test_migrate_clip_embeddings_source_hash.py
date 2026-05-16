"""Tests for scripts/migrate_clip_embeddings_source_hash.py.

The migration must:
  1. Add the source_hash column to clip_embeddings if missing.
  2. Backfill NULL source_hash values with the per-clip dependency hash
     computed from current upstream state.
  3. Be idempotent: re-running on a fully-backfilled DB must change nothing.
"""

from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from modules.database import Base
from scripts.migrate_clip_embeddings_source_hash import migrate_database


def _legacy_clip_embeddings_ddl() -> str:
    """Pre-source_hash schema for clip_embeddings on SQLite."""
    return """
    CREATE TABLE clip_embeddings (
        clip_id BIGINT NOT NULL,
        embedding_case TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
        PRIMARY KEY (clip_id, embedding_case),
        FOREIGN KEY (clip_id) REFERENCES clips(id),
        CONSTRAINT uq_clip_embeddings_clip_case UNIQUE (clip_id, embedding_case)
    )
    """


def _seed_minimal_upstream(conn) -> None:
    # The migration computes source_hash via dependency_rows_for_case, which
    # queries clips and music. Seed the minimum the helper needs.
    conn.execute(
        text("INSERT INTO users (id, is_selected, is_eligible) VALUES (1, 1, 1)")
    )
    conn.execute(
        text(
            "INSERT INTO clips (id, user_id, is_selected, is_downloaded) "
            "VALUES (10, 1, 1, 1), (11, 1, 1, 1)"
        )
    )


def test_migration_adds_column_and_backfills(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")

    # Build the full schema EXCEPT clip_embeddings, then create the legacy
    # clip_embeddings table by hand so the migration has something to ALTER.
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE clip_embeddings"))
        conn.execute(text(_legacy_clip_embeddings_ddl()))
        _seed_minimal_upstream(conn)
        conn.execute(
            text(
                "INSERT INTO clip_embeddings (clip_id, embedding_case, embedding) "
                "VALUES (10, 'video', X'00'), (11, 'video', X'00')"
            )
        )

    # Pre-condition: column does not exist.
    cols_before = {c["name"] for c in inspect(eng).get_columns("clip_embeddings")}
    assert "source_hash" not in cols_before

    migrate_database(eng)

    # Column exists and rows are backfilled.
    cols_after = {c["name"] for c in inspect(eng).get_columns("clip_embeddings")}
    assert "source_hash" in cols_after

    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT clip_id, source_hash FROM clip_embeddings "
                "WHERE embedding_case = 'video' ORDER BY clip_id"
            )
        ).all()
    assert len(rows) == 2
    assert all(r.source_hash is not None for r in rows), (
        "every row must carry a non-NULL source_hash after backfill"
    )

    # Sanity: per-clip hashes match what the runner would compute.
    from sqlalchemy.orm import Session as _Session

    from modules.embeddings.state import (
        per_clip_source_hashes_and_aggregate,
    )

    with _Session(eng) as session:
        per_clip, _ = per_clip_source_hashes_and_aggregate(session, "video", [10, 11])
    by_id = dict(rows)
    assert by_id[10] == per_clip[10]
    assert by_id[11] == per_clip[11]


def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "fresh.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")

    Base.metadata.create_all(
        eng
    )  # new-schema clip_embeddings (source_hash already present)
    with eng.begin() as conn:
        _seed_minimal_upstream(conn)
        conn.execute(
            text(
                "INSERT INTO clip_embeddings (clip_id, embedding_case, embedding, source_hash) "
                "VALUES (10, 'video', X'00', 'preexisting'), (11, 'video', X'00', NULL)"
            )
        )

    migrate_database(eng)

    with eng.connect() as conn:
        rows = {
            r.clip_id: r.source_hash
            for r in conn.execute(
                text("SELECT clip_id, source_hash FROM clip_embeddings")
            ).all()
        }
    # Existing non-NULL hash must not be overwritten; NULL is backfilled.
    assert rows[10] == "preexisting"
    assert rows[11] is not None

    # Second run: nothing changes.
    migrate_database(eng)
    with eng.connect() as conn:
        rows_after = {
            r.clip_id: r.source_hash
            for r in conn.execute(
                text("SELECT clip_id, source_hash FROM clip_embeddings")
            ).all()
        }
    assert rows_after == rows

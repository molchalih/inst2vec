"""Tests for scripts/migrate_clip_embeddings_source_hash.py.

The migration must:
  1. Add the source_hash column to clip_embeddings if missing.
  2. Backfill NULL source_hash only when the case's stored StageState
     fingerprint matches current upstream — proof that the stored
     embeddings still match current inputs. Otherwise leave NULL so the
     next pipeline run re-embeds.
  3. Be idempotent: re-running on a fully-backfilled DB must change nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from modules import fingerprint as fp
from modules.database import Base, StageState
from modules.embeddings.cases import CASE_REGISTRY, case_config_identity
from modules.embeddings.state import per_clip_source_hashes_and_aggregate
from scripts.migrate_clip_embeddings_source_hash import migrate_database


def _settings() -> SimpleNamespace:
    """Minimal settings stub that satisfies case_config_identity + the
    migration's candidate filter."""
    return SimpleNamespace(
        paths=SimpleNamespace(model_path="/fake/Qwen3-VL-Embedding-8B"),
        embeddings=SimpleNamespace(
            exclude_disqualified_users=False,
            embed_max_length=1024,
            adaptive_max_frames=8,
            adaptive_default_fps=1.0,
        ),
    )


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
    conn.execute(
        text("INSERT INTO users (id, is_selected, is_eligible) VALUES (1, 1, 1)")
    )
    conn.execute(
        text(
            "INSERT INTO clips (id, user_id, is_selected, is_downloaded) "
            "VALUES (10, 1, 1, 1), (11, 1, 1, 1)"
        )
    )


def _seed_matching_stage_state(eng, case: str, settings: SimpleNamespace) -> None:
    """Pre-seal the case's StageState so the migration recognizes the
    stored embeddings as current and is allowed to backfill hashes."""
    spec = CASE_REGISTRY[case]
    with Session(eng) as s:
        candidate_ids = [10, 11]
        _, dep_agg = per_clip_source_hashes_and_aggregate(s, case, candidate_ids)
        current = fp.Fingerprint(
            data=fp.hash_rows((cid,) for cid in candidate_ids),
            config=fp.hash_text(case_config_identity(spec, settings)),
            dependency=dep_agg,
        )
        fp.mark_complete(s, "clip_embeddings", case, current)
        s.commit()


def test_migration_adds_column_and_backfills_when_stage_is_current(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")

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

    settings = _settings()
    _seed_matching_stage_state(eng, "video", settings)

    cols_before = {c["name"] for c in inspect(eng).get_columns("clip_embeddings")}
    assert "source_hash" not in cols_before

    migrate_database(eng, settings)

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
        "every row must carry a non-NULL source_hash after a safe backfill"
    )

    with Session(eng) as session:
        per_clip, _ = per_clip_source_hashes_and_aggregate(session, "video", [10, 11])
    by_id = dict(rows)
    assert by_id[10] == per_clip[10]
    assert by_id[11] == per_clip[11]


def test_migration_leaves_null_when_no_stage_state(tmp_path):
    """Without a StageState row we cannot prove embeddings match current
    upstream — leaving rows NULL is the only safe option."""
    db_path = tmp_path / "no_state.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")

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

    migrate_database(eng, _settings())

    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT clip_id, source_hash FROM clip_embeddings "
                "WHERE embedding_case = 'video' ORDER BY clip_id"
            )
        ).all()
    assert len(rows) == 2
    assert all(r.source_hash is None for r in rows), (
        "absent StageState ⇒ leave NULL; next pipeline run re-embeds"
    )


def test_migration_leaves_null_when_stage_state_stale(tmp_path):
    """A stale StageState (upstream changed since seal) must NOT result in
    current hashes being stamped onto old embeddings — otherwise the
    incremental runner would see matching source_hash values and treat
    stale embeddings as up-to-date forever."""
    db_path = tmp_path / "stale_state.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")

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

    # Seed a StageState with deliberately wrong hashes — simulates the
    # case where upstream drifted after the last seal.
    with Session(eng) as s:
        s.merge(
            StageState(
                stage_name="clip_embeddings",
                scope_key="video",
                data_hash="STALE_DATA",
                config_hash="STALE_CONFIG",
                dependency_hash="STALE_DEPENDENCY",
            )
        )
        s.commit()

    migrate_database(eng, _settings())

    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT clip_id, source_hash FROM clip_embeddings "
                "WHERE embedding_case = 'video' ORDER BY clip_id"
            )
        ).all()
    assert all(r.source_hash is None for r in rows), (
        "stale StageState must result in NULL backfill, not stamped-as-current"
    )


def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "fresh.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")

    Base.metadata.create_all(eng)  # new-schema clip_embeddings (source_hash present)
    with eng.begin() as conn:
        _seed_minimal_upstream(conn)
        conn.execute(
            text(
                "INSERT INTO clip_embeddings (clip_id, embedding_case, embedding, source_hash) "
                "VALUES (10, 'video', X'00', 'preexisting'), "
                "(11, 'video', X'00', NULL)"
            )
        )

    settings = _settings()
    _seed_matching_stage_state(eng, "video", settings)
    migrate_database(eng, settings)

    with eng.connect() as conn:
        rows = {
            r.clip_id: r.source_hash
            for r in conn.execute(
                text("SELECT clip_id, source_hash FROM clip_embeddings")
            ).all()
        }
    # Existing non-NULL hash must not be overwritten; NULL gets backfilled
    # because the stage fingerprint matches current upstream.
    assert rows[10] == "preexisting"
    assert rows[11] is not None

    # Second run: nothing changes.
    migrate_database(eng, settings)
    with eng.connect() as conn:
        rows_after = {
            r.clip_id: r.source_hash
            for r in conn.execute(
                text("SELECT clip_id, source_hash FROM clip_embeddings")
            ).all()
        }
    assert rows_after == rows

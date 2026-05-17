"""Tests for scripts/migrate_user_embeddings_source_hash.py.

The migration must:
  1. Add the source_hash column to user_embeddings if missing.
  2. Backfill NULL source_hash only when the case's stored StageState
     fingerprint matches current upstream — proof that the stored
     embeddings still match current inputs. Otherwise leave NULL so the
     next pipeline run re-aggregates.
  3. Be idempotent: re-running on a fully-backfilled DB must change nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from modules import fingerprint as fp
from modules.database import Base, StageState
from modules.embeddings.state import get_clip_embedding_rows_for_user_aggregation
from modules.embeddings.users import _compute_fingerprint
from scripts.migrate_user_embeddings_source_hash import migrate_database

pytestmark = pytest.mark.xfail(
    reason="lands together with state helpers (task 5) and users.py refactor (task 6)",
    strict=False,
)


def _settings(exclude_disqualified_users: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        embeddings=SimpleNamespace(
            exclude_disqualified_users=exclude_disqualified_users,
        )
    )


def _legacy_user_embeddings_ddl() -> str:
    return """
    CREATE TABLE user_embeddings (
        user_id INTEGER NOT NULL,
        embedding_case TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
        PRIMARY KEY (user_id, embedding_case),
        FOREIGN KEY (user_id) REFERENCES users(id),
        CONSTRAINT uq_user_embeddings_user_case UNIQUE (user_id, embedding_case)
    )
    """


def _seed_minimal_upstream(conn) -> None:
    conn.execute(
        text("INSERT INTO users (id, is_selected, is_eligible) VALUES (1, 1, 1)")
    )
    conn.execute(
        text(
            "INSERT INTO clips (id, user_id, is_selected, is_downloaded) "
            "VALUES (10, 1, 1, 1)"
        )
    )
    conn.execute(
        text(
            "INSERT INTO clip_embeddings (clip_id, embedding_case, embedding) "
            "VALUES (10, 'video', X'0000803F')"
        )
    )


def _seed_matching_stage_state(eng, case: str) -> None:
    with Session(eng) as s:
        rows = get_clip_embedding_rows_for_user_aggregation(s, case, False)
        current = _compute_fingerprint(s, case, rows)
        fp.mark_complete(s, "user_embeddings", case, current)
        s.commit()


def test_migration_adds_column_and_backfills_when_stage_is_current(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE user_embeddings"))
        conn.execute(text(_legacy_user_embeddings_ddl()))
        _seed_minimal_upstream(conn)
        conn.execute(
            text(
                "INSERT INTO user_embeddings (user_id, embedding_case, embedding) "
                "VALUES (1, 'video', X'0000803F')"
            )
        )
    _seed_matching_stage_state(eng, "video")

    cols_before = {c["name"] for c in inspect(eng).get_columns("user_embeddings")}
    assert "source_hash" not in cols_before

    migrate_database(eng, _settings())

    cols_after = {c["name"] for c in inspect(eng).get_columns("user_embeddings")}
    assert "source_hash" in cols_after

    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT user_id, source_hash FROM user_embeddings "
                "WHERE embedding_case = 'video' ORDER BY user_id"
            )
        ).all()
    assert len(rows) == 1 and rows[0].source_hash is not None


def test_migration_leaves_null_when_no_stage_state(tmp_path):
    db_path = tmp_path / "no_state.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE user_embeddings"))
        conn.execute(text(_legacy_user_embeddings_ddl()))
        _seed_minimal_upstream(conn)
        conn.execute(
            text(
                "INSERT INTO user_embeddings (user_id, embedding_case, embedding) "
                "VALUES (1, 'video', X'0000803F')"
            )
        )

    migrate_database(eng, _settings())

    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT user_id, source_hash FROM user_embeddings "
                "WHERE embedding_case = 'video' ORDER BY user_id"
            )
        ).all()
    assert rows and all(r.source_hash is None for r in rows)


def test_migration_leaves_null_when_stage_state_stale(tmp_path):
    db_path = tmp_path / "stale_state.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE user_embeddings"))
        conn.execute(text(_legacy_user_embeddings_ddl()))
        _seed_minimal_upstream(conn)
        conn.execute(
            text(
                "INSERT INTO user_embeddings (user_id, embedding_case, embedding) "
                "VALUES (1, 'video', X'0000803F')"
            )
        )
    with Session(eng) as s:
        s.merge(
            StageState(
                stage_name="user_embeddings",
                scope_key="video",
                data_hash="STALE",
                config_hash="STALE",
                dependency_hash="STALE",
            )
        )
        s.commit()

    migrate_database(eng, _settings())

    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT user_id, source_hash FROM user_embeddings "
                "WHERE embedding_case = 'video' ORDER BY user_id"
            )
        ).all()
    assert all(r.source_hash is None for r in rows)


def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "fresh.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        _seed_minimal_upstream(conn)
        conn.execute(
            text(
                "INSERT INTO user_embeddings (user_id, embedding_case, embedding, source_hash) "
                "VALUES (1, 'video', X'0000803F', 'preexisting')"
            )
        )
    _seed_matching_stage_state(eng, "video")

    migrate_database(eng, _settings())
    with eng.connect() as conn:
        before = {
            r.user_id: r.source_hash
            for r in conn.execute(
                text("SELECT user_id, source_hash FROM user_embeddings")
            ).all()
        }
    assert before[1] == "preexisting"

    migrate_database(eng, _settings())
    with eng.connect() as conn:
        after = {
            r.user_id: r.source_hash
            for r in conn.execute(
                text("SELECT user_id, source_hash FROM user_embeddings")
            ).all()
        }
    assert after == before

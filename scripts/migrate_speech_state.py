"""Production migration: rename clips.has_speech → is_speech_detected.

Backfill rule:
    clips.is_speech_detected = clips.has_speech (value-preserving)

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db uv run python scripts/migrate_speech_state.py
    DATABASE_URL=postgresql://... uv run python scripts/migrate_speech_state.py
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
        _run_sqlite_migration(engine)
    elif dialect == "postgresql":
        with engine.begin() as conn:
            _postgres_migrate(conn)
    else:
        raise RuntimeError(f"Unsupported dialect: {dialect}")


def _run_sqlite_migration(engine: Engine) -> None:
    # SQLite silently ignores ``PRAGMA foreign_keys`` inside a transaction, and
    # FK enforcement is a per-connection setting. Toggle PRAGMAs via the raw
    # DBAPI cursor (bypassing SQLAlchemy's autobegin), then run the migration
    # body transactionally on the same connection.
    with engine.connect() as conn:
        dbapi_conn = conn.connection.dbapi_connection
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA foreign_keys = OFF")
            cur.execute("PRAGMA legacy_alter_table = ON")
        finally:
            cur.close()

        try:
            with conn.begin():
                _sqlite_migrate(conn)
        finally:
            cur = dbapi_conn.cursor()
            try:
                cur.execute("PRAGMA legacy_alter_table = OFF")
                cur.execute("PRAGMA foreign_keys = ON")
                violations = list(cur.execute("PRAGMA foreign_key_check"))
            finally:
                cur.close()
        if violations:
            raise RuntimeError(f"FK integrity violated after migration: {violations}")


def _sqlite_migrate(conn) -> None:
    clip_cols = {
        r[1] for r in conn.execute(text("PRAGMA table_info(clips)")).fetchall()
    }

    fully_migrated = "has_speech" not in clip_cols and "is_speech_detected" in clip_cols
    if fully_migrated:
        print("  SKIP: already migrated")
        return

    if "is_speech_detected" not in clip_cols:
        conn.execute(text("ALTER TABLE clips ADD COLUMN is_speech_detected BOOLEAN"))
        print("  OK: added clips.is_speech_detected")

    if "has_speech" in clip_cols:
        conn.execute(text("UPDATE clips SET is_speech_detected = has_speech"))
        print("  OK: backfilled is_speech_detected from has_speech")
        _drop_has_speech_sqlite(conn)
        print("  OK: dropped clips.has_speech")


def _drop_has_speech_sqlite(conn) -> None:
    """SQLite < 3.35 needs a table rebuild to drop a column. Rebuild even on
    3.35+ to keep behavior uniform across SQLite versions in CI.

    Assumes the caller already toggled ``PRAGMA foreign_keys = OFF`` and
    ``PRAGMA legacy_alter_table = ON`` outside any active transaction
    (SQLite ignores those PRAGMAs inside one)."""
    fk_rows = conn.execute(text("PRAGMA foreign_key_list(clips)")).fetchall()
    fk_by_id: dict[int, list[tuple[int, str, str, str]]] = {}
    for row in fk_rows:
        fk_id, seq, fk_table, fk_from, fk_to = row[0], row[1], row[2], row[3], row[4]
        fk_by_id.setdefault(fk_id, []).append((seq, fk_from, fk_table, fk_to))

    conn.execute(text("ALTER TABLE clips RENAME TO clips_old"))

    pragma = conn.execute(text("PRAGMA table_info(clips_old)")).fetchall()
    col_defs: list[str] = []
    kept_cols: list[str] = []
    for row in pragma:
        name, typ, notnull, dflt, pk = row[1], row[2], row[3], row[4], row[5]
        if name == "has_speech":
            continue
        kept_cols.append(name)
        pk_str = " PRIMARY KEY" if pk else ""
        nn_str = " NOT NULL" if notnull else ""
        df_str = f" DEFAULT {dflt}" if dflt is not None else ""
        col_defs.append(f"{name} {typ}{pk_str}{nn_str}{df_str}")

    fk_defs: list[str] = []
    for fk_id in sorted(fk_by_id.keys()):
        cols_in_fk = sorted(fk_by_id[fk_id], key=lambda x: x[0])
        from_cols = [col[1] for col in cols_in_fk]
        fk_table = cols_in_fk[0][2]
        to_cols = [col[3] for col in cols_in_fk]
        fk_defs.append(
            f"FOREIGN KEY ({', '.join(from_cols)}) "
            f"REFERENCES {fk_table}({', '.join(to_cols)})"
        )

    all_defs = col_defs + fk_defs
    conn.execute(text("CREATE TABLE clips (\n  " + ",\n  ".join(all_defs) + "\n)"))
    conn.execute(
        text(
            f"INSERT INTO clips ({', '.join(kept_cols)}) "
            f"SELECT {', '.join(kept_cols)} FROM clips_old"
        )
    )
    conn.execute(text("DROP TABLE clips_old"))


def _postgres_migrate(conn) -> None:
    clip_cols = {
        r[0]
        for r in conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='clips'"
            )
        ).fetchall()
    }

    if "is_speech_detected" not in clip_cols:
        conn.execute(text("ALTER TABLE clips ADD COLUMN is_speech_detected BOOLEAN"))

    if "has_speech" in clip_cols:
        conn.execute(text("UPDATE clips SET is_speech_detected = has_speech"))
        conn.execute(text("ALTER TABLE clips DROP COLUMN has_speech"))
    print("  OK: postgres migration complete")


if __name__ == "__main__":
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    migrate_database(database_url)

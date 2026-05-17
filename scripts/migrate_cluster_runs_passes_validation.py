"""Production migration: replace legacy `cluster_runs` columns with `passes_validation`.

Removed columns (legacy validation/idempotence):
    eligibility, in_current_grid, dataset_hash, validation_config_hash

Added column:
    passes_validation BOOLEAN NULL  (tri-state: None/True/False)

SQLite:     rebuild-table approach (RENAME → recreate from ORM metadata → INSERT
            SELECT overlapping columns → DROP old).
PostgreSQL: ALTER TABLE DROP COLUMN per legacy column + ADD COLUMN passes_validation.

Idempotent: detects schema state via PRAGMA / information_schema and skips work
if the table is already on the new schema or does not exist.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db \\
        uv run python scripts/migrate_cluster_runs_passes_validation.py
    DATABASE_URL=postgresql://... \\
        uv run python scripts/migrate_cluster_runs_passes_validation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

# Allow `from core.database import ...` when run as a top-level script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import Base

TABLE = "cluster_runs"
LEGACY_COLUMNS: tuple[str, ...] = (
    "eligibility",
    "in_current_grid",
    "dataset_hash",
    "validation_config_hash",
)
NEW_COLUMN = "passes_validation"


def _existing_columns(engine: Engine, table: str) -> set[str]:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table)}


def _migrate_sqlite(engine: Engine, existing: set[str]) -> None:
    new_table_meta = Base.metadata.tables[TABLE]
    new_columns = {col.name for col in new_table_meta.columns}
    overlap = sorted(existing & new_columns)

    print(f"SQLite detected — rebuilding {TABLE} to apply new schema...")

    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {TABLE} RENAME TO {TABLE}_old"))
        # Create new table (and any missing tables) from ORM metadata.
        Base.metadata.create_all(conn)

        cols_sql = ", ".join(overlap)
        conn.execute(
            text(f"INSERT INTO {TABLE} ({cols_sql}) SELECT {cols_sql} FROM {TABLE}_old")
        )
        conn.execute(text(f"DROP TABLE {TABLE}_old"))

    dropped = sorted(existing - new_columns)
    added = sorted(new_columns - existing)
    print(f"  OK: dropped {dropped or '[]'}, added {added or '[]'}")
    print("Migration complete.")


def _migrate_postgres(engine: Engine, existing: set[str]) -> None:
    print(f"PostgreSQL detected — altering {TABLE} in place...")
    with engine.begin() as conn:
        for col in LEGACY_COLUMNS:
            if col in existing:
                conn.execute(text(f"ALTER TABLE {TABLE} DROP COLUMN {col}"))
                print(f"  OK: dropped {TABLE}.{col}")
        if NEW_COLUMN not in existing:
            conn.execute(
                text(f"ALTER TABLE {TABLE} ADD COLUMN {NEW_COLUMN} BOOLEAN NULL")
            )
            print(f"  OK: added {TABLE}.{NEW_COLUMN}")
    print("Migration complete.")


def migrate_database(engine: Engine) -> None:
    existing = _existing_columns(engine, TABLE)
    if not existing:
        print(f"Table {TABLE!r} does not exist — nothing to migrate.")
        return

    has_legacy = any(col in existing for col in LEGACY_COLUMNS)
    has_new = NEW_COLUMN in existing
    if has_new and not has_legacy:
        print(f"Table {TABLE!r} already on new schema — nothing to migrate.")
        return

    dialect = engine.dialect.name
    if dialect == "sqlite":
        _migrate_sqlite(engine, existing)
    elif dialect == "postgresql":
        _migrate_postgres(engine, existing)
    else:
        raise SystemExit(f"Unsupported dialect: {dialect}")


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("Set DATABASE_URL environment variable.", file=sys.stderr)
        raise SystemExit(1)
    migrate_database(create_engine(url))


if __name__ == "__main__":
    main()

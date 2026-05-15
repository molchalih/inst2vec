"""Production migration: convert INT flag columns to BOOLEAN.

SQLite: validates data integrity only (no schema change needed;
        SQLAlchemy stores Boolean as INTEGER in SQLite).
PostgreSQL: runs ALTER TABLE ... ALTER COLUMN ... TYPE BOOLEAN for each column.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db uv run python scripts/migrate_bool_columns.py
    DATABASE_URL=postgresql://... uv run python scripts/migrate_bool_columns.py
"""

import os
import sys

from sqlalchemy import create_engine, text

BOOL_COLUMNS: list[tuple[str, str]] = [
    ("users", "user_disqualified"),
    ("clips", "has_music"),
    ("clips", "is_music_recognized"),
    ("clips", "is_speech_detected"),
    ("clips", "disqualified"),
    ("cluster_runs", "disqualified"),
    ("cluster_runs", "in_current_grid"),
]


def _validate_sqlite(conn) -> None:
    print("SQLite detected — validating data integrity...")
    for table, col in BOOL_COLUMNS:
        exists = conn.execute(
            text("SELECT COUNT(*) FROM pragma_table_info(:tbl) WHERE name=:col"),
            {"tbl": table, "col": col},
        ).scalar()
        if not exists:
            print(f"  SKIP: {table}.{col} (column not present)")
            continue
        bad = conn.execute(
            text(
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE {col} NOT IN (0, 1) AND {col} IS NOT NULL"
            )
        ).scalar()
        if bad:
            print(
                f"  ERROR: {table}.{col} has {bad} non-boolean value(s). "
                "Fix before migrating.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  OK: {table}.{col}")
    print("All columns validated. No schema change needed for SQLite.")


def _migrate_postgres(conn) -> None:
    print("PostgreSQL detected — altering column types...")
    for table, col in BOOL_COLUMNS:
        exists = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_name=:tbl AND column_name=:col"
            ),
            {"tbl": table, "col": col},
        ).scalar()
        if not exists:
            print(f"  SKIP: {table}.{col} (column not present)")
            continue
        sql = (
            f"ALTER TABLE {table} ALTER COLUMN {col} "
            f"TYPE BOOLEAN USING ({col}::boolean)"
        )
        print(f"  Running: {sql}")
        conn.execute(text(sql))
        print(f"  OK: {table}.{col}")
    conn.execute(text("COMMIT"))
    print("Migration complete.")


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("Set DATABASE_URL environment variable.", file=sys.stderr)
        sys.exit(1)

    engine = create_engine(url)
    dialect = engine.dialect.name

    with engine.connect() as conn:
        if dialect == "sqlite":
            _validate_sqlite(conn)
        elif dialect == "postgresql":
            _migrate_postgres(conn)
        else:
            print(f"Unsupported dialect: {dialect}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

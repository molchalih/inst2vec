"""Production migration: rename legacy boolean-style eligibility columns to `eligibility`
and remap stored values NULL/0/1 → 0/1/2 (PENDING/ELIGIBLE/DISQUALIFIED).

SQLite:  rebuild-table approach (RENAME → recreate from ORM metadata → INSERT SELECT → DROP old)
         because SQLite does not support ALTER COLUMN or RENAME COLUMN in older versions.
PostgreSQL: simple RENAME COLUMN + ALTER COLUMN TYPE USING CASE expression.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db uv run python scripts/migrate_eligibility_columns.py
    DATABASE_URL=postgresql://... uv run python scripts/migrate_eligibility_columns.py
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, text

# (table, legacy_column_name, new_column_name)
TARGETS: list[tuple[str, str, str]] = [
    ("users", "user_disqualified", "eligibility"),
    ("clips", "disqualified", "eligibility"),
    ("cluster_runs", "disqualified", "eligibility"),
]


def _remap_sql(column: str) -> str:
    return (
        f"CASE "
        f"WHEN {column} IS NULL THEN 0 "
        f"WHEN {column} IN (0, false) THEN 1 "
        f"ELSE 2 END"
    )


def _pragma_columns(conn, table_name: str) -> list[dict]:
    """Return PRAGMA table_info rows as dicts: name, type, notnull, dflt_value, pk."""
    rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    return [
        {
            "cid": r[0],
            "name": r[1],
            "type": r[2],
            "notnull": r[3],
            "dflt_value": r[4],
            "pk": r[5],
        }
        for r in rows
    ]


def _sqlite_rebuild(
    conn,
    table_name: str,
    legacy_col: str,
    new_col: str,
) -> None:
    """Rebuild a SQLite table, renaming legacy_col → new_col and remapping values.

    Uses PRAGMA table_info on the old table to reconstruct the DDL so that
    only columns that actually existed are included — this allows the migration
    to work against minimal legacy schemas (e.g. in tests) as well as the full
    production schema.
    """
    old_name = f"{table_name}_old"
    conn.execute(text(f"ALTER TABLE {table_name} RENAME TO {old_name}"))

    # Read the legacy column list.
    old_col_defs = _pragma_columns(conn, old_name)

    # Build a CREATE TABLE that mirrors the old table but renames legacy_col
    # to new_col, adds NOT NULL + DEFAULT 0 on that column, and preserves
    # everything else.
    col_ddl_parts: list[str] = []
    for col in old_col_defs:
        name = col["name"]
        if name == legacy_col:
            col_ddl_parts.append(f"{new_col} INTEGER NOT NULL DEFAULT 0")
        else:
            col_type = col["type"] or "TEXT"
            null_part = "NOT NULL" if col["notnull"] else ""
            dflt = (
                f"DEFAULT {col['dflt_value']}" if col["dflt_value"] is not None else ""
            )
            pk_part = "PRIMARY KEY" if col["pk"] else ""
            parts = [name, col_type, pk_part, null_part, dflt]
            col_ddl_parts.append(" ".join(p for p in parts if p))

    create_ddl = f"CREATE TABLE {table_name} ({', '.join(col_ddl_parts)})"
    conn.execute(text(create_ddl))

    # Build INSERT SELECT, remapping the legacy column value.
    dest_cols: list[str] = []
    src_exprs: list[str] = []
    for col in old_col_defs:
        name = col["name"]
        if name == legacy_col:
            dest_cols.append(new_col)
            src_exprs.append(_remap_sql(legacy_col))
        else:
            dest_cols.append(name)
            src_exprs.append(name)

    cols_sql = ", ".join(dest_cols)
    vals_sql = ", ".join(src_exprs)
    conn.execute(
        text(f"INSERT INTO {table_name} ({cols_sql}) SELECT {vals_sql} FROM {old_name}")
    )
    conn.execute(text(f"DROP TABLE {old_name}"))


def _migrate_sqlite(conn) -> None:
    print("SQLite detected — rebuilding tables to rename columns and remap values...")

    _sqlite_rebuild(conn, "users", "user_disqualified", "eligibility")
    print("  OK: users.user_disqualified → users.eligibility")

    _sqlite_rebuild(conn, "clips", "disqualified", "eligibility")
    print("  OK: clips.disqualified → clips.eligibility")

    _sqlite_rebuild(conn, "cluster_runs", "disqualified", "eligibility")
    print("  OK: cluster_runs.disqualified → cluster_runs.eligibility")

    print("Migration complete.")


def _migrate_postgres(conn) -> None:
    print("PostgreSQL detected — altering columns in place...")
    for table, old_name, new_name in TARGETS:
        conn.execute(
            text(f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}")
        )
        conn.execute(
            text(
                f"ALTER TABLE {table} ALTER COLUMN {new_name} "
                f"TYPE INTEGER USING {_remap_sql(new_name)}"
            )
        )
        conn.execute(
            text(f"UPDATE {table} SET {new_name} = 0 WHERE {new_name} IS NULL")
        )
        conn.execute(text(f"ALTER TABLE {table} ALTER COLUMN {new_name} SET DEFAULT 0"))
        conn.execute(text(f"ALTER TABLE {table} ALTER COLUMN {new_name} SET NOT NULL"))
        print(f"  OK: {table}.{old_name} → {table}.{new_name}")
    print("Migration complete.")


def migrate_database(engine) -> None:
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            _migrate_sqlite(conn)
        elif engine.dialect.name == "postgresql":
            _migrate_postgres(conn)
        else:
            raise SystemExit(f"Unsupported dialect: {engine.dialect.name}")


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("Set DATABASE_URL environment variable.", file=sys.stderr)
        raise SystemExit(1)
    migrate_database(create_engine(url))


if __name__ == "__main__":
    main()

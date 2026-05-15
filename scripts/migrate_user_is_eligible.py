"""Production migration: rename users.eligibility (INTEGER) to users.is_eligible (BOOLEAN).

Value mapping:
    0 (PENDING)      → NULL
    1 (ELIGIBLE)     → 1 / True
    2 (DISQUALIFIED) → 0 / False

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db uv run python scripts/migrate_user_is_eligible.py
    DATABASE_URL=postgresql://... uv run python scripts/migrate_user_is_eligible.py
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

    with engine.begin() as conn:
        dialect = engine.dialect.name
        if dialect == "sqlite":
            _sqlite_migrate(conn)
        elif dialect == "postgresql":
            _postgres_migrate(conn)
        else:
            raise RuntimeError(f"Unsupported dialect: {dialect}")


def _sqlite_migrate(conn) -> None:
    cols = {r[1] for r in conn.execute(text("PRAGMA table_info(users)")).fetchall()}
    if "eligibility" not in cols:
        print("  SKIP: users.eligibility not found (already migrated?)")
        return
    if "is_eligible" in cols:
        print("  SKIP: users.is_eligible already exists")
        return

    conn.execute(text("ALTER TABLE users RENAME TO users_old"))

    pragma = conn.execute(text("PRAGMA table_info(users_old)")).fetchall()
    col_defs: list[str] = []
    for row in pragma:
        name, typ, notnull, dflt, pk = row[1], row[2], row[3], row[4], row[5]
        if name == "eligibility":
            # Use INTEGER for SQLite; will be treated as BOOLEAN by ORM
            col_defs.append("is_eligible INTEGER")
        else:
            pk_str = " PRIMARY KEY" if pk else ""
            nn_str = " NOT NULL" if notnull else ""
            df_str = f" DEFAULT {dflt}" if dflt is not None else ""
            col_defs.append(f"{name} {typ}{pk_str}{nn_str}{df_str}")

    conn.execute(text(f"CREATE TABLE users ({', '.join(col_defs)})"))

    old_names = [r[1] for r in pragma]
    new_names = ["is_eligible" if c == "eligibility" else c for c in old_names]
    select_parts: list[str] = []
    for col in old_names:
        if col == "eligibility":
            select_parts.append(
                "CASE eligibility WHEN 1 THEN 1 WHEN 2 THEN 0 ELSE NULL END"
            )
        else:
            select_parts.append(col)

    conn.execute(
        text(
            f"INSERT INTO users ({', '.join(new_names)}) "
            f"SELECT {', '.join(select_parts)} FROM users_old"
        )
    )
    conn.execute(text("DROP TABLE users_old"))
    print("  OK: users.eligibility → users.is_eligible (0→NULL, 1→True, 2→False)")


def _postgres_migrate(conn) -> None:
    result = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='users' AND column_name='eligibility'"
        )
    ).fetchone()
    if result is None:
        print("  SKIP: users.eligibility not found")
        return

    conn.execute(text("ALTER TABLE users ADD COLUMN is_eligible BOOLEAN"))
    conn.execute(
        text(
            "UPDATE users SET is_eligible = "
            "CASE eligibility WHEN 1 THEN TRUE WHEN 2 THEN FALSE ELSE NULL END"
        )
    )
    conn.execute(text("ALTER TABLE users DROP COLUMN eligibility"))
    print("  OK: users.eligibility → users.is_eligible")


if __name__ == "__main__":
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    migrate_database(database_url)

"""One-shot migration: add Clip.is_uploaded column.

Adds the column if missing; leaves all existing rows NULL (treated as
"not yet uploaded" by the Upload pipeline stage). Re-running is a no-op.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db \\
        uv run python scripts/migrate_clip_is_uploaded.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TABLE = "clips"
NEW_COLUMN = "is_uploaded"


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    engine = create_engine(database_url)
    inspector = inspect(engine)
    if TABLE not in inspector.get_table_names():
        print(f"Table {TABLE!r} does not exist — nothing to migrate.")
        return
    existing = {col["name"] for col in inspector.get_columns(TABLE)}
    if NEW_COLUMN in existing:
        print(f"  SKIP: {TABLE}.{NEW_COLUMN} already present")
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {NEW_COLUMN} BOOLEAN"))
    print(f"  OK: added {TABLE}.{NEW_COLUMN}")


if __name__ == "__main__":
    main()

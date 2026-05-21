"""Production migration: create the audio_mir table.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db \
      uv run python scripts/migrate_audio_mir.py
    DATABASE_URL=postgresql://... \
      uv run python scripts/migrate_audio_mir.py
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, inspect, make_url
from sqlalchemy.engine import Engine

from core.database import AudioMIR


def migrate_database(database_url_or_engine: str | Engine) -> None:
    if isinstance(database_url_or_engine, Engine):
        engine = database_url_or_engine
    else:
        engine = create_engine(database_url_or_engine)
    if inspect(engine).has_table("audio_mir"):
        print("  SKIP: audio_mir already exists")
        return
    AudioMIR.__table__.create(engine)
    print("  OK: created audio_mir")


def main() -> int:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 1
    safe_url = make_url(url).render_as_string(hide_password=True)
    print(f"Migrating {safe_url} ...")
    migrate_database(url)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

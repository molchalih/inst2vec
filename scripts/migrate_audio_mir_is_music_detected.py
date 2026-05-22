"""Production migration: add audio_mir.is_music_detected column.

Idempotent: skips if the column already exists. After running, the next MIR
fingerprint drift (new MirSettings.music_min_confidence/music_min_margin
fields enter ``_MIR_CONFIG_FIELDS``) will null and re-extract existing rows,
populating the new column.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db \\
      uv run python scripts/migrate_audio_mir_is_music_detected.py
    DATABASE_URL=postgresql://... \\
      uv run python scripts/migrate_audio_mir_is_music_detected.py
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, inspect, make_url, text
from sqlalchemy.engine import Engine


def migrate_database(database_url_or_engine: str | Engine) -> bool:
    """Returns True if the column was added, False if already present."""
    if isinstance(database_url_or_engine, Engine):
        engine = database_url_or_engine
    else:
        engine = create_engine(database_url_or_engine)
    cols = {c["name"] for c in inspect(engine).get_columns("audio_mir")}
    if "is_music_detected" in cols:
        print("  SKIP: audio_mir.is_music_detected already exists")
        return False
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE audio_mir ADD COLUMN is_music_detected BOOLEAN"))
    print("  OK: added audio_mir.is_music_detected")
    return True


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

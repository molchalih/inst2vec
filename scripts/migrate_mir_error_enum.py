"""Production migration: validate AudioMIR.mir_error against the new enum.

Any value not in {maest, effnet, audio_load, no_audio_file, NULL} is logged,
NULLed, and the row's is_mir_extracted flag is cleared so the next pipeline
pass re-runs that clip.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db \
      uv run python scripts/migrate_mir_error_enum.py
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, make_url, text


def migrate_database(database_url: str) -> int:
    """Returns the count of rows reset to NULL."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT clip_id, mir_error FROM audio_mir "
                "WHERE mir_error IS NOT NULL "
                "AND mir_error NOT IN ('maest','effnet','audio_load','no_audio_file')"
            )
        ).all()
        for clip_id, mir_error in rows:
            print(f"  RESET: clip_id={clip_id} mir_error={mir_error!r}")
        if not rows:
            return 0
        conn.execute(
            text(
                "UPDATE audio_mir SET mir_error = NULL, is_mir_extracted = NULL "
                "WHERE mir_error IS NOT NULL "
                "AND mir_error NOT IN ('maest','effnet','audio_load','no_audio_file')"
            )
        )
    return len(rows)


def main() -> int:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 1
    safe_url = make_url(url).render_as_string(hide_password=True)
    print(f"Validating mir_error vocabulary in {safe_url} ...")
    n = migrate_database(url)
    print(f"Done. {n} rows reset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
